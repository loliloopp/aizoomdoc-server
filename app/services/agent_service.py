"""
Сервис агента - оркестратор пайплайна обработки запросов.
"""

import json
import logging
import os
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from pathlib import Path
from uuid import UUID
from datetime import datetime
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image

from app.config import settings
from app.models.internal import UserWithSettings, SearchResult
from app.models.api import StreamEvent, PhaseStartedEvent, PhaseProgressEvent, LLMTokenEvent, ToolCallEvent
from app.models.llm_schemas import (
    AnswerResponse,
    FlashCollectorResponse,
    SelectedBlock,
    ImageRequest,
    ROIRequest,
    MaterialsJSON,
    MaterialImage,
)
from app.db.supabase_client import SupabaseClient
from app.db.supabase_projects_client import SupabaseProjectsClient
from app.db.s3_client import S3Client
from app.services.llm_service import create_llm_service
from app.services.llm_logger import LLMDialogLogger
from app.services.search_service import SearchService
from app.services.evidence_service import EvidenceService
from app.services.html_ocr_service import HtmlOcrService

logger = logging.getLogger(__name__)

# Путь к локальным промптам
PROMPTS_DIR = Path(__file__).parent.parent.parent / "data" / "promts"


def load_prompt(name: str) -> str:
    """Загрузить промпт из файла."""
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    logger.warning(f"Prompt file not found: {prompt_file}")
    return ""


class AgentService:
    """Сервис агента для обработки запросов пользователя."""
    
    def __init__(
        self,
        user: UserWithSettings,
        supabase: SupabaseClient,
        projects_db: SupabaseProjectsClient,
        s3_client: S3Client
    ):
        """
        Инициализация сервиса агента.
        
        Args:
            user: Пользователь с настройками
            supabase: Клиент основной БД
            projects_db: Клиент Projects DB
            s3_client: Клиент S3
        """
        self.user = user
        self.supabase = supabase
        self.projects_db = projects_db
        self.s3_client = s3_client
        
        # Инициализация сервисов
        self.llm_service = create_llm_service(user)
        self.search_service = SearchService(projects_db, s3_client)
        self.evidence_service = EvidenceService()
    
    async def process_message(
        self,
        chat_id: UUID,
        user_message: str,
        client_id: Optional[str] = None,
        document_ids: Optional[List[UUID]] = None,
        compare_document_ids_a: Optional[List[UUID]] = None,
        compare_document_ids_b: Optional[List[UUID]] = None,
        google_file_uris: Optional[List[str]] = None,
        save_user_message: bool = True
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Обработать сообщение пользователя с стримингом событий.
        
        Пайплайн:
        1. Поиск в документах (search)
        2. Сбор контекста (processing)
        3. Генерация ответа LLM (llm)
        4. Обработка tool calls (zoom, request_images)
        5. Финальный ответ
        
        Args:
            chat_id: ID чата
            user_message: Сообщение пользователя
            client_id: ID клиента для поиска документов
            document_ids: ID документов для контекста
            google_file_uris: URI файлов из Google File API
        
        Yields:
            События стриминга
        """
        llm_logger: Optional[LLMDialogLogger] = None
        try:
            llm_logger = LLMDialogLogger(str(chat_id))
            llm_logger.log_section("USER MESSAGE", user_message)
            llm_logger.log_section(
                "REQUEST CONTEXT",
                {
                    "client_id": client_id,
                    "document_ids": [str(x) for x in (document_ids or [])],
                    "compare_document_ids_a": [str(x) for x in (compare_document_ids_a or [])],
                    "compare_document_ids_b": [str(x) for x in (compare_document_ids_b or [])],
                    "google_files": google_file_uris or [],
                },
            )
            if self._has_html_files(google_file_uris):
                llm_logger.log_line("HTML attachment detected (text/html).")
            # Сохраняем сообщение пользователя в БД (если нужно)
            if save_user_message:
                await self.supabase.add_message(
                    chat_id=chat_id,
                    role="user",
                    content=user_message
                )
            
            context_text = ""

            if compare_document_ids_a and compare_document_ids_b:
                async for event in self._process_compare_mode(
                    chat_id=chat_id,
                    user_message=user_message,
                    document_ids_a=compare_document_ids_a,
                    document_ids_b=compare_document_ids_b,
                    llm_logger=llm_logger,
                ):
                    yield event
                yield StreamEvent(
                    event="completed",
                    data={"message": "Обработка завершена"},
                    timestamp=datetime.utcnow()
                )
                return

            # Фаза 1: Сбор контекста документов (если есть)
            if document_ids:
                yield self._create_phase_event("processing", "Загрузка документов...")
                context_text = await self._build_document_context(document_ids)
                yield self._create_progress_event("processing", 1.0, "Документы загружены")
            elif client_id:
                # Фаза 1: Поиск в документах
                yield self._create_phase_event("search", "Поиск в документах...")
                search_result = await self.search_service.search_in_documents(
                    query=user_message,
                    client_id=client_id
                )
                yield self._create_progress_event(
                    "search",
                    1.0,
                    f"Найдено {search_result.total_blocks_found} блоков"
                )
                
                # Фаза 2: Обработка и сбор контекста
                yield self._create_phase_event("processing", "Подготовка контекста...")
                context_text = self._format_search_context(search_result)
                yield self._create_progress_event("processing", 1.0, "Контекст подготовлен")
            else:
                context_text = ""
            
            # Фаза 3: Генерация ответа
            yield self._create_phase_event("llm", "Генерация ответа...")
            
            # Выбор режима (simple или complex)
            if self.user.settings.model_profile == "simple":
                async for event in self._process_simple_mode(
                    chat_id,
                    user_message,
                    context_text,
                    document_ids=document_ids,
                    client_id=client_id,
                    google_file_uris=google_file_uris,
                    llm_logger=llm_logger,
                ):
                    yield event
            else:  # complex
                async for event in self._process_complex_mode(
                    chat_id, user_message, context_text, client_id,
                    document_ids=document_ids,
                    google_file_uris=google_file_uris,
                    llm_logger=llm_logger,
                ):
                    yield event
            
            # Завершение
            yield StreamEvent(
                event="completed",
                data={"message": "Обработка завершена"},
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            logger.error(f"Error in process_message: {e}", exc_info=True)
            if llm_logger:
                try:
                    llm_logger.log_section("ERROR", str(e))
                except Exception:
                    pass
            yield StreamEvent(
                event="error",
                data={"message": str(e)},
                timestamp=datetime.utcnow()
            )

    async def _build_document_payloads(self, document_ids: List[UUID]) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for doc_id in document_ids:
            node = await self.projects_db.get_node_by_id(doc_id)
            doc_name = node.get("name") if node else str(doc_id)
            files = await self.projects_db.get_document_results(doc_id)

            md_text = await self._download_text_file(files, "result_md")
            html_text = await self._download_text_file(files, "ocr_html")
            if html_text:
                html_text = self._normalize_html_text(html_text)
            full_parts: List[str] = []
            if md_text:
                full_parts.append(md_text)
            if html_text:
                full_parts.append(html_text)
            full_text = "\n\n".join(full_parts).strip()

            blocks = self.search_service.parse_md_blocks(md_text or "")
            block_map = self.search_service.build_block_map(blocks)

            payloads.append(
                {
                    "doc_id": str(doc_id),
                    "doc_name": doc_name,
                    "full_text": full_text,
                    "block_map": block_map,
                    "blocks": blocks,
                }
            )
        return payloads

    async def _download_text_file(self, files: List[dict], file_type: str) -> str:
        target = next((f for f in files if f.get("file_type") == file_type), None)
        if not target:
            return ""
        key = target.get("r2_key")
        if not key:
            return ""
        data = await self.s3_client.download_bytes(key)
        if not data:
            return ""
        return data.decode("utf-8", errors="ignore")

    async def _build_html_crop_map(self, google_file_uris: Optional[List[Any]]) -> Dict[str, str]:
        if not google_file_uris:
            return {}
        crop_map: Dict[str, str] = {}
        for item in google_file_uris:
            if not isinstance(item, dict):
                continue
            mime = (item.get("mime_type") or "").lower()
            storage_path = item.get("storage_path")
            if "text/html" not in mime or not storage_path:
                continue
            data = await self._download_bytes(storage_path)
            if not data:
                logger.warning(f"Failed to load HTML from storage_path={storage_path}")
                continue
            html_text = data.decode("utf-8", errors="ignore")
            html_map = HtmlOcrService.extract_image_map(html_text)
            crop_map.update(html_map)
        return crop_map

    async def _download_crop_bytes(self, source: str) -> Optional[bytes]:
        if not source:
            return None
        if source.startswith("http://") or source.startswith("https://"):
            return await self._download_public(source)
        return await self._download_bytes(source)

    def _is_pdf_bytes(self, data: bytes) -> bool:
        return data[:4] == b"%PDF"

    def _combine_document_texts(self, payloads: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for payload in payloads:
            if payload.get("full_text"):
                header = f"=== DOCUMENT: {payload.get('doc_name')} ({payload.get('doc_id')}) ==="
                parts.append(header)
                parts.append(payload["full_text"])
        return "\n\n".join(parts).strip()

    def _normalize_html_text(self, html_text: str) -> str:
        # Basic HTML cleanup for OCR files.
        cleaned = re.sub(r"<[^>]+>", " ", html_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _has_html_files(self, google_file_uris: Optional[List[Any]]) -> bool:
        if not google_file_uris:
            return False
        for item in google_file_uris:
            if isinstance(item, dict):
                mime = (item.get("mime_type") or "").lower()
                uri = (item.get("uri") or "").lower()
            else:
                mime = ""
                uri = str(item).lower()
            if "text/html" in mime or uri.endswith(".html") or ".html?" in uri:
                return True
        return False

    def _compose_system_prompt(self, base_prompt: str, google_file_uris: Optional[List[Any]]) -> str:
        html_prompt = load_prompt("html_ocr_prompt")
        if html_prompt and self._has_html_files(google_file_uris):
            return f"{base_prompt}\n\n{html_prompt}"
        return base_prompt

    def _html_attachment_note(self, google_file_uris: Optional[List[Any]]) -> str:
        if self._has_html_files(google_file_uris):
            return "NOTE: HTML OCR file is attached via Google File API. Use its content as the main text source."
        return ""

    def _format_materials_prompt(self, materials_json: dict, user_message: str) -> str:
        return (
            "MATERIALS_JSON:\n"
            f"{json.dumps(materials_json, ensure_ascii=False)}\n\n"
            f"USER QUESTION:\n{user_message}"
        )

    def _apply_coverage_check(
        self,
        flash_response: FlashCollectorResponse,
        block_map: Dict[str, Any],
        query: str,
        max_add: int = 10,
    ) -> FlashCollectorResponse:
        selected_ids = {b.block_id for b in flash_response.selected_blocks}
        linked_ids = self.search_service.find_linked_blocks(selected_ids, block_map)
        preferred_pages = {b.page_number for b in flash_response.selected_blocks if b.page_number}

        all_blocks = list(block_map.values())
        extra_blocks = self.search_service.suggest_additional_blocks(
            blocks=all_blocks,
            selected_ids=selected_ids | linked_ids,
            query=query,
            preferred_pages=preferred_pages,
            max_add=max_add,
        )

        new_blocks: Dict[str, SelectedBlock] = {b.block_id: b for b in flash_response.selected_blocks}

        def add_block_from_map(block_id: str) -> None:
            if block_id in new_blocks:
                return
            block = block_map.get(block_id)
            if not block:
                return
            new_blocks[block_id] = SelectedBlock(
                block_id=block.block_id,
                block_kind=block.block_kind,
                page_number=max(1, block.page_number),
                content_raw=block.content_raw,
                linked_block_ids=block.linked_block_ids,
            )

        for block_id in linked_ids:
            add_block_from_map(block_id)
        for block in extra_blocks:
            add_block_from_map(block.block_id)

        requested_images = {r.block_id for r in flash_response.requested_images}
        for block in new_blocks.values():
            if block.block_kind == "IMAGE" and block.block_id not in requested_images:
                flash_response.requested_images.append(
                    ImageRequest(block_id=block.block_id, reason="coverage-check", priority="medium")
                )
                requested_images.add(block.block_id)

        flash_response.selected_blocks = list(new_blocks.values())
        return flash_response

    async def _build_materials(
        self,
        *,
        document_ids: List[UUID],
        selected_blocks: List[SelectedBlock],
        requested_images: List[ImageRequest],
        requested_rois: List[ROIRequest],
        block_map: Dict[str, Any],
        existing_materials: Optional[dict] = None,
        llm_logger: Optional[LLMDialogLogger] = None,
        html_crop_map: Optional[Dict[str, str]] = None,
    ) -> tuple[dict, List[dict]]:
        blocks_by_id: Dict[str, SelectedBlock] = {b.block_id: b for b in selected_blocks}
        for req in requested_images:
            if req.block_id in blocks_by_id:
                continue
            parsed = block_map.get(req.block_id)
            if parsed:
                blocks_by_id[req.block_id] = SelectedBlock(
                    block_id=parsed.block_id,
                    block_kind=parsed.block_kind,
                    page_number=parsed.page_number,
                    content_raw=parsed.content_raw,
                    linked_block_ids=parsed.linked_block_ids,
                )

        materials_images: List[MaterialImage] = []
        google_files: List[dict] = []
        seen_keys: set[tuple] = set()
        if existing_materials:
            try:
                existing = MaterialsJSON.model_validate(existing_materials)
                for img in existing.images:
                    key = (img.block_id, img.kind, tuple(img.bbox_norm) if img.bbox_norm else None)
                    seen_keys.add(key)
                    google_files.append({"uri": img.png_uri, "mime_type": "image/png"})
            except Exception:
                pass

        async def upload_render(block_id: str, render) -> Optional[MaterialImage]:
            key = (block_id, render.kind, tuple(render.bbox_norm) if render.bbox_norm else None)
            if key in seen_keys:
                return None
            seen_keys.add(key)
            file_name = f"{block_id}_{render.kind}"
            google_file = await self._upload_png_to_google(render.png_bytes, file_name)
            if not google_file:
                return None
            google_files.append(google_file)
            if llm_logger:
                llm_logger.log_section(
                    "UPLOAD_PNG",
                    {
                        "block_id": block_id,
                        "kind": render.kind,
                        "bbox_norm": render.bbox_norm,
                        "uri": google_file.get("uri"),
                    },
                )
            return MaterialImage(
                block_id=block_id,
                kind=render.kind,
                png_uri=google_file["uri"],
                width=render.width,
                height=render.height,
                scale_factor=render.scale_factor,
                bbox_norm=render.bbox_norm,
            )

        for req in requested_images:
            crop = await self._find_crop_by_image_id(req.block_id, document_ids)
            pdf_bytes = None
            cache_key = None
            if crop:
                r2_key = crop.get("r2_key")
                if r2_key:
                    pdf_bytes = await self.s3_client.download_bytes(r2_key)
                    cache_key = r2_key
            if not pdf_bytes and html_crop_map:
                crop_url = html_crop_map.get(req.block_id)
                if crop_url:
                    pdf_bytes = await self._download_crop_bytes(crop_url)
                    cache_key = crop_url
                    if llm_logger:
                        llm_logger.log_section(
                            "HTML_CROP_MAP",
                            {"block_id": req.block_id, "crop_url": crop_url},
                        )

            if not pdf_bytes:
                logger.warning(f"Crop not found for image_id: {req.block_id}")
                if llm_logger:
                    llm_logger.log_section("MISSING_CROP", {"block_id": req.block_id})
                continue
            if not self._is_pdf_bytes(pdf_bytes):
                logger.warning(f"Non-PDF crop for image_id: {req.block_id}")
                if llm_logger:
                    llm_logger.log_section("NON_PDF_CROP", {"block_id": req.block_id})
                continue

            renders = self.evidence_service.build_preview_and_quadrants(
                pdf_bytes, cache_key=cache_key or req.block_id, page=0, dpi=150
            )
            for render in renders:
                material = await upload_render(req.block_id, render)
                if material:
                    materials_images.append(material)

        for roi in requested_rois:
            crop = await self._find_crop_by_image_id(roi.block_id, document_ids)
            pdf_bytes = None
            cache_key = None
            if crop:
                r2_key = crop.get("r2_key")
                if r2_key:
                    pdf_bytes = await self.s3_client.download_bytes(r2_key)
                    cache_key = r2_key
            if not pdf_bytes and html_crop_map:
                crop_url = html_crop_map.get(roi.block_id)
                if crop_url:
                    pdf_bytes = await self._download_crop_bytes(crop_url)
                    cache_key = crop_url
                    if llm_logger:
                        llm_logger.log_section(
                            "HTML_CROP_MAP",
                            {"block_id": roi.block_id, "crop_url": crop_url},
                        )

            if not pdf_bytes:
                logger.warning(f"Crop not found for ROI: {roi.block_id}")
                if llm_logger:
                    llm_logger.log_section("MISSING_CROP", {"block_id": roi.block_id})
                continue
            if not self._is_pdf_bytes(pdf_bytes):
                logger.warning(f"Non-PDF crop for ROI: {roi.block_id}")
                if llm_logger:
                    llm_logger.log_section("NON_PDF_CROP", {"block_id": roi.block_id})
                continue
            # Crop PDFs are single-page extracts, always use page 0
            # (roi.page refers to the original document page, not the crop)
            page_index = 0
            dpi = roi.dpi or 300
            if dpi < 72:
                dpi = 72
            if dpi > 400:
                dpi = 400
            render = self.evidence_service.build_roi(
                pdf_bytes, cache_key=cache_key or roi.block_id, bbox_norm=roi.bbox_norm, page=page_index, dpi=dpi
            )
            material = await upload_render(roi.block_id, render)
            if material:
                materials_images.append(material)

        if existing_materials:
            existing = MaterialsJSON.model_validate(existing_materials)
            materials_images = existing.images + materials_images
            blocks_by_id = {b.block_id: b for b in existing.blocks} | blocks_by_id

        materials = MaterialsJSON(
            blocks=list(blocks_by_id.values()),
            images=materials_images,
            source_documents=[str(doc_id) for doc_id in document_ids],
        )
        return materials.model_dump(), google_files
    
    async def _process_simple_mode(
        self,
        chat_id: UUID,
        user_message: str,
        context_text: str,
        document_ids: Optional[List[UUID]] = None,
        client_id: Optional[str] = None,
        google_file_uris: Optional[List[str]] = None,
        llm_logger: Optional[LLMDialogLogger] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Обработка в simple (flash-only) режиме со строгим JSON."""

        system_prompt = load_prompt("flash_answer_prompt") or load_prompt("llm_system_prompt")
        if not system_prompt:
            system_prompt = await self.llm_service.load_system_prompts(self.supabase)
        system_prompt = self._compose_system_prompt(system_prompt, google_file_uris)

        payloads = await self._build_document_payloads(document_ids or [])
        full_context = self._combine_document_texts(payloads) or context_text
        html_note = self._html_attachment_note(google_file_uris)
        html_crop_map = await self._build_html_crop_map(google_file_uris)
        if llm_logger and html_crop_map:
            llm_logger.log_section("HTML_CROP_MAP_SIZE", {"count": len(html_crop_map)})
        block_map: Dict[str, Any] = {}
        for payload in payloads:
            block_map.update(payload.get("block_map", {}))

        max_iterations = 5
        iteration = 0
        materials_json: Optional[dict] = None
        google_files: List[dict] = list(google_file_uris) if google_file_uris else []
        final_answer: Optional[AnswerResponse] = None

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Flash-only iteration {iteration}")
            if materials_json:
                user_prompt = f"{full_context}\n\n{html_note}\n\n{self._format_materials_prompt(materials_json, user_message)}"
            else:
                user_prompt = f"{full_context}\n\n{html_note}\n\nUSER QUESTION:\n{user_message}"

            if llm_logger:
                llm_logger.log_request(
                    phase=f"simple_answer_{iteration}",
                    model=settings.default_flash_model or settings.default_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    google_files=google_files or [],
                )

            answer_dict, raw_text = await self.llm_service.run_answer(
                system_prompt=system_prompt,
                user_message=user_prompt,
                google_file_uris=google_files if google_files else None,
                model_name=settings.default_flash_model or settings.default_model,
                return_text=True,
            )
            if llm_logger:
                llm_logger.log_response(phase=f"simple_answer_{iteration}", response_text=raw_text)
            try:
                answer = AnswerResponse.model_validate(answer_dict)
            except Exception as e:
                logger.error(f"Invalid AnswerResponse: {e}")
                if llm_logger:
                    llm_logger.log_section("VALIDATION_ERROR", str(e))
                raise

            if answer.followup_images or answer.followup_rois:
                if llm_logger:
                    llm_logger.log_section(
                        "FOLLOWUP_REQUESTS",
                        {
                            "followup_images": answer.followup_images,
                            "followup_rois": [r.model_dump() for r in answer.followup_rois],
                        },
                    )
                image_reqs = [ImageRequest(block_id=bid, reason="followup", priority="high") for bid in answer.followup_images]
                roi_reqs = [ROIRequest.model_validate(r) for r in answer.followup_rois]

                if image_reqs:
                    yield StreamEvent(
                        event="tool_call",
                        data=ToolCallEvent(
                            tool="request_images",
                            parameters={"image_ids": [r.block_id for r in image_reqs]},
                            reason="followup_images"
                        ).dict(),
                        timestamp=datetime.utcnow()
                    )
                if roi_reqs:
                    yield StreamEvent(
                        event="tool_call",
                        data=ToolCallEvent(
                            tool="zoom",
                            parameters={"count": len(roi_reqs)},
                            reason="followup_rois"
                        ).dict(),
                        timestamp=datetime.utcnow()
                    )

                yield StreamEvent(
                    event="phase_started",
                    data={"phase": "tool_execution", "description": "Подготовка PNG изображений..."},
                    timestamp=datetime.utcnow()
                )

                materials_json, google_files = await self._build_materials(
                    document_ids=document_ids or [],
                    selected_blocks=[],
                    requested_images=image_reqs,
                    requested_rois=roi_reqs,
                    block_map=block_map,
                    existing_materials=materials_json,
                    llm_logger=llm_logger,
                    html_crop_map=html_crop_map,
                )
                if llm_logger:
                    llm_logger.log_section("MATERIALS_JSON_UPDATE", materials_json)

                if not google_files:
                    logger.warning("No PNG files produced for followups")
                    final_answer = answer
                    break
                continue

            final_answer = answer
            break

        if final_answer is None:
            raise RuntimeError("Failed to obtain final answer")

        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=final_answer.answer_markdown,
        )

        yield StreamEvent(
            event="llm_final",
            data={"content": final_answer.answer_markdown, "answer_json": final_answer.model_dump()},
            timestamp=datetime.utcnow()
        )

    async def _process_compare_mode(
        self,
        *,
        chat_id: UUID,
        user_message: str,
        document_ids_a: List[UUID],
        document_ids_b: List[UUID],
        llm_logger: Optional[LLMDialogLogger] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Compare two document sets (Flash collector -> Pro answer)."""

        flash_prompt = load_prompt("flash_extractor_prompt")
        if not flash_prompt:
            flash_prompt = await self.llm_service.load_system_prompts(self.supabase)
        flash_prompt = self._compose_system_prompt(flash_prompt, None)

        pro_prompt = load_prompt("pro_answer_prompt") or load_prompt("llm_system_prompt")
        if not pro_prompt:
            pro_prompt = await self.llm_service.load_system_prompts(self.supabase)
        pro_prompt = self._compose_system_prompt(pro_prompt, None)

        payloads_a = await self._build_document_payloads(document_ids_a)
        payloads_b = await self._build_document_payloads(document_ids_b)

        yield self._create_phase_event("flash_stage", "Flash собирает контекст для сравнения...")

        combined_blocks: List[SelectedBlock] = []
        combined_images: List[ImageRequest] = []
        combined_rois: List[ROIRequest] = []

        def label_block(block: SelectedBlock, label: str) -> SelectedBlock:
            return SelectedBlock(
                block_id=block.block_id,
                block_kind=block.block_kind,
                page_number=max(1, block.page_number),
                content_raw=f"[{label}]\n{block.content_raw}",
                linked_block_ids=block.linked_block_ids,
            )

        block_map: Dict[str, Any] = {}

        async def collect(payloads: List[Dict[str, Any]], label_prefix: str) -> None:
            for payload in payloads:
                full_text = payload.get("full_text") or ""
                user_prompt = f"{full_text}\n\nUSER QUESTION:\n{user_message}"
                if llm_logger:
                    llm_logger.log_request(
                        phase=f"flash_collect_{label_prefix}_{payload.get('doc_id')}",
                        model=settings.default_flash_model or settings.default_model,
                        system_prompt=flash_prompt,
                        user_prompt=user_prompt,
                        google_files=[],
                    )
                flash_dict, raw_text = await self.llm_service.run_flash_collector(
                    system_prompt=flash_prompt,
                    user_message=user_prompt,
                    model_name=settings.default_flash_model or settings.default_model,
                    return_text=True,
                )
                if llm_logger:
                    llm_logger.log_response(
                        phase=f"flash_collect_{label_prefix}_{payload.get('doc_id')}",
                        response_text=raw_text,
                    )
                flash_resp = FlashCollectorResponse.model_validate(flash_dict)
                flash_resp = self._apply_coverage_check(
                    flash_response=flash_resp,
                    block_map=payload.get("block_map", {}),
                    query=user_message,
                    max_add=10,
                )
                if llm_logger:
                    llm_logger.log_section(
                        "COVERAGE_CHECK",
                        {
                            "doc_id": payload.get("doc_id"),
                            "selected_blocks": len(flash_resp.selected_blocks),
                            "requested_images": len(flash_resp.requested_images),
                            "requested_rois": len(flash_resp.requested_rois),
                        },
                    )
                label = f"{label_prefix}: {payload.get('doc_name')} ({payload.get('doc_id')})"
                combined_blocks.extend([label_block(b, label) for b in flash_resp.selected_blocks])
                combined_images.extend(flash_resp.requested_images)
                combined_rois.extend(flash_resp.requested_rois)

                for block in payload.get("blocks", []):
                    block_map[block.block_id] = SelectedBlock(
                        block_id=block.block_id,
                        block_kind=block.block_kind,
                        page_number=max(1, block.page_number),
                        content_raw=f"[{label}]\n{block.content_raw}",
                        linked_block_ids=block.linked_block_ids,
                    )

        await collect(payloads_a, "DOC_A")
        await collect(payloads_b, "DOC_B")

        yield self._create_progress_event("flash_stage", 1.0, "Контекст собран")

        # Prepare materials
        yield self._create_phase_event("tool_execution", "Подготовка PNG изображений...")
        combined_doc_ids = document_ids_a + document_ids_b
        materials_json, google_files = await self._build_materials(
            document_ids=combined_doc_ids,
            selected_blocks=combined_blocks,
            requested_images=combined_images,
            requested_rois=combined_rois,
            block_map=block_map,
            llm_logger=llm_logger,
        )
        if llm_logger:
            llm_logger.log_section("MATERIALS_JSON", materials_json)

        # Pro answer
        yield self._create_phase_event("pro_stage", "Pro сравнивает документы...")
        max_iterations = 5
        iteration = 0
        final_answer: Optional[AnswerResponse] = None

        while iteration < max_iterations:
            iteration += 1
            compare_question = f"Compare DOC_A vs DOC_B. {user_message}"
            user_prompt = self._format_materials_prompt(materials_json, compare_question)
            if llm_logger:
                llm_logger.log_request(
                    phase=f"compare_pro_answer_{iteration}",
                    model=settings.default_pro_model or settings.default_model,
                    system_prompt=pro_prompt,
                    user_prompt=user_prompt,
                    google_files=google_files or [],
                )
            answer_dict, raw_text = await self.llm_service.run_answer(
                system_prompt=pro_prompt,
                user_message=user_prompt,
                google_file_uris=google_files if google_files else None,
                model_name=settings.default_pro_model or settings.default_model,
                return_text=True,
            )
            if llm_logger:
                llm_logger.log_response(phase=f"compare_pro_answer_{iteration}", response_text=raw_text)
            answer = AnswerResponse.model_validate(answer_dict)

            if answer.followup_images or answer.followup_rois:
                if llm_logger:
                    llm_logger.log_section(
                        "FOLLOWUP_REQUESTS",
                        {
                            "followup_images": answer.followup_images,
                            "followup_rois": [r.model_dump() for r in answer.followup_rois],
                        },
                    )
                image_reqs = [ImageRequest(block_id=bid, reason="followup", priority="high") for bid in answer.followup_images]
                roi_reqs = [ROIRequest.model_validate(r) for r in answer.followup_rois]

                yield self._create_phase_event("tool_execution", "Подготовка доп. PNG изображений...")
                materials_json, google_files = await self._build_materials(
                    document_ids=combined_doc_ids,
                    selected_blocks=combined_blocks,
                    requested_images=image_reqs,
                    requested_rois=roi_reqs,
                    block_map=block_map,
                    existing_materials=materials_json,
                    llm_logger=llm_logger,
                )
                if llm_logger:
                    llm_logger.log_section("MATERIALS_JSON_UPDATE", materials_json)
                continue

            final_answer = answer
            break

        if final_answer is None:
            raise RuntimeError("Failed to obtain final compare answer")

        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=final_answer.answer_markdown,
        )

        yield StreamEvent(
            event="llm_final",
            data={"content": final_answer.answer_markdown, "answer_json": final_answer.model_dump()},
            timestamp=datetime.utcnow()
        )
    
    async def _fetch_and_upload_images(
        self,
        image_ids: List[str],
        document_ids: Optional[List[UUID]]
    ) -> List[dict]:
        """Найти изображения по ID и загрузить PNG в Google File API."""
        uploaded_files = []
        
        for image_id in image_ids:
            try:
                crop = await self._find_crop_by_image_id(image_id, document_ids)
                if not crop:
                    logger.warning(f"Crop not found for image_id: {image_id}")
                    continue
                
                r2_key = crop.get("r2_key")
                if not r2_key:
                    continue
                
                # Скачиваем файл из S3/R2
                file_bytes = await self.s3_client.download_bytes(r2_key)
                if not file_bytes:
                    logger.warning(f"Failed to download: {r2_key}")
                    continue

                renders = self.evidence_service.build_preview_and_quadrants(
                    file_bytes, cache_key=r2_key, page=0, dpi=150
                )
                for render in renders:
                    google_file = await self._upload_png_to_google(render.png_bytes, f"{image_id}_{render.kind}")
                    if google_file:
                        uploaded_files.append(google_file)
                        logger.info(f"Uploaded image {image_id} to Google: {google_file.get('uri')}")
                    
            except Exception as e:
                logger.error(f"Error fetching image {image_id}: {e}")
        
        return uploaded_files
    
    async def _fetch_and_upload_zoom(
        self,
        image_id: Optional[str],
        document_ids: Optional[List[UUID]],
        coords: Optional[List[float]]
    ) -> List[dict]:
        """Создать zoom (PNG) и загрузить в Google File API."""
        if not image_id or not coords:
            return []
        
        try:
            crop = await self._find_crop_by_image_id(image_id, document_ids)
            if not crop:
                return []
            
            r2_key = crop.get("r2_key")
            if not r2_key:
                return []
            
            # Скачиваем файл
            file_bytes = await self.s3_client.download_bytes(r2_key)
            if not file_bytes:
                return []
            
            # Определяем тип файла
            is_pdf = r2_key.lower().endswith(".pdf")
            
            # Вырезаем zoom область
            zoom_bytes = await self._crop_image(file_bytes, coords, is_pdf)
            if not zoom_bytes:
                return []
            
            # Загружаем в Google
            zoom_name = f"zoom_{image_id}_{coords[0]:.2f}_{coords[1]:.2f}"
            google_file = await self._upload_png_to_google(zoom_bytes, zoom_name)
            if google_file:
                logger.info(f"Uploaded zoom to Google: {google_file.get('uri')}")
                return [google_file]
            
        except Exception as e:
            logger.error(f"Error creating zoom: {e}")
        
        return []
    
    async def _upload_to_google(self, file_bytes: bytes, name: str, mime_type: str) -> Optional[dict]:
        """Загрузить файл в Google File API."""
        try:
            from google import genai
            from app.config import settings
            
            api_key = self.user.gemini_api_key or settings.default_gemini_api_key
            if not api_key:
                return None
            
            client = genai.Client(api_key=api_key)
            
            # Сохраняем во временный файл
            import tempfile
            ext = ".png" if "image" in mime_type else ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                uploaded = client.files.upload(
                    file=tmp_path,
                    config={"display_name": name, "mime_type": mime_type}
                )
                return {"uri": uploaded.uri, "mime_type": mime_type}
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error uploading to Google: {e}")
            return None

    async def _upload_png_to_google(self, png_bytes: bytes, name: str) -> Optional[dict]:
        """Upload PNG only to Google File API."""
        return await self._upload_to_google(png_bytes, name, "image/png")
    
    async def _crop_image(self, file_bytes: bytes, coords: List[float], is_pdf: bool) -> Optional[bytes]:
        """Вырезать область из изображения/PDF."""
        try:
            if is_pdf:
                # Рендерим PDF в изображение
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            else:
                img = Image.open(BytesIO(file_bytes))
            
            # Вырезаем область
            w, h = img.size
            x1, y1, x2, y2 = coords
            
            # Нормализованные координаты
            if max(coords) <= 1.0:
                x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            
            cropped = img.crop((x1, y1, x2, y2))
            
            # Сохраняем в bytes
            output = BytesIO()
            cropped.save(output, format="PNG")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
            return None
    
    async def _process_complex_mode(
        self,
        chat_id: UUID,
        user_message: str,
        context_text: str,
        client_id: str,
        document_ids: Optional[List[UUID]] = None,
        google_file_uris: Optional[List[str]] = None,
        llm_logger: Optional[LLMDialogLogger] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Обработка в complex (flash+pro) режиме со строгим JSON."""

        flash_prompt = load_prompt("flash_extractor_prompt")
        if not flash_prompt:
            flash_prompt = await self.llm_service.load_system_prompts(self.supabase)

        pro_prompt = load_prompt("pro_answer_prompt") or load_prompt("llm_system_prompt")
        if not pro_prompt:
            pro_prompt = await self.llm_service.load_system_prompts(self.supabase)

        payloads = await self._build_document_payloads(document_ids or [])
        block_map: Dict[str, Any] = {}
        for payload in payloads:
            block_map.update(payload.get("block_map", {}))
        html_crop_map = await self._build_html_crop_map(google_file_uris)
        if llm_logger and html_crop_map:
            llm_logger.log_section("HTML_CROP_MAP_SIZE", {"count": len(html_crop_map)})

        # Этап 1: Flash collector по каждому документу
        yield self._create_phase_event("flash_stage", "Flash собирает контекст...")

        combined_blocks: List[SelectedBlock] = []
        combined_images: List[ImageRequest] = []
        combined_rois: List[ROIRequest] = []
        html_note = self._html_attachment_note(google_file_uris)

        if payloads:
            for payload in payloads:
                full_text = payload.get("full_text") or ""
                user_prompt = f"{full_text}\n\n{html_note}\n\nUSER QUESTION:\n{user_message}"
                if llm_logger:
                    llm_logger.log_request(
                        phase=f"flash_collect_{payload.get('doc_id')}",
                        model=settings.default_flash_model or settings.default_model,
                        system_prompt=flash_prompt,
                        user_prompt=user_prompt,
                        google_files=google_file_uris or [],
                    )
                flash_dict, raw_text = await self.llm_service.run_flash_collector(
                    system_prompt=flash_prompt,
                    user_message=user_prompt,
                    google_file_uris=google_file_uris,
                    model_name=settings.default_flash_model or settings.default_model,
                    return_text=True,
                )
                if llm_logger:
                    llm_logger.log_response(
                        phase=f"flash_collect_{payload.get('doc_id')}",
                        response_text=raw_text,
                    )
                flash_resp = FlashCollectorResponse.model_validate(flash_dict)
                flash_resp = self._apply_coverage_check(
                    flash_response=flash_resp,
                    block_map=payload.get("block_map", {}),
                    query=user_message,
                    max_add=10,
                )
                if llm_logger:
                    llm_logger.log_section(
                        "COVERAGE_CHECK",
                        {
                            "doc_id": payload.get("doc_id"),
                            "selected_blocks": len(flash_resp.selected_blocks),
                            "requested_images": len(flash_resp.requested_images),
                            "requested_rois": len(flash_resp.requested_rois),
                        },
                    )
                combined_blocks.extend(flash_resp.selected_blocks)
                combined_images.extend(flash_resp.requested_images)
                combined_rois.extend(flash_resp.requested_rois)
        else:
            # Fallback: use provided context_text
            user_prompt = f"{context_text}\n\n{html_note}\n\nUSER QUESTION:\n{user_message}"
            if llm_logger:
                llm_logger.log_request(
                    phase="flash_collect_fallback",
                    model=settings.default_flash_model or settings.default_model,
                    system_prompt=flash_prompt,
                    user_prompt=user_prompt,
                    google_files=google_file_uris or [],
                )
            flash_dict, raw_text = await self.llm_service.run_flash_collector(
                system_prompt=flash_prompt,
                user_message=user_prompt,
                google_file_uris=google_file_uris,
                model_name=settings.default_flash_model or settings.default_model,
                return_text=True,
            )
            if llm_logger:
                llm_logger.log_response(phase="flash_collect_fallback", response_text=raw_text)
            flash_resp = FlashCollectorResponse.model_validate(flash_dict)
            combined_blocks.extend(flash_resp.selected_blocks)
            combined_images.extend(flash_resp.requested_images)
            combined_rois.extend(flash_resp.requested_rois)

        yield self._create_progress_event("flash_stage", 1.0, "Контекст собран")

        # Этап 2: подготовка материалов (PNG-only)
        yield self._create_phase_event("tool_execution", "Подготовка PNG изображений...")
        materials_json, google_files = await self._build_materials(
            document_ids=document_ids or [],
            selected_blocks=combined_blocks,
            requested_images=combined_images,
            requested_rois=combined_rois,
            block_map=block_map,
            llm_logger=llm_logger,
            html_crop_map=html_crop_map,
        )
        if llm_logger:
            llm_logger.log_section("MATERIALS_JSON", materials_json)

        # Этап 3: Pro отвечает
        yield self._create_phase_event("pro_stage", "Pro формирует ответ...")
        max_iterations = 5
        iteration = 0
        final_answer: Optional[AnswerResponse] = None

        while iteration < max_iterations:
            iteration += 1
            user_prompt = self._format_materials_prompt(materials_json, user_message)
            if llm_logger:
                llm_logger.log_request(
                    phase=f"pro_answer_{iteration}",
                    model=settings.default_pro_model or settings.default_model,
                    system_prompt=pro_prompt,
                    user_prompt=user_prompt,
                    google_files=google_files or [],
                )
            answer_dict, raw_text = await self.llm_service.run_answer(
                system_prompt=pro_prompt,
                user_message=user_prompt,
                google_file_uris=google_files if google_files else None,
                model_name=settings.default_pro_model or settings.default_model,
                return_text=True,
            )
            if llm_logger:
                llm_logger.log_response(phase=f"pro_answer_{iteration}", response_text=raw_text)
            answer = AnswerResponse.model_validate(answer_dict)

            if answer.followup_images or answer.followup_rois:
                if llm_logger:
                    llm_logger.log_section(
                        "FOLLOWUP_REQUESTS",
                        {
                            "followup_images": answer.followup_images,
                            "followup_rois": [r.model_dump() for r in answer.followup_rois],
                        },
                    )
                image_reqs = [ImageRequest(block_id=bid, reason="followup", priority="high") for bid in answer.followup_images]
                roi_reqs = [ROIRequest.model_validate(r) for r in answer.followup_rois]

                if image_reqs:
                    yield StreamEvent(
                        event="tool_call",
                        data=ToolCallEvent(
                            tool="request_images",
                            parameters={"image_ids": [r.block_id for r in image_reqs]},
                            reason="followup_images"
                        ).dict(),
                        timestamp=datetime.utcnow()
                    )
                if roi_reqs:
                    yield StreamEvent(
                        event="tool_call",
                        data=ToolCallEvent(
                            tool="zoom",
                            parameters={"count": len(roi_reqs)},
                            reason="followup_rois"
                        ).dict(),
                        timestamp=datetime.utcnow()
                    )

                yield self._create_phase_event("tool_execution", "Подготовка доп. PNG изображений...")

                materials_json, google_files = await self._build_materials(
                    document_ids=document_ids or [],
                    selected_blocks=combined_blocks,
                    requested_images=image_reqs,
                    requested_rois=roi_reqs,
                    block_map=block_map,
                    existing_materials=materials_json,
                    llm_logger=llm_logger,
                    html_crop_map=html_crop_map,
                )
                if llm_logger:
                    llm_logger.log_section("MATERIALS_JSON_UPDATE", materials_json)
                continue

            final_answer = answer
            break

        if final_answer is None:
            raise RuntimeError("Failed to obtain final answer")

        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=final_answer.answer_markdown,
        )

        yield StreamEvent(
            event="llm_final",
            data={"content": final_answer.answer_markdown, "answer_json": final_answer.model_dump()},
            timestamp=datetime.utcnow()
        )
    
    def _format_search_context(self, search_result: SearchResult) -> str:
        """Форматировать результаты поиска в текстовый контекст."""
        context = f"НАЙДЕННЫЙ ТЕКСТ:\n\n"
        
        for i, block in enumerate(search_result.text_blocks, 1):
            context += f"=== БЛОК {i} ===\n"
            if block.block_id:
                context += f"ID: {block.block_id}\n"
            if block.page:
                context += f"Страница: {block.page}\n"
            context += f"{block.text}\n\n"
        
        return context

    async def _build_document_context(self, document_ids: List[UUID]) -> str:
        """Собрать контекст из MD/HTML файлов документа."""
        context_parts = []
        max_chars_per_file = 20000

        for doc_id in document_ids:
            node = await self.projects_db.get_node_by_id(doc_id)
            doc_name = node.get("name") if node else str(doc_id)
            context_parts.append(f"=== ДОКУМЕНТ: {doc_name} ({doc_id}) ===")

            files = await self.projects_db.get_document_results(doc_id)
            for f in files:
                file_type = f.get("file_type")
                if file_type not in ("result_md", "ocr_html"):
                    continue

                key = f.get("r2_key")
                if not key:
                    continue

                data = await self.s3_client.download_bytes(key)
                if not data:
                    # fallback: try public url
                    url = self._build_public_url(key)
                    if url:
                        data = await self._download_public(url)
                if not data:
                    continue

                text = data.decode("utf-8", errors="ignore")
                if file_type == "ocr_html":
                    # простая очистка HTML
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                if len(text) > max_chars_per_file:
                    text = text[:max_chars_per_file] + "\n[...TRUNCATED...]"

                label = "MD" if file_type == "result_md" else "HTML_OCR"
                context_parts.append(f"[{label}]:\n{text}\n")

            # Добавляем каталог изображений
            annotation = next((x for x in files if x.get("file_type") == "annotation"), None)
            if annotation and annotation.get("r2_key"):
                catalog = await self._build_image_catalog(annotation.get("r2_key"))
                if catalog:
                    context_parts.append("КАТАЛОГ ИЗОБРАЖЕНИЙ (block_id):\n" + catalog)

        if not context_parts:
            return ""

        return "\n".join(context_parts)

    async def _build_image_catalog(self, r2_key: str) -> str:
        """Собрать каталог block_id из annotation.json."""
        data = await self.s3_client.download_bytes(r2_key)
        if not data:
            url = self._build_public_url(r2_key)
            if url:
                data = await self._download_public(url)
        if not data:
            return ""

        import json
        try:
            payload = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            return ""

        lines = []
        pages = payload.get("pages", [])
        for page in pages:
            page_number = page.get("page_number") or page.get("page_index")
            for block in page.get("blocks", []):
                block_id = block.get("id") or block.get("block_id")
                if not block_id:
                    continue
                lines.append(f"- {block_id} (стр. {page_number})")

        return "\n".join(lines[:500])

    async def _handle_request_images(
        self,
        chat_id: UUID,
        image_ids: List[str],
        document_ids: Optional[List[UUID]]
    ) -> None:
        """Создать вложения на основе image_ids (crop)."""
        if not image_ids or not document_ids:
            return

        # Найти последнее сообщение ассистента
        msg = await self.supabase.get_last_message(chat_id, role="assistant")
        if not msg:
            msg = await self.supabase.add_message(
                chat_id=chat_id,
                role="assistant",
                content="Запрошенные изображения"
            )
            if not msg:
                return

        for image_id in image_ids:
            crop = await self._find_crop_by_image_id(image_id, document_ids)
            if not crop:
                continue

            r2_key = crop.get("r2_key")
            file_name = crop.get("file_name") or Path(r2_key).name
            mime = crop.get("mime_type") or ("application/pdf" if file_name.endswith(".pdf") else "image/png")

            # Создаём запись storage_files
            storage_file = await self.supabase.register_file(
                user_id=self.user.user.id,
                filename=file_name,
                mime_type=mime,
                size_bytes=crop.get("file_size") or 0,
                storage_path=r2_key,
                source_type="projects_crop"
            )

            # Создаём chat_images
            if storage_file:
                await self.supabase.add_chat_image(
                    chat_id=chat_id,
                    message_id=msg.id,
                    file_id=storage_file.id,
                    image_type="crop",
                    description=image_id
                )

    async def _handle_zoom(
        self,
        chat_id: UUID,
        image_id: Optional[str],
        document_ids: Optional[List[UUID]],
        coords_norm: Optional[List[float]],
        reason: str = ""
    ) -> None:
        """Обработка zoom: создаём увеличенный фрагмент по coords_norm."""
        if not image_id or not document_ids:
            return
        if not coords_norm or len(coords_norm) != 4:
            # fallback
            await self._handle_request_images(chat_id, [image_id], document_ids)
            return

        crop = await self._find_crop_by_image_id(image_id, document_ids)
        if not crop:
            return

        r2_key = crop.get("r2_key")
        if not r2_key:
            return

        data = await self._download_bytes(r2_key)
        if not data:
            return

        # Load image (pdf or image)
        img = None
        if str(r2_key).lower().endswith(".pdf"):
            try:
                doc = fitz.open(stream=data, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(BytesIO(pix.tobytes("png")))
            except Exception:
                img = None
        else:
            try:
                img = Image.open(BytesIO(data))
            except Exception:
                img = None

        if img is None:
            return

        w, h = img.size
        x1 = int(max(0, min(1, coords_norm[0])) * w)
        y1 = int(max(0, min(1, coords_norm[1])) * h)
        x2 = int(max(0, min(1, coords_norm[2])) * w)
        y2 = int(max(0, min(1, coords_norm[3])) * h)
        if x2 <= x1 or y2 <= y1:
            return

        crop_img = img.crop((x1, y1, x2, y2))
        out = BytesIO()
        crop_img.save(out, format="PNG")
        out_bytes = out.getvalue()

        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", image_id)
        zoom_key = f"chats/{chat_id}/images/zoom_{safe_id}.png"
        await self.s3_client.upload_bytes(out_bytes, zoom_key, content_type="image/png")

        storage_file = await self.supabase.register_file(
            user_id=self.user.user.id,
            filename=f"zoom_{safe_id}.png",
            mime_type="image/png",
            size_bytes=len(out_bytes),
            storage_path=zoom_key,
            source_type="llm_generated"
        )
        if storage_file:
            msg = await self.supabase.get_last_message(chat_id, role="assistant")
            if msg:
                await self.supabase.add_chat_image(
                    chat_id=chat_id,
                    message_id=msg.id,
                    file_id=storage_file.id,
                    image_type="zoom_crop",
                    description=reason or image_id,
                    width=crop_img.size[0],
                    height=crop_img.size[1]
                )

    async def _handle_request_documents(
        self,
        chat_id: UUID,
        document_names: List[str],
        user_message: str,
        original_context: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Запрос дополнительных документов и генерация ответа."""
        if not document_names:
            return

        # Поиск документов по именам
        matched_docs: List[UUID] = []
        for name in document_names:
            if not isinstance(name, str) or not name.strip():
                continue
            docs = await self.projects_db.search_documents_any(
                query=name.strip(),
                limit=5
            )
            for d in docs:
                doc_id = d.get("id")
                if doc_id and doc_id not in matched_docs:
                    matched_docs.append(doc_id)

        if not matched_docs:
            return

        # Собираем доп. контекст
        extra_context = await self._build_document_context(matched_docs)
        if not extra_context:
            return

        yield self._create_phase_event("processing", "Загрузка доп. документов...")
        yield self._create_progress_event("processing", 1.0, "Доп. документы загружены")

        system_prompt = await self.llm_service.load_system_prompts(self.supabase)
        full_message = f"{original_context}\n\n{extra_context}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_message}"

        accumulated = ""
        async for token in self.llm_service.generate_simple(
            user_message=full_message,
            system_prompt=system_prompt
        ):
            accumulated += token
            yield StreamEvent(
                event="llm_token",
                data=LLMTokenEvent(token=token, accumulated=accumulated).dict(),
                timestamp=datetime.utcnow()
            )

        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=accumulated
        )

        yield StreamEvent(
            event="llm_final",
            data={"content": accumulated},
            timestamp=datetime.utcnow()
        )

    async def _find_crop_by_image_id(
        self,
        image_id: str,
        document_ids: Optional[List[UUID]]
    ) -> Optional[Dict[str, Any]]:
        """Найти crop по image_id в выбранных документах."""
        if not document_ids:
            return None

        crops = []
        for doc_id in document_ids:
            crops.extend(await self.projects_db.get_document_crops(doc_id))

        def normalize_id(name: str) -> str:
            base = Path(name).name
            return base.rsplit(".", 1)[0]

        crop_map = {normalize_id(c.get("r2_key", "")): c for c in crops if c.get("r2_key")}
        crop = crop_map.get(image_id)
        if not crop:
            for key, val in crop_map.items():
                if image_id in key:
                    return val
        return crop

    async def _download_bytes(self, key: str) -> Optional[bytes]:
        data = await self.s3_client.download_bytes(key)
        if data:
            return data
        url = self._build_public_url(key)
        if url:
            return await self._download_public(url)
        return None

    def _build_public_url(self, key: str) -> Optional[str]:
        """Публичная ссылка на файл в R2/S3."""
        if settings.use_s3_dev_url and settings.s3_dev_url:
            return f"{settings.s3_dev_url.rstrip('/')}/{key}"
        if settings.r2_public_domain:
            domain = settings.r2_public_domain.replace("https://", "").replace("http://", "")
            return f"https://{domain}/{key}"
        return None

    async def _download_public(self, url: str) -> Optional[bytes]:
        try:
            import httpx
            resp = httpx.get(url, timeout=20.0)
            if resp.status_code == 200:
                return resp.content
            return None
        except Exception:
            return None
    
    def _format_relevant_context(self, flash_result: Dict[str, Any]) -> str:
        """Форматировать релевантный контекст из Flash результата."""
        # TODO: Реализовать форматирование
        return ""
    
    def _create_phase_event(self, phase: str, description: str) -> StreamEvent:
        """Создать событие начала фазы."""
        return StreamEvent(
            event="phase_started",
            data=PhaseStartedEvent(
                phase=phase,
                description=description
            ).dict(),
            timestamp=datetime.utcnow()
        )
    
    def _create_progress_event(
        self,
        phase: str,
        progress: float,
        message: str
    ) -> StreamEvent:
        """Создать событие прогресса."""
        return StreamEvent(
            event="phase_progress",
            data=PhaseProgressEvent(
                phase=phase,
                progress=progress,
                message=message
            ).dict(),
            timestamp=datetime.utcnow()
        )

