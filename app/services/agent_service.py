"""
Сервис агента - оркестратор пайплайна обработки запросов.
"""

import logging
import re
from typing import Optional, AsyncGenerator, Dict, Any, List
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image
from uuid import UUID
from datetime import datetime

from app.models.internal import UserWithSettings, SearchResult, LLMResponse
from app.models.api import StreamEvent, PhaseStartedEvent, PhaseProgressEvent, LLMTokenEvent, ToolCallEvent
from app.db.supabase_client import SupabaseClient
from app.db.supabase_projects_client import SupabaseProjectsClient
from app.db.s3_client import S3Client
from app.services.llm_service import create_llm_service
from app.services.search_service import SearchService
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)


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
        self.search_service = SearchService(projects_db)
        self.image_service = ImageService(s3_client)
    
    async def process_message(
        self,
        chat_id: UUID,
        user_message: str,
        client_id: Optional[str] = None,
        document_ids: Optional[List[UUID]] = None,
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
        try:
            # Сохраняем сообщение пользователя в БД (если нужно)
            if save_user_message:
                await self.supabase.add_message(
                    chat_id=chat_id,
                    role="user",
                    content=user_message
                )
            
            context_text = ""

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
                    google_file_uris=google_file_uris
                ):
                    yield event
            else:  # complex
                async for event in self._process_complex_mode(
                    chat_id, user_message, context_text, client_id,
                    google_file_uris=google_file_uris
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
            yield StreamEvent(
                event="error",
                data={"message": str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def _process_simple_mode(
        self,
        chat_id: UUID,
        user_message: str,
        context_text: str,
        document_ids: Optional[List[UUID]] = None,
        client_id: Optional[str] = None,
        google_file_uris: Optional[List[str]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Обработка в simple (flash) режиме."""
        
        # Загружаем системные промпты
        system_prompt = await self.llm_service.load_system_prompts(self.supabase)
        
        # Формируем полный промпт с контекстом
        full_message = f"{context_text}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_message}"
        
        # Стримим ответ
        accumulated_response = ""
        
        async for token in self.llm_service.generate_simple(
            user_message=full_message,
            system_prompt=system_prompt,
            google_file_uris=google_file_uris
        ):
            accumulated_response += token
            
            yield StreamEvent(
                event="llm_token",
                data=LLMTokenEvent(
                    token=token,
                    accumulated=accumulated_response
                ).dict(),
                timestamp=datetime.utcnow()
            )
        
        # Сохраняем ответ в БД
        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=accumulated_response
        )

        # Парсим tool calls (request_images / zoom / request_documents)
        tool_calls = await self.llm_service.parse_tool_calls(accumulated_response)
        for call in tool_calls:
            tool = call.get("tool")
            reason = call.get("reason", "")
            params = {k: v for k, v in call.items() if k not in ("tool", "reason")}
            if tool:
                yield StreamEvent(
                    event="tool_call",
                    data=ToolCallEvent(
                        tool=tool,
                        parameters=params,
                        reason=reason
                    ).dict(),
                    timestamp=datetime.utcnow()
                )
                # Обработка request_images
                if tool == "request_images":
                    image_ids = params.get("image_ids") or []
                    await self._handle_request_images(
                        chat_id=chat_id,
                        image_ids=image_ids,
                        document_ids=document_ids
                    )
                elif tool == "zoom":
                    await self._handle_zoom(
                        chat_id=chat_id,
                        image_id=params.get("image_id"),
                        document_ids=document_ids,
                        coords_norm=params.get("coords_norm"),
                        reason=reason
                    )
                elif tool == "request_documents":
                    # Запрос дополнительных документов по именам
                    doc_names = params.get("document_names") or params.get("document_ids") or []
                    async for ev in self._handle_request_documents(
                        chat_id=chat_id,
                        document_names=doc_names,
                        user_message=user_message,
                        original_context=context_text
                    ):
                        yield ev
        
        yield StreamEvent(
            event="llm_final",
            data={"content": accumulated_response},
            timestamp=datetime.utcnow()
        )
    
    async def _process_complex_mode(
        self,
        chat_id: UUID,
        user_message: str,
        context_text: str,
        client_id: str,
        google_file_uris: Optional[List[str]] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Обработка в complex (flash+pro) режиме."""
        
        # Этап 1: Flash собирает контекст
        yield self._create_phase_event("flash_stage", "Flash собирает контекст...")
        
        flash_result = await self.llm_service.generate_complex_flash(
            user_message=user_message,
            document_context=context_text,
            supabase=self.supabase
        )
        
        # TODO: Обработка tool calls (request_images, zoom)
        
        yield self._create_progress_event("flash_stage", 1.0, "Контекст собран")
        
        # Этап 2: Pro генерирует ответ
        yield self._create_phase_event("pro_stage", "Pro формирует ответ...")
        
        # Формируем релевантный контекст
        relevant_context = self._format_relevant_context(flash_result)
        
        # Стримим ответ от Pro
        accumulated_response = ""
        
        async for token in self.llm_service.generate_complex_pro(
            user_message=user_message,
            relevant_context=relevant_context,
            images=[],  # TODO: Добавить изображения из flash_result
            supabase=self.supabase
        ):
            accumulated_response += token
            
            yield StreamEvent(
                event="llm_token",
                data=LLMTokenEvent(
                    token=token,
                    accumulated=accumulated_response
                ).dict(),
                timestamp=datetime.utcnow()
            )
        
        # Сохраняем ответ в БД
        await self.supabase.add_message(
            chat_id=chat_id,
            role="assistant",
            content=accumulated_response
        )
        
        yield StreamEvent(
            event="llm_final",
            data={"content": accumulated_response},
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

        context_parts.append(
            "Если нужно запросить изображения или дополнительные документы, "
            "используй tool_call JSON: {\"tool\":\"request_images\",\"image_ids\":[...],\"reason\":\"...\"} "
            "или {\"tool\":\"request_documents\",\"document_names\":[...],\"reason\":\"...\"}."
        )

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

