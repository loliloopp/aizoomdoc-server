"""
Сервис для работы с Google Gemini LLM.
"""

import json
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator, Iterable, Union
from pathlib import Path
from uuid import UUID

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from app.config import settings
from app.db.supabase_client import SupabaseClient
from app.models.internal import UserWithSettings
from app.models.llm_schemas import get_flash_collector_schema, get_answer_schema

logger = logging.getLogger(__name__)


class LLMService:
    """Сервис для работы с Gemini LLM."""
    
    def __init__(self, user: UserWithSettings):
        """
        Инициализация сервиса LLM.
        
        Args:
            user: Пользователь с настройками и API ключом
        """
        self.user = user
        
        # Используем пользовательский ключ или дефолтный
        api_key = user.gemini_api_key or settings.default_gemini_api_key
        
        if not api_key:
            raise ValueError("Gemini API key not configured")
        
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        
        # Инициализация клиента Gemini
        self.client = genai.Client(api_key=api_key)
        self.model_name = settings.default_model
    
    async def load_system_prompts(self, supabase: SupabaseClient) -> str:
        """
        Загрузить и скомпоновать системные промпты.
        
        Args:
            supabase: Клиент Supabase
        
        Returns:
            Скомпонованный системный промпт
        """
        prompts = []
        
        # Если выбрана роль, добавляем её первой
        if self.user.settings.selected_role_prompt_id:
            role = await supabase.get_user_prompt_by_id(
                self.user.settings.selected_role_prompt_id
            )
            if role:
                prompts.append(role.content)
        
        # Добавляем системные промпты
        system_prompts = await supabase.get_system_prompts(active_only=True)
        
        # Порядок: llm_system, json_annotation, html_ocr
        prompt_order = ["llm_system", "json_annotation", "html_ocr"]
        
        for name in prompt_order:
            prompt = next((p for p in system_prompts if p.name == name), None)
            if prompt:
                prompts.append(prompt.content)
        
        return "\n\n".join(prompts)
    
    async def generate_simple(
        self,
        user_message: str,
        system_prompt: str,
        images: Optional[List[Dict[str, Any]]] = None,
        google_file_uris: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Генерация в simple (flash) режиме со стримингом.
        
        Args:
            user_message: Сообщение пользователя
            system_prompt: Системный промпт
            images: Список изображений для контекста
            google_file_uris: URI файлов из Google File API
        
        Yields:
            Токены ответа
        """
        try:
            # Формируем contents для Gemini
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=system_prompt)]
                )
            ]
            
            # Добавляем изображения если есть
            user_parts = []
            if images:
                for img_data in images:
                    # TODO: Загрузить изображение и добавить в parts
                    pass
            
            # Добавляем файлы из Google File API
            if google_file_uris:
                for uri_item in google_file_uris:
                    # uri_item может быть строкой или dict с uri и mime_type
                    if isinstance(uri_item, dict):
                        uri = uri_item.get("uri", "")
                        mime = uri_item.get("mime_type", "text/plain")
                    else:
                        uri = uri_item
                        # По умолчанию text/plain для текстовых файлов
                        mime = "text/plain"
                    
                    if uri:
                        logger.info(f"Adding file to LLM: uri={uri}, mime_type={mime}")
                        user_parts.append(genai_types.Part.from_uri(
                            file_uri=uri,
                            mime_type=mime
                        ))
            
            user_parts.append(genai_types.Part(text=user_message))
            contents.append(
                genai_types.Content(role="user", parts=user_parts)
            )
            
            # Получаем параметры из настроек пользователя или дефолтные
            user_settings = self.user.settings
            temperature = getattr(user_settings, 'temperature', None) or settings.llm_temperature
            top_p = getattr(user_settings, 'top_p', None) or settings.llm_top_p
            thinking_enabled = getattr(user_settings, 'thinking_enabled', True)
            thinking_budget = getattr(user_settings, 'thinking_budget', 0)
            media_resolution = getattr(user_settings, 'media_resolution', 'high')
            
            logger.info(f"LLM params: temp={temperature}, top_p={top_p}, thinking={thinking_enabled}, budget={thinking_budget}, media={media_resolution}")
            
            # Конфигурация генерации
            config_params = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": settings.max_tokens,
            }
            
            # Добавляем thinking config если включен
            if thinking_enabled and hasattr(genai_types, 'ThinkingConfig'):
                thinking_config = genai_types.ThinkingConfig(
                    thinking_budget=thinking_budget if thinking_budget > 0 else None
                )
                config_params["thinking_config"] = thinking_config
            
            # Добавляем media resolution
            if hasattr(genai_types, 'MediaResolution'):
                media_res_map = {
                    "low": genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    "medium": genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                    "high": genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                }
                if media_resolution in media_res_map:
                    config_params["media_resolution"] = media_res_map[media_resolution]
            
            generation_config = genai_types.GenerateContentConfig(**config_params)
            
            # Стриминг ответа
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generation_config
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        
        except Exception as e:
            logger.error(f"Error in generate_simple: {e}")
            raise

    def _build_generation_config(
        self,
        system_prompt: str,
        response_schema: Optional[dict] = None
    ) -> "genai_types.GenerateContentConfig":
        """Сформировать конфигурацию генерации с учётом настроек пользователя."""
        user_settings = self.user.settings
        temperature = getattr(user_settings, "temperature", None) or settings.llm_temperature
        top_p = getattr(user_settings, "top_p", None) or settings.llm_top_p
        thinking_enabled = getattr(user_settings, "thinking_enabled", True)
        thinking_budget = getattr(user_settings, "thinking_budget", 0)
        media_resolution = getattr(user_settings, "media_resolution", "high")

        config_params: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": settings.max_tokens,
            "system_instruction": system_prompt,
        }

        if thinking_enabled and hasattr(genai_types, "ThinkingConfig"):
            config_params["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget if thinking_budget > 0 else None
            )

        if hasattr(genai_types, "MediaResolution"):
            media_res_map = {
                "low": genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            if media_resolution in media_res_map:
                config_params["media_resolution"] = media_res_map[media_resolution]

        if response_schema:
            config_params["response_mime_type"] = "application/json"
            config_params["response_schema"] = response_schema

        return genai_types.GenerateContentConfig(**config_params)

    def _build_contents(
        self,
        user_message: str,
        google_file_uris: Optional[Iterable[Union[dict, str]]] = None
    ) -> List["genai_types.Content"]:
        """Сформировать contents для Gemini."""
        parts: List[Any] = []
        if google_file_uris:
            for uri_item in google_file_uris:
                if isinstance(uri_item, dict):
                    uri = uri_item.get("uri", "")
                    mime = uri_item.get("mime_type") or self._guess_mime_type(uri)
                else:
                    uri = uri_item
                    mime = self._guess_mime_type(uri)
                if uri:
                    parts.append(genai_types.Part.from_uri(file_uri=uri, mime_type=mime))
        parts.append(genai_types.Part(text=user_message))
        return [genai_types.Content(role="user", parts=parts)]

    async def generate_json_response(
        self,
        *,
        system_prompt: str,
        user_message: str,
        google_file_uris: Optional[Iterable[Union[dict, str]]] = None,
        response_schema: Optional[dict] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Выполнить вызов LLM и вернуть JSON текст."""
        try:
            contents = self._build_contents(user_message, google_file_uris)
            config = self._build_generation_config(system_prompt, response_schema=response_schema)
            response = self.client.models.generate_content(
                model=model_name or self.model_name,
                contents=contents,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.error(f"Error in generate_json_response: {e}")
            raise

    def parse_json(self, text: str) -> dict:
        """Попытаться распарсить JSON из ответа LLM."""
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Попытка вырезать JSON-объект из текста
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise
    
    async def run_flash_collector(
        self,
        *,
        system_prompt: str,
        user_message: str,
        google_file_uris: Optional[Iterable[Union[dict, str]]] = None,
        model_name: Optional[str] = None,
        return_text: bool = False,
    ) -> Union[dict, tuple[dict, str]]:
        """Запуск Flash-collector (строгий JSON)."""
        text = await self.generate_json_response(
            system_prompt=system_prompt,
            user_message=user_message,
            google_file_uris=google_file_uris,
            response_schema=get_flash_collector_schema(),
            model_name=model_name,
        )
        parsed = self.parse_json(text)
        if return_text:
            return parsed, text
        return parsed

    async def run_answer(
        self,
        *,
        system_prompt: str,
        user_message: str,
        google_file_uris: Optional[Iterable[Union[dict, str]]] = None,
        model_name: Optional[str] = None,
        return_text: bool = False,
    ) -> Union[dict, tuple[dict, str]]:
        """Запуск ответа (Flash или Pro) со строгим JSON."""
        text = await self.generate_json_response(
            system_prompt=system_prompt,
            user_message=user_message,
            google_file_uris=google_file_uris,
            response_schema=get_answer_schema(),
            model_name=model_name,
        )
        parsed = self.parse_json(text)
        if return_text:
            return parsed, text
        return parsed
    
    def _guess_mime_type(self, uri: str) -> str:
        """Определить MIME тип по расширению в URI."""
        uri_lower = uri.lower()
        if '.pdf' in uri_lower:
            return 'application/pdf'
        elif '.md' in uri_lower:
            return 'text/markdown'
        elif '.html' in uri_lower:
            return 'text/html'
        elif '.txt' in uri_lower:
            return 'text/plain'
        elif '.json' in uri_lower:
            return 'application/json'
        elif '.csv' in uri_lower:
            return 'text/csv'
        elif '.png' in uri_lower:
            return 'image/png'
        elif '.jpg' in uri_lower or '.jpeg' in uri_lower:
            return 'image/jpeg'
        elif '.webp' in uri_lower:
            return 'image/webp'
        elif '.gif' in uri_lower:
            return 'image/gif'
        else:
            return 'application/octet-stream'
    
    async def parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Распарсить tool calls из ответа LLM.
        
        Args:
            response_text: Текст ответа от LLM
        
        Returns:
            Список tool calls
        """
        import json

        def extract_json_objects(text: str) -> List[Any]:
            results = []
            decoder = json.JSONDecoder()
            pos = 0
            length = len(text)

            while pos < length:
                idx_brace = text.find("{", pos)
                if idx_brace == -1:
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text[idx_brace:])
                    results.append(obj)
                    pos = idx_brace + end_idx
                except json.JSONDecodeError:
                    pos = idx_brace + 1
            return results

        objs = extract_json_objects(response_text)
        tool_calls: List[Dict[str, Any]] = []

        for obj in objs:
            if isinstance(obj, dict):
                if "tool" in obj:
                    tool_calls.append(obj)
                elif "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                    for tc in obj["tool_calls"]:
                        if isinstance(tc, dict) and "tool" in tc:
                            tool_calls.append(tc)

        return tool_calls


def create_llm_service(user: UserWithSettings) -> LLMService:
    """
    Фабрика для создания LLM сервиса.
    
    Args:
        user: Пользователь с настройками
    
    Returns:
        Экземпляр LLMService
    """
    return LLMService(user)

