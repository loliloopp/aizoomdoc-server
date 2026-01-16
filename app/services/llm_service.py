"""
Сервис для работы с Google Gemini LLM.
"""

import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
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
from app.models.internal import UserWithSettings, LLMResponse, ZoomRequest

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
                for uri in google_file_uris:
                    # Используем файловую ссылку через genai_types.FileData
                    user_parts.append(genai_types.Part.from_uri(
                        file_uri=uri,
                        mime_type=self._guess_mime_type(uri)
                    ))
            
            user_parts.append(genai_types.Part(text=user_message))
            contents.append(
                genai_types.Content(role="user", parts=user_parts)
            )
            
            # Конфигурация генерации
            generation_config = genai_types.GenerateContentConfig(
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_output_tokens=settings.max_tokens,
            )
            
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
    
    async def generate_complex_flash(
        self,
        user_message: str,
        document_context: str,
        supabase: SupabaseClient
    ) -> Dict[str, Any]:
        """
        Первый этап complex режима: Flash собирает контекст.
        
        Args:
            user_message: Сообщение пользователя
            document_context: Контекст документа
            supabase: Клиент Supabase
        
        Returns:
            Результат работы Flash (релевантные блоки, запросы изображений/зумов)
        """
        # TODO: Реализовать Flash экстрактор
        # 1. Загрузить flash_extractor промпт
        # 2. Сформировать запрос с каталогом изображений
        # 3. Обработать tool calls (request_images, zoom)
        # 4. Вернуть список релевантных блоков и изображений
        
        return {
            "status": "ready",
            "relevant_blocks": [],
            "relevant_images": [],
            "tool_calls": []
        }
    
    async def generate_complex_pro(
        self,
        user_message: str,
        relevant_context: str,
        images: List[Dict[str, Any]],
        supabase: SupabaseClient
    ) -> AsyncGenerator[str, None]:
        """
        Второй этап complex режима: Pro формирует финальный ответ.
        
        Args:
            user_message: Сообщение пользователя
            relevant_context: Релевантный контекст из Flash
            images: Изображения и зумы
            supabase: Клиент Supabase
        
        Yields:
            Токены ответа
        """
        # TODO: Реализовать Pro stage
        # 1. Загрузить системные промпты + роль
        # 2. Сформировать запрос с релевантным контекстом
        # 3. Стримить ответ
        
        system_prompt = await self.load_system_prompts(supabase)
        
        async for token in self.generate_simple(
            user_message=user_message,
            system_prompt=system_prompt,
            images=images
        ):
            yield token
    
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

