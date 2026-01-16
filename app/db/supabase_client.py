"""
Клиент для работы с Supabase (основная БД для чатов и пользователей).
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from supabase import create_client, Client as SupabaseClientSDK
from postgrest.exceptions import APIError

from app.config import settings
from app.models.internal import (
    User,
    UserWithSettings,
    Settings,
    SystemPrompt,
    UserPrompt,
    Chat,
    Message,
    ChatImage,
    StorageFile,
)

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Клиент для работы с Supabase Chat DB."""
    
    def __init__(self):
        """Инициализация клиента Supabase."""
        # Используем service_key если есть, иначе anon_key
        key = settings.supabase_service_key or settings.supabase_anon_key
        self.client: SupabaseClientSDK = create_client(
            settings.supabase_url,
            key
        )
    
    # ===== USER METHODS =====
    
    async def get_user_by_static_token(self, static_token: str) -> Optional[User]:
        """
        Получить пользователя по статичному токену.
        
        Args:
            static_token: Статичный токен пользователя
        
        Returns:
            Пользователь или None
        """
        try:
            response = self.client.table("users").select("*").eq("static_token", static_token).execute()
            
            if not response.data:
                return None
            
            user_data = response.data[0]
            return User(**user_data)
        
        except Exception as e:
            logger.error(f"Error getting user by static token: {e}")
            return None
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Получить пользователя по ID.
        
        Args:
            user_id: UUID пользователя
        
        Returns:
            Пользователь или None
        """
        try:
            response = self.client.table("users").select("*").eq("id", str(user_id)).execute()
            
            if not response.data:
                return None
            
            return User(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    async def update_user_last_seen(self, user_id: UUID) -> None:
        """Обновить время последнего входа пользователя."""
        try:
            self.client.table("users").update({
                "last_seen_at": datetime.utcnow().isoformat()
            }).eq("id", str(user_id)).execute()
        except Exception as e:
            logger.error(f"Error updating user last seen: {e}")
    
    # ===== SETTINGS METHODS =====
    
    async def get_user_settings(self, user_id: UUID) -> Optional[Settings]:
        """
        Получить настройки пользователя.
        
        Args:
            user_id: UUID пользователя
        
        Returns:
            Настройки или None
        """
        try:
            # В settings используется user_id как VARCHAR (username)
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            response = self.client.table("settings").select("*").eq("user_id", user.username).execute()
            
            if not response.data:
                # Создаем настройки по умолчанию
                return await self.create_default_settings(user_id, user.username)
            
            settings_data = response.data[0]
            # Преобразуем user_id обратно в UUID для модели
            settings_data["user_id"] = str(user_id)
            return Settings(**settings_data)
        
        except Exception as e:
            logger.error(f"Error getting user settings: {e}")
            return None
    
    async def create_default_settings(self, user_id: UUID, username: str) -> Settings:
        """Создать настройки по умолчанию для пользователя."""
        try:
            now = datetime.utcnow()
            settings_data = {
                "user_id": username,
                "model_profile": "simple",
                "page_settings": {},
                "created_at": now.isoformat(),
                "updated_at": now.isoformat()
            }
            
            response = self.client.table("settings").insert(settings_data).execute()
            
            result_data = response.data[0]
            result_data["user_id"] = str(user_id)
            return Settings(**result_data)
        
        except Exception as e:
            logger.error(f"Error creating default settings: {e}")
            raise
    
    async def update_user_settings(
        self,
        user_id: UUID,
        model_profile: Optional[str] = None,
        selected_role_prompt_id: Optional[int] = None
    ) -> Optional[Settings]:
        """
        Обновить настройки пользователя.
        
        Args:
            user_id: UUID пользователя
            model_profile: Режим модели (simple/complex)
            selected_role_prompt_id: ID выбранной роли
        
        Returns:
            Обновленные настройки
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            update_data = {"updated_at": datetime.utcnow().isoformat()}
            
            if model_profile is not None:
                update_data["model_profile"] = model_profile
            
            if selected_role_prompt_id is not None:
                update_data["selected_role_prompt_id"] = selected_role_prompt_id
            
            response = self.client.table("settings").update(update_data).eq("user_id", user.username).execute()
            
            if not response.data:
                return None
            
            result_data = response.data[0]
            result_data["user_id"] = str(user_id)
            return Settings(**result_data)
        
        except Exception as e:
            logger.error(f"Error updating user settings: {e}")
            return None
    
    async def get_user_with_settings(self, user_id: UUID) -> Optional[UserWithSettings]:
        """
        Получить пользователя с настройками.
        
        Args:
            user_id: UUID пользователя
        
        Returns:
            Пользователь с настройками
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        
        user_settings = await self.get_user_settings(user_id)
        if not user_settings:
            return None
        
        # TODO: В будущем здесь загружать зашифрованный gemini_api_key
        gemini_api_key = None
        
        return UserWithSettings(
            user=user,
            settings=user_settings,
            gemini_api_key=gemini_api_key
        )
    
    # ===== PROMPTS METHODS =====
    
    async def get_system_prompts(self, active_only: bool = True) -> List[SystemPrompt]:
        """
        Получить системные промпты.
        
        Args:
            active_only: Только активные промпты
        
        Returns:
            Список системных промптов
        """
        try:
            query = self.client.table("prompts_system").select("*")
            
            if active_only:
                query = query.eq("is_active", True)
            
            response = query.execute()
            
            return [SystemPrompt(**prompt) for prompt in response.data]
        
        except Exception as e:
            logger.error(f"Error getting system prompts: {e}")
            return []
    
    async def get_system_prompt_by_name(self, name: str) -> Optional[SystemPrompt]:
        """Получить системный промпт по имени."""
        try:
            response = self.client.table("prompts_system").select("*").eq("name", name).eq("is_active", True).execute()
            
            if not response.data:
                return None
            
            return SystemPrompt(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error getting system prompt by name: {e}")
            return None
    
    async def get_user_prompts(self, active_only: bool = True) -> List[UserPrompt]:
        """
        Получить пользовательские промпты (роли).
        
        Args:
            active_only: Только активные промпты
        
        Returns:
            Список ролей
        """
        try:
            query = self.client.table("user_prompts").select("*")
            
            # Фильтруем только общие роли (для default_user)
            query = query.eq("user_id", "default_user")
            
            response = query.execute()
            
            return [UserPrompt(**prompt) for prompt in response.data]
        
        except Exception as e:
            logger.error(f"Error getting user prompts: {e}")
            return []
    
    async def get_user_prompt_by_id(self, prompt_id: int) -> Optional[UserPrompt]:
        """Получить роль по ID."""
        try:
            response = self.client.table("user_prompts").select("*").eq("id", prompt_id).execute()
            
            if not response.data:
                return None
            
            return UserPrompt(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error getting user prompt by ID: {e}")
            return None
    
    # ===== CHAT METHODS =====
    
    async def create_chat(
        self,
        user_id: UUID,
        title: str,
        description: Optional[str] = None
    ) -> Optional[Chat]:
        """
        Создать чат.
        
        Args:
            user_id: UUID пользователя
            title: Заголовок чата
            description: Описание чата
        
        Returns:
            Созданный чат
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            chat_data = {
                "title": title,
                "description": description,
                "user_id": user.username,
                "metadata": {}
            }
            
            response = self.client.table("chats").insert(chat_data).execute()
            
            return Chat(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error creating chat: {e}")
            return None
    
    async def get_chat(self, chat_id: UUID) -> Optional[Chat]:
        """Получить чат по ID."""
        try:
            response = self.client.table("chats").select("*").eq("id", str(chat_id)).execute()
            
            if not response.data:
                return None
            
            return Chat(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error getting chat: {e}")
            return None
    
    async def get_user_chats(self, user_id: UUID, limit: int = 50) -> List[Chat]:
        """Получить чаты пользователя."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return []
            
            response = (
                self.client.table("chats")
                .select("*")
                .eq("user_id", user.username)
                .eq("is_archived", False)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
            
            return [Chat(**chat) for chat in response.data]
        
        except Exception as e:
            logger.error(f"Error getting user chats: {e}")
            return []
    
    # ===== MESSAGE METHODS =====
    
    async def add_message(
        self,
        chat_id: UUID,
        role: str,
        content: str,
        message_type: str = "text"
    ) -> Optional[Message]:
        """
        Добавить сообщение в чат.
        
        Args:
            chat_id: UUID чата
            role: Роль (user/assistant/system)
            content: Содержимое сообщения
            message_type: Тип сообщения
        
        Returns:
            Созданное сообщение
        """
        try:
            message_data = {
                "chat_id": str(chat_id),
                "role": role,
                "content": content,
                "message_type": message_type
            }
            
            response = self.client.table("chat_messages").insert(message_data).execute()
            
            return Message(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return None
    
    async def get_chat_messages(self, chat_id: UUID, limit: int = 100) -> List[Message]:
        """Получить сообщения чата."""
        try:
            response = (
                self.client.table("chat_messages")
                .select("*")
                .eq("chat_id", str(chat_id))
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            
            return [Message(**msg) for msg in response.data]
        
        except Exception as e:
            logger.error(f"Error getting chat messages: {e}")
            return []

    async def get_last_message(
        self,
        chat_id: UUID,
        role: Optional[str] = None
    ) -> Optional[Message]:
        """Получить последнее сообщение в чате (опционально по роли)."""
        try:
            query = (
                self.client.table("chat_messages")
                .select("*")
                .eq("chat_id", str(chat_id))
                .order("created_at", desc=True)
                .limit(1)
            )
            if role:
                query = query.eq("role", role)

            response = query.execute()
            if response.data:
                return Message(**response.data[0])
            return None
        except Exception as e:
            logger.error(f"Error getting last message: {e}")
            return None
    
    async def get_message_images(self, message_id: UUID) -> List[dict]:
        """Получить изображения сообщения."""
        try:
            # Получаем chat_images с join на storage_files
            response = (
                self.client.table("chat_images")
                .select("*, storage_files(*)")
                .eq("message_id", str(message_id))
                .execute()
            )
            
            images = []
            for item in response.data:
                storage_file = item.get("storage_files", {})
                images.append({
                    "id": item.get("id"),
                    "message_id": item.get("message_id"),
                    "file_id": item.get("file_id"),
                    "image_type": item.get("image_type"),
                    "description": item.get("description"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                    "storage_path": storage_file.get("storage_path") if storage_file else None,
                    "external_url": storage_file.get("external_url") if storage_file else None,
                    "filename": storage_file.get("filename") if storage_file else None,
                })
            
            return images
        
        except Exception as e:
            logger.error(f"Error getting message images: {e}")
            return []

    async def add_chat_image(
        self,
        chat_id: UUID,
        message_id: UUID,
        file_id: Optional[UUID],
        image_type: Optional[str] = None,
        description: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Optional[ChatImage]:
        """Добавить изображение к сообщению."""
        try:
            data = {
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "file_id": str(file_id) if file_id else None,
                "image_type": image_type,
                "description": description,
                "width": width,
                "height": height
            }
            response = self.client.table("chat_images").insert(data).execute()
            if response.data:
                return ChatImage(**response.data[0])
            return None
        except Exception as e:
            logger.error(f"Error adding chat image: {e}")
            return None
    
    # ===== FILE METHODS =====
    
    async def register_file(
        self,
        user_id: UUID,
        filename: str,
        mime_type: str,
        size_bytes: int,
        storage_path: Optional[str],
        source_type: str = "user_upload",
        external_url: Optional[str] = None
    ) -> Optional[StorageFile]:
        """Зарегистрировать файл в БД."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            file_data = {
                "user_id": user.username,
                "filename": filename,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "storage_path": storage_path,
                "source_type": source_type,
                "external_url": external_url
            }
            
            response = self.client.table("storage_files").insert(file_data).execute()
            
            return StorageFile(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error registering file: {e}")
            return None
    
    async def get_file(self, file_id: UUID) -> Optional[StorageFile]:
        """Получить файл по ID."""
        try:
            response = self.client.table("storage_files").select("*").eq("id", str(file_id)).execute()
            
            if not response.data:
                return None
            
            return StorageFile(**response.data[0])
        
        except Exception as e:
            logger.error(f"Error getting file: {e}")
            return None

