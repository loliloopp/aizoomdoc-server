"""
Pydantic модели для API контрактов.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field


# ===== AUTH MODELS =====

class TokenExchangeRequest(BaseModel):
    """Запрос обмена статичного токена на JWT."""
    static_token: str = Field(..., description="Статичный токен пользователя")


class TokenExchangeResponse(BaseModel):
    """Ответ с JWT токенами."""
    access_token: str = Field(..., description="Access JWT token")
    token_type: str = Field(default="bearer", description="Тип токена")
    expires_in: int = Field(..., description="Время жизни access token в секундах")
    user: "UserInfo" = Field(..., description="Информация о пользователе")


class UserInfo(BaseModel):
    """Информация о пользователе."""
    id: UUID
    username: str
    status: str = Field(default="active")
    created_at: datetime


# ===== SETTINGS MODELS =====

class UserSettings(BaseModel):
    """Настройки пользователя."""
    model_profile: Literal["simple", "complex"] = Field(
        default="simple",
        description="Режим модели: simple (flash) или complex (flash+pro)"
    )
    selected_role_prompt_id: Optional[int] = Field(
        default=None,
        description="ID выбранной роли из prompts_user"
    )
    # LLM параметры
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Температура генерации")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    thinking_enabled: bool = Field(default=True, description="Включить режим thinking (deep think)")
    thinking_budget: int = Field(default=0, ge=0, le=24576, description="Бюджет токенов для thinking (0=авто)")
    media_resolution: Literal["low", "medium", "high"] = Field(default="high", description="Разрешение медиафайлов")


class UserSettingsUpdate(BaseModel):
    """Запрос обновления настроек пользователя."""
    model_profile: Optional[Literal["simple", "complex"]] = None
    selected_role_prompt_id: Optional[int] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    thinking_enabled: Optional[bool] = None
    thinking_budget: Optional[int] = Field(default=None, ge=0, le=24576)
    media_resolution: Optional[Literal["low", "medium", "high"]] = None


class UserMeResponse(BaseModel):
    """Ответ с информацией о текущем пользователе."""
    user: UserInfo
    settings: UserSettings
    gemini_api_key_configured: bool = Field(
        ...,
        description="Есть ли у пользователя настроенный Gemini API key"
    )


# ===== PROMPTS MODELS =====

class PromptBase(BaseModel):
    """Базовая модель промпта."""
    name: str = Field(..., description="Название промпта")
    content: str = Field(..., description="Содержимое промпта")
    description: Optional[str] = Field(None, description="Описание назначения")
    is_active: bool = Field(default=True, description="Активен ли промпт")


class PromptSystemCreate(PromptBase):
    """Создание системного промпта."""
    pass


class PromptSystem(PromptBase):
    """Системный промпт."""
    id: UUID
    version: int = Field(default=1)
    created_at: datetime
    updated_at: datetime


class PromptUserRole(PromptBase):
    """Пользовательский промпт-роль."""
    id: int  # bigint в БД
    version: int = Field(default=1)
    created_at: datetime
    updated_at: datetime


# ===== CHAT MODELS =====

class ChatCreate(BaseModel):
    """Создание чата."""
    title: Optional[str] = Field(None, description="Заголовок чата")
    description: Optional[str] = Field(None, description="Описание чата")


class ChatResponse(BaseModel):
    """Ответ с информацией о чате."""
    id: UUID
    title: str
    description: Optional[str]
    user_id: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    """Создание сообщения в чате."""
    content: str = Field(..., description="Содержимое сообщения")
    attached_file_ids: Optional[List[UUID]] = Field(
        default=None,
        description="ID прикрепленных файлов"
    )
    attached_document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="ID документов из дерева (projects DB)"
    )
    google_file_uris: Optional[List[str]] = Field(
        default=None,
        description="URI файлов из Google File API"
    )


class MessageImage(BaseModel):
    """Изображение в сообщении."""
    id: UUID
    file_id: Optional[UUID] = None
    image_type: Optional[str] = None
    description: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    url: Optional[str] = None  # URL для доступа к изображению


class MessageResponse(BaseModel):
    """Ответ с сообщением."""
    id: UUID
    chat_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    message_type: str = Field(default="text")
    created_at: datetime
    images: List[MessageImage] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """История чата."""
    chat: ChatResponse
    messages: List[MessageResponse]


# ===== STREAMING MODELS =====

class StreamEvent(BaseModel):
    """Событие стриминга."""
    event: Literal[
        "phase_started",
        "phase_progress",
        "llm_token",
        "llm_final",
        "tool_call",
        "error",
        "completed"
    ]
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PhaseStartedEvent(BaseModel):
    """Событие начала фазы обработки."""
    phase: str = Field(..., description="Название фазы: search, processing, llm, zoom")
    description: str = Field(..., description="Описание фазы")


class PhaseProgressEvent(BaseModel):
    """Событие прогресса фазы."""
    phase: str
    progress: float = Field(..., ge=0.0, le=1.0, description="Прогресс от 0 до 1")
    message: str


class LLMTokenEvent(BaseModel):
    """Событие токена от LLM."""
    token: str
    accumulated: str = Field(..., description="Накопленный текст")


class ToolCallEvent(BaseModel):
    """Событие вызова инструмента (request_images, zoom)."""
    tool: Literal["request_images", "zoom", "request_documents"]
    parameters: Dict[str, Any]
    reason: str


# ===== FILE MODELS =====

class FileUploadResponse(BaseModel):
    """Ответ после загрузки файла."""
    id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    storage_path: str
    created_at: datetime


class GoogleFileUploadResponse(BaseModel):
    """Ответ после загрузки файла в Google File API."""
    id: Optional[UUID] = None
    filename: str
    mime_type: str
    size_bytes: int
    google_file_uri: str
    google_file_name: str
    state: str = "ACTIVE"
    storage_path: Optional[str] = None


class FileInfo(BaseModel):
    """Информация о файле."""
    id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    source_type: str
    storage_path: Optional[str]
    external_url: Optional[str]
    created_at: datetime


# ===== PROJECTS TREE MODELS (read-only) =====

class TreeNode(BaseModel):
    """Узел дерева проектов."""
    id: UUID
    parent_id: Optional[UUID]
    client_id: Optional[str] = None
    node_type: str
    name: str
    code: Optional[str]
    version: Optional[int]
    status: str = Field(default="active")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    sort_order: int = Field(default=0)
    created_at: datetime
    updated_at: datetime


class DocumentResults(BaseModel):
    """Результаты обработки документа."""
    document_node_id: UUID
    files: List[FileInfo]


# ===== ERROR MODELS =====

class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")

