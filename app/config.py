"""
Конфигурация приложения.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # JWT (Обязательный параметр)
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Supabase Chat DB (основная БД)
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_anon_key: str = Field(..., alias="SUPABASE_ANON_KEY")
    supabase_service_key: str = Field(default="", alias="SUPABASE_SERVICE_KEY")
    
    # Supabase Projects DB (read-only)
    supabase_projects_url: str = Field(..., alias="SUPABASE_PROJECTS_URL")
    supabase_projects_anon_key: str = Field(..., alias="SUPABASE_PROJECTS_ANON_KEY")
    supabase_projects_service_key: str = Field(default="", alias="SUPABASE_PROJECTS_SERVICE_KEY")
    use_projects_database: bool = Field(default=True, alias="USE_PROJECTS_DATABASE")
    
    # S3/R2 Storage
    # Приоритет на R2_* переменные из вашего .env
    r2_endpoint_url: str = Field(
        ..., 
        validation_alias=AliasChoices("R2_ENDPOINT_URL", "S3_ENDPOINT"),
    )
    r2_access_key_id: str = Field(
        ..., 
        validation_alias=AliasChoices("R2_ACCESS_KEY_ID", "S3_ACCESS_KEY"),
    )
    r2_secret_access_key: str = Field(
        ..., 
        validation_alias=AliasChoices("R2_SECRET_ACCESS_KEY", "S3_SECRET_KEY"),
    )
    r2_bucket_name: str = Field(
        default="aizoomdoc", 
        validation_alias=AliasChoices("R2_BUCKET_NAME", "S3_BUCKET"),
    )
    r2_public_domain: str = Field(default="", alias="R2_PUBLIC_DOMAIN")
    s3_dev_url: str = Field(default="", alias="S3_DEV_URL")
    use_s3_dev_url: bool = Field(default=False, alias="USE_S3_DEV_URL")
    
    # LLM
    default_gemini_api_key: str = Field(
        default="", 
        validation_alias=AliasChoices("GOOGLE_API_KEY", "DEFAULT_GEMINI_API_KEY")
    )
    default_model: str = Field(default="gemini-3-flash-preview", alias="DEFAULT_MODEL")
    default_flash_model: str = Field(default="gemini-3-flash-preview", alias="DEFAULT_FLASH_MODEL")
    default_pro_model: str = Field(default="gemini-3-pro-preview", alias="DEFAULT_PRO_MODEL")
    max_tokens: int = Field(default=15000, alias="MAX_TOKENS")
    llm_temperature: float = Field(default=1.0, alias="LLM_TEMPERATURE")
    llm_top_p: float = Field(default=0.95, alias="LLM_TOP_P")
    media_resolution: str = Field(default="high", alias="MEDIA_RESOLUTION")
    
    # Thinking
    thinking_enabled: bool = Field(default=True, alias="THINKING_ENABLED")
    thinking_budget: int = Field(default=0, alias="THINKING_BUDGET")
    
    # Image Processing
    preview_max_side: int = Field(default=2000, alias="PREVIEW_MAX_SIDE")
    zoom_preview_max_side: int = Field(default=2000, alias="ZOOM_PREVIEW_MAX_SIDE")
    auto_quadrants_threshold: float = Field(default=2.5, alias="AUTO_QUADRANTS_THRESHOLD")
    viewport_size: int = Field(default=2048, alias="VIEWPORT_SIZE")
    viewport_padding: int = Field(default=512, alias="VIEWPORT_PADDING")

    # Local LLM logging
    llm_log_enabled: bool = Field(default=True, alias="LLM_LOG_ENABLED")
    llm_log_dir: str = Field(default="logs", alias="LLM_LOG_DIR")
    llm_log_truncate_chars: int = Field(default=20000, alias="LLM_LOG_TRUNCATE_CHARS")
    
    # File Upload
    max_file_size_mb: int = Field(default=100, alias="MAX_FILE_SIZE_MB")
    
    # CORS
    cors_origins: str = Field(default="http://localhost:3000,http://localhost:5173", alias="CORS_ORIGINS")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Парсинг списка CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore" # Игнорировать лишние переменные


# Глобальный экземпляр настроек
settings = Settings()
