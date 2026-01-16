"""
API роутер для работы с файлами.
"""

import logging
import tempfile
import os
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import Optional

try:
    from google import genai
except ImportError:
    genai = None

from app.config import settings
from app.core.dependencies import get_current_user_id, get_current_user
from app.db.supabase_client import SupabaseClient
from app.db.s3_client import S3Client
from app.models.api import FileUploadResponse, FileInfo, GoogleFileUploadResponse
from app.models.internal import UserWithSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id),
    supabase: SupabaseClient = Depends(),
    s3: S3Client = Depends()
):
    """
    Загрузить файл.
    
    Args:
        file: Загружаемый файл
        user_id: ID текущего пользователя
        supabase: Клиент Supabase
        s3: Клиент S3
    
    Returns:
        Информация о загруженном файле
    
    Raises:
        HTTPException: При ошибке загрузки
    """
    # Проверяем размер файла
    contents = await file.read()
    file_size = len(contents)
    
    max_size = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )
    
    # Генерируем ключ для S3
    user = await supabase.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    s3_key = s3.generate_key(
        user_id=user.username,
        filename=file.filename,
        prefix="uploads"
    )
    
    # Загружаем в S3
    url = await s3.upload_bytes(
        data=contents,
        key=s3_key,
        content_type=file.content_type or "application/octet-stream"
    )
    
    if not url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file to storage"
        )
    
    # Регистрируем в БД
    stored_file = await supabase.register_file(
        user_id=user_id,
        filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=file_size,
        storage_path=s3_key,
        source_type="user_upload"
    )
    
    if not stored_file:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register file in database"
        )
    
    return FileUploadResponse(
        id=stored_file.id,
        filename=stored_file.filename,
        mime_type=stored_file.mime_type,
        size_bytes=stored_file.size_bytes,
        storage_path=stored_file.storage_path,
        created_at=stored_file.created_at
    )


@router.post("/upload-for-llm", response_model=GoogleFileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file_for_llm(
    file: UploadFile = File(...),
    user: UserWithSettings = Depends(get_current_user),
    supabase: SupabaseClient = Depends()
):
    """
    Загрузить файл через Google File API для использования в LLM.
    
    Поддерживаемые форматы:
    - Текстовые: .txt, .md, .html, .csv, .json
    - Документы: .pdf
    - Изображения: .png, .jpg, .jpeg, .webp, .gif
    
    Args:
        file: Загружаемый файл
        user: Текущий пользователь с настройками
        supabase: Клиент Supabase
    
    Returns:
        Информация о загруженном файле с Google File URI
    """
    if genai is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google AI SDK not available"
        )
    
    # Получаем API ключ
    api_key = user.gemini_api_key or settings.default_gemini_api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API key not configured"
        )
    
    # Читаем содержимое файла
    contents = await file.read()
    file_size = len(contents)
    
    # Проверяем размер (Google File API имеет свои лимиты)
    max_size = 20 * 1024 * 1024  # 20MB для большинства файлов
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of 20MB for Google File API"
        )
    
    try:
        # Инициализируем клиент Gemini
        client = genai.Client(api_key=api_key)
        
        # Определяем расширение файла (ASCII-safe)
        import re
        ext_match = re.search(r'\.([a-zA-Z0-9]+)$', file.filename or '')
        ext = f".{ext_match.group(1)}" if ext_match else ""
        
        # Сохраняем во временный файл с ASCII именем
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Загружаем в Google File API
            mime_type = file.content_type or "application/octet-stream"
            
            # display_name может содержать кириллицу
            display_name = file.filename or "uploaded_file"
            
            # Используем синхронный upload (google-genai работает синхронно)
            uploaded_file = client.files.upload(
                file=tmp_path,
                config={
                    "display_name": display_name,
                    "mime_type": mime_type
                }
            )
            
            logger.info(f"File uploaded to Google: {uploaded_file.name}, URI: {uploaded_file.uri}")
            
            # Сохраняем информацию в БД
            stored_file = await supabase.register_file(
                user_id=user.user.id,
                filename=file.filename,
                mime_type=mime_type,
                size_bytes=file_size,
                storage_path=None,
                source_type="google_file_api",
                external_url=uploaded_file.uri
            )
            
            return GoogleFileUploadResponse(
                id=stored_file.id if stored_file else None,
                filename=file.filename,
                mime_type=mime_type,
                size_bytes=file_size,
                google_file_uri=uploaded_file.uri,
                google_file_name=uploaded_file.name,
                state=str(uploaded_file.state) if hasattr(uploaded_file, 'state') else "ACTIVE"
            )
            
        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error uploading to Google File API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to Google: {str(e)}"
        )


@router.get("/{file_id}", response_model=FileInfo)
async def get_file_info(
    file_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
    supabase: SupabaseClient = Depends(),
    s3: S3Client = Depends()
):
    """
    Получить информацию о файле и ссылку для скачивания.
    
    Args:
        file_id: UUID файла
        user_id: ID текущего пользователя
        supabase: Клиент Supabase
        s3: Клиент S3
    
    Returns:
        Информация о файле
    
    Raises:
        HTTPException: Если файл не найден
    """
    stored_file = await supabase.get_file(file_id)
    
    if not stored_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    # Генерируем URL для доступа
    url = None
    if stored_file.storage_path:
        url = s3.get_url(stored_file.storage_path)
    elif stored_file.external_url:
        url = stored_file.external_url
    
    return FileInfo(
        id=stored_file.id,
        filename=stored_file.filename,
        mime_type=stored_file.mime_type,
        size_bytes=stored_file.size_bytes,
        source_type=stored_file.source_type,
        storage_path=stored_file.storage_path,
        external_url=url,
        created_at=stored_file.created_at
    )

