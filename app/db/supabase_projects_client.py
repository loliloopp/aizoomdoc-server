"""
Клиент для работы с Supabase Projects DB (read-only).
"""

import logging
from typing import Optional, List
from uuid import UUID

from supabase import create_client, Client as SupabaseClientSDK

from app.config import settings
from app.models.internal import StorageFile

logger = logging.getLogger(__name__)


class SupabaseProjectsClient:
    """Клиент для работы с Supabase Projects DB (read-only)."""
    
    def __init__(self):
        """Инициализация клиента Supabase Projects."""
        # Используем service_key если есть, иначе anon_key
        key = settings.supabase_projects_service_key or settings.supabase_projects_anon_key
        self.client: SupabaseClientSDK = create_client(
            settings.supabase_projects_url,
            key
        )
    
    async def get_tree_nodes(
        self,
        client_id: str,
        parent_id: Optional[UUID] = None,
        node_type: Optional[str] = None
    ) -> List[dict]:
        """
        Получить узлы дерева проектов.
        
        Args:
            client_id: ID клиента
            parent_id: ID родительского узла (None для корневых)
            node_type: Тип узла (client, project, section, stage, task, document)
        
        Returns:
            Список узлов дерева
        """
        try:
            query = self.client.table("tree_nodes").select("*").eq("client_id", client_id)
            
            if parent_id is None:
                query = query.is_("parent_id", "null")
            else:
                query = query.eq("parent_id", str(parent_id))
            
            if node_type:
                query = query.eq("node_type", node_type)
            
            query = query.order("sort_order").order("name")
            response = query.execute()
            
            return response.data
        
        except Exception as e:
            logger.error(f"Error getting tree nodes: {e}")
            return []
    
    async def get_node_by_id(self, node_id: UUID) -> Optional[dict]:
        """Получить узел по ID."""
        try:
            response = self.client.table("tree_nodes").select("*").eq("id", str(node_id)).execute()
            
            if not response.data:
                return None
            
            return response.data[0]
        
        except Exception as e:
            logger.error(f"Error getting node by ID: {e}")
            return None
    
    async def get_node_files(self, node_id: UUID) -> List[dict]:
        """
        Получить файлы узла (PDF, аннотации, результаты, кропы).
        
        Args:
            node_id: UUID узла
        
        Returns:
            Список файлов узла
        """
        try:
            response = (
                self.client.table("node_files")
                .select("*")
                .eq("node_id", str(node_id))
                .order("created_at", desc=True)
                .execute()
            )
            
            return response.data
        
        except Exception as e:
            logger.error(f"Error getting node files: {e}")
            return []
    
    async def get_document_results(self, document_node_id: UUID) -> List[dict]:
        """
        Получить файлы результатов обработки документа.
        
        Args:
            document_node_id: UUID узла документа
        
        Returns:
            Список файлов результатов (annotation, ocr_html, result_md, result_json, crops)
        """
        try:
            # Фильтруем только файлы результатов
            result_file_types = [
                "annotation",
                "ocr_html",
                "result_md",
                "result_json",
                "crops_folder"
            ]
            
            response = (
                self.client.table("node_files")
                .select("*")
                .eq("node_id", str(document_node_id))
                .in_("file_type", result_file_types)
                .execute()
            )
            
            return response.data
        
        except Exception as e:
            logger.error(f"Error getting document results: {e}")
            return []

    async def get_document_crops(self, document_node_id: UUID) -> List[dict]:
        """Получить кропы (изображения) документа."""
        try:
            response = (
                self.client.table("node_files")
                .select("*")
                .eq("node_id", str(document_node_id))
                .eq("file_type", "crop")
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error getting document crops: {e}")
            return []
    
    async def search_documents(
        self,
        client_id: str,
        query: str,
        limit: int = 20
    ) -> List[dict]:
        """
        Поиск документов по имени или коду.
        
        Args:
            client_id: ID клиента
            query: Поисковый запрос
            limit: Максимальное количество результатов
        
        Returns:
            Список найденных документов
        """
        try:
            # Используем текстовый поиск по имени и коду
            response = (
                self.client.table("tree_nodes")
                .select("*")
                .eq("client_id", client_id)
                .eq("node_type", "document")
                .or_(f"name.ilike.%{query}%,code.ilike.%{query}%")
                .limit(limit)
                .execute()
            )
            
            return response.data
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

