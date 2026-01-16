"""
Search utilities for document blocks (MD/HTML).
"""

from dataclasses import dataclass, field
import logging
import re
from typing import List, Dict, Any, Optional, Iterable

from app.db.supabase_projects_client import SupabaseProjectsClient
from app.db.s3_client import S3Client
from app.models.internal import SearchResult, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class ParsedBlock:
    block_id: str
    block_kind: str
    page_number: int
    content_raw: str
    linked_block_ids: List[str] = field(default_factory=list)


class SearchService:
    """Search and coverage utilities for document blocks."""

    def __init__(self, projects_db: SupabaseProjectsClient, s3_client: S3Client):
        self.projects_db = projects_db
        self.s3_client = s3_client

    def parse_md_blocks(self, text: str) -> List[ParsedBlock]:
        if not text:
            return []

        blocks: List[ParsedBlock] = []
        page_number: Optional[int] = None
        lines = text.splitlines()
        block_header = re.compile(r"^###\s+BLOCK\s+\[(TEXT|IMAGE|TABLE)\]:\s+([A-Z0-9-]+)")
        page_header = re.compile(r"^##\s+.*?(\d+)\s*$")
        link_re = re.compile(r"\u2192([A-Z0-9-]+)")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("## "):
                match = page_header.match(line)
                if match:
                    try:
                        page_number = int(match.group(1))
                    except ValueError:
                        page_number = None
                i += 1
                continue

            match = block_header.match(line)
            if match:
                block_kind, block_id = match.group(1), match.group(2)
                content_lines: List[str] = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if block_header.match(next_line) or next_line.startswith("## "):
                        break
                    content_lines.append(lines[i])
                    i += 1
                content_raw = "\n".join(content_lines).strip()
                linked_ids = link_re.findall(content_raw)
                blocks.append(
                    ParsedBlock(
                        block_id=block_id,
                        block_kind=block_kind,
                        page_number=page_number or 1,
                        content_raw=content_raw,
                        linked_block_ids=linked_ids,
                    )
                )
                continue
            i += 1

        return blocks

    def build_block_map(self, blocks: Iterable[ParsedBlock]) -> Dict[str, ParsedBlock]:
        return {block.block_id: block for block in blocks}

    def extract_terms(self, query: str) -> List[str]:
        terms = re.findall(r"\w+", query.lower())
        return [t for t in terms if len(t) >= 2]

    def score_block(self, block: ParsedBlock, terms: List[str], preferred_pages: Optional[set[int]] = None) -> float:
        content = block.content_raw.lower()
        hits = sum(1 for t in terms if t in content)
        score = float(hits)
        if preferred_pages and block.page_number in preferred_pages:
            score += 1.5
        if len(content) < 20:
            score -= 0.5
        return score

    def find_linked_blocks(self, selected_ids: set[str], block_map: Dict[str, ParsedBlock]) -> set[str]:
        linked_ids: set[str] = set()
        for block_id in selected_ids:
            block = block_map.get(block_id)
            if block:
                linked_ids.update(block.linked_block_ids)
        # Reverse links
        for block in block_map.values():
            for link_id in block.linked_block_ids:
                if link_id in selected_ids:
                    linked_ids.add(block.block_id)
        return linked_ids

    def suggest_additional_blocks(
        self,
        *,
        blocks: List[ParsedBlock],
        selected_ids: set[str],
        query: str,
        preferred_pages: Optional[set[int]] = None,
        max_add: int = 10,
    ) -> List[ParsedBlock]:
        terms = self.extract_terms(query)
        scored: List[tuple[float, ParsedBlock]] = []
        for block in blocks:
            if block.block_id in selected_ids:
                continue
            score = self.score_block(block, terms, preferred_pages)
            if score >= 2.0:
                scored.append((score, block))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:max_add]]

    async def search_in_documents(
        self,
        query: str,
        client_id: str,
        document_ids: Optional[List[str]] = None
    ) -> SearchResult:
        text_blocks: List[TextBlock] = []

        doc_ids = document_ids or []
        if not doc_ids:
            docs = await self.projects_db.search_documents(client_id=client_id, query=query, limit=10)
            doc_ids = [d.get("id") for d in docs if d.get("id")]

        terms = self.extract_terms(query)

        for doc_id in doc_ids:
            files = await self.projects_db.get_document_results(doc_id)
            md_file = next((f for f in files if f.get("file_type") == "result_md"), None)
            html_file = next((f for f in files if f.get("file_type") == "ocr_html"), None)

            doc_text = ""
            if md_file and md_file.get("r2_key"):
                data = await self.s3_client.download_bytes(md_file["r2_key"])
                if data:
                    doc_text = data.decode("utf-8", errors="ignore")
            elif html_file and html_file.get("r2_key"):
                data = await self.s3_client.download_bytes(html_file["r2_key"])
                if data:
                    doc_text = data.decode("utf-8", errors="ignore")

            if not doc_text:
                continue

            blocks = self.parse_md_blocks(doc_text)
            for block in blocks:
                if block.block_kind not in ("TEXT", "TABLE"):
                    continue
                score = self.score_block(block, terms)
                if score >= 2.0:
                    text_blocks.append(
                        TextBlock(
                            text=block.content_raw,
                            block_id=block.block_id,
                            page=block.page_number or None,
                            metadata={"score": score, "document_id": str(doc_id)},
                        )
                    )

        return SearchResult(
            text_blocks=text_blocks,
            images=[],
            query=query,
            total_blocks_found=len(text_blocks),
        )

    async def extract_context_from_block(
        self,
        block_id: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        files = await self.projects_db.get_document_results(document_id)
        md_file = next((f for f in files if f.get("file_type") == "result_md"), None)
        if not md_file or not md_file.get("r2_key"):
            return None
        data = await self.s3_client.download_bytes(md_file["r2_key"])
        if not data:
            return None
        blocks = self.parse_md_blocks(data.decode("utf-8", errors="ignore"))
        block_map = self.build_block_map(blocks)
        block = block_map.get(block_id)
        if not block:
            return None
        return {
            "block_id": block.block_id,
            "page_number": block.page_number,
            "content_raw": block.content_raw,
            "linked_block_ids": block.linked_block_ids,
        }

