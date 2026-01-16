"""
HTML OCR parsing utilities.
"""

from __future__ import annotations

import html as html_module
import json
import logging
import re
from typing import Dict, Optional, Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class HtmlOcrService:
    """Extracts image crop URLs from HTML OCR files."""

    HEADER_PATTERN_OLD = re.compile(
        r'Блок\s+#(\d+)\s+\(стр\.\s+(\d+)\)\s+\|\s+Тип:\s+(\w+)\s+\|\s+ID:\s+([\w-]+)',
        re.IGNORECASE,
    )
    HEADER_PATTERN_NEW = re.compile(
        r'Блок\s+#(\d+)\s+\(стр\.\s+(\d+)\)\s+\|\s+Тип:\s+(\w+)',
        re.IGNORECASE,
    )
    BLOCK_ID_PATTERN = re.compile(r'BLOCK:\s+([\w-]+)', re.IGNORECASE)

    @classmethod
    def extract_image_map(cls, html_text: str) -> Dict[str, str]:
        if not html_text:
            return {}
        soup = BeautifulSoup(html_text, "html.parser")
        image_map: Dict[str, str] = {}

        for block_div in soup.find_all("div", class_="block"):
            header_div = block_div.find("div", class_="block-header")
            content_div = block_div.find("div", class_="block-content")
            if not header_div or not content_div:
                continue

            header_text = header_div.get_text(strip=True)
            block_type, block_id = cls._parse_header(header_text)

            # Prefer block id from content if present.
            content_text = content_div.get_text(" ", strip=True)
            id_match = cls.BLOCK_ID_PATTERN.search(content_text)
            if id_match:
                block_id = id_match.group(1)

            if block_type != "image" or not block_id:
                continue

            crop_url = cls._extract_crop_url(content_div)
            if crop_url:
                image_map[block_id] = crop_url

        return image_map

    @classmethod
    def _parse_header(cls, header_text: str) -> tuple[Optional[str], Optional[str]]:
        match = cls.HEADER_PATTERN_OLD.search(header_text)
        if match:
            block_type = match.group(3)
            block_id = match.group(4)
            return block_type.lower(), block_id
        match = cls.HEADER_PATTERN_NEW.search(header_text)
        if match:
            block_type = match.group(3)
            return block_type.lower(), None
        return None, None

    @classmethod
    def _extract_crop_url(cls, content_div) -> Optional[str]:
        # Prefer JSON in <pre> if present.
        pre_elem = content_div.find("pre")
        if pre_elem:
            json_text = html_module.unescape(pre_elem.get_text())
            json_text = re.sub(r"^```[a-zA-Z]*\s*", "", json_text, flags=re.MULTILINE)
            json_text = re.sub(r"^```\s*", "", json_text, flags=re.MULTILINE).strip()
            crop_url = cls._find_crop_url_in_json(json_text)
            if crop_url:
                return crop_url

        # Fallback: any link to a PDF/image.
        for a_tag in content_div.find_all("a", href=True):
            href = a_tag["href"]
            if cls._looks_like_media_url(href):
                return href

        return None

    @classmethod
    def _find_crop_url_in_json(cls, json_text: str) -> Optional[str]:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = cls._parse_multiple_json(json_text)
        return cls._find_crop_url_recursive(data)

    @classmethod
    def _parse_multiple_json(cls, text: str) -> Any:
        results = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            try:
                obj, idx = decoder.raw_decode(text, pos)
                results.append(obj)
                pos = idx
            except json.JSONDecodeError:
                next_brace = text.find("{", pos + 1)
                if next_brace == -1:
                    break
                pos = next_brace
        if len(results) == 1:
            return results[0]
        return results

    @classmethod
    def _find_crop_url_recursive(cls, data: Any) -> Optional[str]:
        if isinstance(data, dict):
            for key in ("crop_url", "cropUrl", "crop_url_pdf", "cropUrlPdf"):
                value = data.get(key)
                if isinstance(value, str) and cls._looks_like_media_url(value):
                    return value
            for value in data.values():
                found = cls._find_crop_url_recursive(value)
                if found:
                    return found
        elif isinstance(data, list):
            for item in data:
                found = cls._find_crop_url_recursive(item)
                if found:
                    return found
        return None

    @staticmethod
    def _looks_like_media_url(url: str) -> bool:
        lower = url.lower()
        return lower.endswith(".pdf") or lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".webp")

