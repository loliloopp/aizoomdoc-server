"""
Evidence service for rendering PDF crops to PNG and generating preview/quadrants/ROI.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import fitz  # PyMuPDF
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RenderedImage:
    """Rendered image bytes and metadata."""

    kind: str
    png_bytes: bytes
    width: int
    height: int
    scale_factor: float
    bbox_norm: Optional[list[float]] = None


class EvidenceService:
    """Render PDF crops to PNG and generate preview/quadrants/ROI."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "aizoomdoc_evidence_cache"
        self.cache_dir = cache_dir
        self.renders_dir = cache_dir / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _render_cache_path(self, cache_key: str, page: int, dpi: int) -> Path:
        safe = self._hash_key(f"{cache_key}:{page}:{dpi}")
        return self.renders_dir / f"{safe}.png"

    def render_pdf_page(
        self,
        pdf_bytes: bytes,
        *,
        cache_key: str,
        page: int = 0,
        dpi: int = 150,
    ) -> Image.Image:
        """Render a PDF page to PIL Image with disk caching."""
        cache_path = self._render_cache_path(cache_key, page, dpi)
        if cache_path.exists():
            return Image.open(cache_path).convert("RGB")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_obj = doc.load_page(page)
            zoom = dpi / 72.0
            pix = page_obj.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(cache_path, format="PNG")
            return img
        finally:
            doc.close()

    def _scale_to_max_side(self, img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
        w, h = img.size
        max_dim = max(w, h)
        if max_dim <= max_side:
            return img, 1.0
        scale = max_dim / float(max_side)
        new_w = max(1, int(w / scale))
        new_h = max(1, int(h / scale))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = img.resize((new_w, new_h), resample=resample)
        return resized, scale

    def build_preview_and_quadrants(
        self,
        pdf_bytes: bytes,
        *,
        cache_key: str,
        page: int = 0,
        dpi: int = 150,
    ) -> list[RenderedImage]:
        """Generate preview and optional quadrants from a PDF crop."""
        base_img = self.render_pdf_page(pdf_bytes, cache_key=cache_key, page=page, dpi=dpi)
        w, h = base_img.size

        preview_img, scale_factor = self._scale_to_max_side(base_img, settings.preview_max_side)
        preview_bytes = self._to_png_bytes(preview_img)
        results = [
            RenderedImage(
                kind="overview",
                png_bytes=preview_bytes,
                width=preview_img.size[0],
                height=preview_img.size[1],
                scale_factor=scale_factor,
                bbox_norm=None,
            )
        ]

        if scale_factor > settings.auto_quadrants_threshold:
            quadrants = [
                ([0.0, 0.0, 0.55, 0.55], "quadrant"),
                ([0.45, 0.0, 1.0, 0.55], "quadrant"),
                ([0.0, 0.45, 0.55, 1.0], "quadrant"),
                ([0.45, 0.45, 1.0, 1.0], "quadrant"),
            ]
            for bbox_norm, kind in quadrants:
                crop = self._crop_norm(base_img, bbox_norm)
                crop_img, crop_scale = self._scale_to_max_side(crop, settings.zoom_preview_max_side)
                crop_bytes = self._to_png_bytes(crop_img)
                results.append(
                    RenderedImage(
                        kind=kind,
                        png_bytes=crop_bytes,
                        width=crop_img.size[0],
                        height=crop_img.size[1],
                        scale_factor=crop_scale,
                        bbox_norm=bbox_norm,
                    )
                )
        return results

    def build_roi(
        self,
        pdf_bytes: bytes,
        *,
        cache_key: str,
        bbox_norm: Iterable[float],
        page: int = 0,
        dpi: int = 300,
    ) -> RenderedImage:
        """Render ROI from PDF at requested DPI and return PNG bytes."""
        base_img = self.render_pdf_page(pdf_bytes, cache_key=cache_key, page=page, dpi=dpi)
        crop = self._crop_norm(base_img, list(bbox_norm))
        crop_img, crop_scale = self._scale_to_max_side(crop, settings.zoom_preview_max_side)
        crop_bytes = self._to_png_bytes(crop_img)
        return RenderedImage(
            kind="roi",
            png_bytes=crop_bytes,
            width=crop_img.size[0],
            height=crop_img.size[1],
            scale_factor=crop_scale,
            bbox_norm=list(bbox_norm),
        )

    def _crop_norm(self, img: Image.Image, bbox_norm: list[float]) -> Image.Image:
        x1, y1, x2, y2 = bbox_norm
        w, h = img.size
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bbox_norm for ROI")
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)
        return img.crop((left, top, right, bottom))

    def _to_png_bytes(self, img: Image.Image) -> bytes:
        from io import BytesIO

        output = BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

