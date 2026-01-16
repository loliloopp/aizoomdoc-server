"""
JSON schemas and Pydantic models for structured LLM outputs.
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated


BBoxNorm = Annotated[
    List[float],
    Field(min_length=4, max_length=4, description="Normalized bbox [x1,y1,x2,y2] in 0..1 range"),
]


class LLMBaseModel(BaseModel):
    """Base model for LLM outputs (ignore unknown fields)."""

    model_config = ConfigDict(extra="ignore")


class SelectedBlock(LLMBaseModel):
    """Full block extracted from MD/HTML."""

    block_id: str = Field(..., description="Block ID")
    block_kind: Literal["TEXT", "IMAGE", "TABLE"] = Field(..., description="Block kind")
    page_number: int = Field(..., ge=1, description="Page number")
    content_raw: str = Field(..., description="Full block content")
    linked_block_ids: List[str] = Field(default_factory=list, description="Linked block IDs")


class ImageRequest(LLMBaseModel):
    """Image request by block_id."""

    block_id: str = Field(..., description="Image block ID")
    reason: Optional[str] = Field(default=None, description="Reason for request")
    priority: Literal["high", "medium", "low"] = Field(default="medium", description="Priority")


class ROIRequest(LLMBaseModel):
    """ROI/zoom request."""

    block_id: str = Field(..., description="Image block ID")
    page: Optional[int] = Field(default=1, ge=1, description="Page number")
    bbox_norm: BBoxNorm = Field(..., description="Normalized bbox [x1,y1,x2,y2]")
    dpi: Optional[int] = Field(default=300, ge=72, le=600, description="Render DPI")
    reason: Optional[str] = Field(default=None, description="Reason for request")


class FlashCollectorResponse(LLMBaseModel):
    """Flash-collector output: blocks and media requests."""

    selected_blocks: List[SelectedBlock] = Field(default_factory=list)
    requested_images: List[ImageRequest] = Field(default_factory=list)
    requested_rois: List[ROIRequest] = Field(default_factory=list)
    materials_summary: Optional[str] = Field(default=None)


class MaterialImage(LLMBaseModel):
    """Image entry in materials_json."""

    block_id: str = Field(..., description="Image block ID")
    kind: Literal["overview", "quadrant", "roi"] = Field(..., description="Image kind")
    png_uri: str = Field(..., description="PNG URI in Google File API")
    width: Optional[int] = Field(default=None, ge=1, description="PNG width")
    height: Optional[int] = Field(default=None, ge=1, description="PNG height")
    scale_factor: Optional[float] = Field(default=None, ge=0.0, description="Preview scale factor")
    bbox_norm: Optional[BBoxNorm] = Field(default=None, description="ROI/quadrant bbox")


class MaterialsJSON(LLMBaseModel):
    """Materials for answer (sent to Pro or Flash-answer)."""

    blocks: List[SelectedBlock] = Field(default_factory=list)
    images: List[MaterialImage] = Field(default_factory=list)
    source_documents: Optional[List[str]] = Field(
        default=None, description="Source document IDs or names"
    )


class Citation(LLMBaseModel):
    """Citation referencing a source."""

    block_id: str = Field(..., description="Source block ID")
    kind: Literal["text_block", "image_block", "roi"] = Field(default="text_block")
    page_number: Optional[int] = Field(default=None, ge=1)
    bbox_norm: Optional[BBoxNorm] = Field(default=None)
    note: Optional[str] = Field(default=None)


class Issue(LLMBaseModel):
    """Detected issue/problem."""

    issue_type: str = Field(..., description="Issue type")
    severity: Literal["high", "medium", "low"] = Field(default="medium")
    description: str = Field(..., description="Issue description")
    evidence: List[Citation] = Field(default_factory=list, description="Evidence")


class Recommendation(LLMBaseModel):
    """Recommendation."""

    title: str = Field(..., description="Short title")
    details: Optional[str] = Field(default=None, description="Details")


class DiffItem(LLMBaseModel):
    """Diff item for compare mode."""

    item: str = Field(..., description="Compared item")
    before: Optional[str] = Field(default=None, description="Before state")
    after: Optional[str] = Field(default=None, description="After state")
    impact: Optional[str] = Field(default=None, description="Impact/risks")
    evidence: List[Citation] = Field(default_factory=list)


class AnswerResponse(LLMBaseModel):
    """Structured answer (Flash or Pro)."""

    answer_markdown: str = Field(..., description="Full answer in Markdown")
    citations: List[Citation] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    diff: List[DiffItem] = Field(default_factory=list)
    needs_more_evidence: bool = Field(default=False)
    followup_images: List[str] = Field(default_factory=list, description="Image block IDs")
    followup_rois: List[ROIRequest] = Field(default_factory=list)


def get_flash_collector_schema() -> dict:
    """JSON schema for FlashCollectorResponse."""
    return FlashCollectorResponse.model_json_schema()


def get_answer_schema() -> dict:
    """JSON schema for AnswerResponse."""
    return AnswerResponse.model_json_schema()


def get_materials_schema() -> dict:
    """JSON schema for MaterialsJSON (debug/validation)."""
    return MaterialsJSON.model_json_schema()

