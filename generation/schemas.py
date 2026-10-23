"""Pydantic Models and Response Schemas.

Defines validated structured output schemas for citations, generated RAG responses,
telemetry metadata, and chat requests.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Schema for individual source citations guaranteeing provenance."""
    doc: str = Field(description="Filename of the source document referenced")
    page_or_chapter: str = Field(description="Chapter title or section identifier")
    chunk_id: str = Field(description="Unique identifier of the retrieved chunk")
    quote_snippet: str = Field(description="Exact verbatim substring quote from the source chunk")


class TelemetryData(BaseModel):
    """Telemetry information attached to the response."""
    ttft_ms: float = Field(default=0.0, description="Time to first token in milliseconds")
    duration_s: float = Field(default=0.0, description="Total execution duration in seconds")
    prompt_tokens: int = Field(default=0, description="Total prompt tokens consumed")
    completion_tokens: int = Field(default=0, description="Total completion tokens generated")
    model_name: str = Field(default="", description="Name of the model utilized")
    computed_cost_usd: float = Field(default=0.0, description="Calculated query cost in USD")


class RAGResponse(BaseModel):
    """Validated structured RAG output returned to clients and evaluation harness."""
    answer: str = Field(description="Synthesized natural language answer")
    citations: List[Citation] = Field(default_factory=list, description="List of validated source citations")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    used_fallback: bool = Field(default=False, description="True if response triggered a fallback gate")
    telemetry: Optional[TelemetryData] = Field(default=None, description="Latency and token telemetry")


class ChatRequest(BaseModel):
    """Request payload submitted by the UI or API clients."""
    query: str = Field(..., min_length=1, description="User question or prompt")
    session_id: Optional[str] = Field(default=None, description="Optional conversation session ID")
