from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Comment: This model represents input for adding a single document chunk.
# We use UUID or custom IDs for uniqueness.
class ChunkIn(BaseModel):
    """Input representation of a single document chunk."""

    id: str = Field(..., description="Unique identifier of the chunk (e.g., UUID)")
    text: str = Field(..., description="The actual text content of the chunk", min_length=1)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional arbitrary metadata (e.g., source, page, date)"
    )

# Comment: This extends ChunkIn for output, adding a score for relevance.
# Score is normalized between 0-1 (lower is better for cosine distance).
class ChunkOut(ChunkIn):
    """Output representation of a retrieved chunk with relevance score."""

    score: Optional[float] = Field(
        None,
        description="Relevance/distance score (lower = more similar in cosine space)",
        ge=0.0,
        le=1.0,  # Adjust if using other metrics
    )

# Comment: Request model for search queries.
# top_k is constrained to prevent overload.
class QueryRequest(BaseModel):
    """Request payload for semantic search / RAG query."""

    query: str = Field(..., description="The user's natural language question/query", min_length=1)
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )

# Comment: Response model for search results.
# Hits are sorted by relevance (best first).
class QueryResponse(BaseModel):
    """Response containing the original query and ranked results."""

    query: str = Field(..., description="The query string that was searched")
    hits: List[ChunkOut] = Field(
        default_factory=list,
        description="List of retrieved chunks sorted by relevance (best first)"
    )
    # Optional fields for future expansion (e.g., performance metrics)
    # total_hits: Optional[int] = None
    # took_ms: Optional[float] = None