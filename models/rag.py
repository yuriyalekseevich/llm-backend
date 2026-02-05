from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class ChunkIn(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class ChunkOut(ChunkIn):
    score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    hits: List[ChunkOut]