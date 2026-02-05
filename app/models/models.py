from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Query(BaseModel):
    # Accept either a list of messages (preferred) or a single `text` field
    messages: Optional[List[Message]] = None
    text: Optional[str] = None

    # quality controls
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512

    # reasoning
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"

    # determinism
    seed: Optional[int] = None

    # optional
    stop: Optional[List[str]] = None


class Response(BaseModel):
    output: str
    tokens_used: Optional[int] = None
