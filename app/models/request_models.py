from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=3, ge=1, le=10)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class DocumentMetadata(BaseModel):
    id: str
    source: str
    type: Optional[str] = None
