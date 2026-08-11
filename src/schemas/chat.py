from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    session_id: str = Field(..., example="session-12345")
    message: str = Field(..., example="Can you compare the DAAD and Chevening scholarships?")
    intent: Optional[str] = Field(None, description="Optional: 'compare', 'roadmap', 'explain'")

class ChatResponse(BaseModel):
    reply: str