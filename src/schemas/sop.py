from pydantic import BaseModel, Field
from typing import Optional

class SOPGenerateRequest(BaseModel):
    session_id: str = Field(..., example="session-12345")
    target_scholarship: str = Field(..., example="DAAD Research Fellowship")
    additional_notes: Optional[str] = ""
    tone: Optional[str] = Field(default="Professional", description="Tone of the document")
    document_type: Optional[str] = Field(default="Statement of Purpose", description="Type of document to generate")

class SOPResponse(BaseModel):
    generated_text: str