from pydantic import BaseModel, Field
from typing import Optional
from src.schemas.student import StudentProfile

class SOPGenerateRequest(BaseModel):
    student_profile: StudentProfile
    scholarship_details: str
    additional_notes: Optional[str] = ""
    tone: Optional[str] = Field(default="Professional", description="Tone of the document")
    document_type: Optional[str] = Field(default="Statement of Purpose", description="Type of document to generate")

class SOPResponse(BaseModel):
    generated_text: str