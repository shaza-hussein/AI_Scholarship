from pydantic import BaseModel, Field
from typing import Optional, List

class ProfileData(BaseModel):
    nationality: Optional[str] = Field(None, example="Egypt")
    academic_level: Optional[str] = Field(None, example="Bachelor")
    academic_major: Optional[str] = Field(None, example="Business Administration")
    gpa: Optional[float] = Field(None, example=3.5)
    research_interests: Optional[str] = Field(None, example="Digital Marketing and Leadership")

class ProfileUpdateRequest(BaseModel):
    session_id: str = Field(..., example="session-12345")
    profile: ProfileData

class SearchRequest(BaseModel):
    session_id: str = Field(..., example="session-12345")
    query: str = Field(..., example="Fully funded AI master scholarships in Germany")
    top_k: int = Field(6, description="Number of scholarship results to return")

class ScholarshipCardResponse(BaseModel):
    title: str
    country: Optional[str] = None
    academic_level: Optional[str] = None
    funding_type: Optional[str] = None
    description: Optional[str] = None
    deadline: Optional[str] = None
    url: Optional[str] = None
    match_score: float = Field(..., description="Relevance score converted or raw from reranker")

class SearchResponse(BaseModel):
    total_found: int
    scholarships: List[ScholarshipCardResponse]

class DocumentGenerateRequest(BaseModel):
    session_id: str = Field(..., example="session-12345")
    target_scholarship: str = Field(..., example="DAAD Research Fellowship")
    document_type: str = Field(..., example="Statement of Purpose")
    tone: str = Field(..., example="Professional")
    additional_notes: Optional[str] = Field(None, example="Highlight my recent AI publication.")

class CVAnalyzeResponse(BaseModel):
    status: str
    session_id: str
    profile: ProfileData