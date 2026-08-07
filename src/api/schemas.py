from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    nationality: str = Field(..., example="Egypt")
    academic_level: str = Field(..., example="Bachelor")
    academic_major: str = Field(..., example="Business Administration")
    gpa: float = Field(..., example=3.5)
    research_interests: Optional[str] = Field(None, example="Digital Marketing and Leadership")
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