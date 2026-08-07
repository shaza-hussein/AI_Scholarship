from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import SearchRequest, SearchResponse, ScholarshipCardResponse
from src.api.dependencies import get_retriever
from src.rag.retriever import ScholarshipRetriever
from src.schemas.student import StudentProfile

router = APIRouter(prefix="/api/v1", tags=["Scholarships Search"])

@router.post("/search", response_model=SearchResponse)
def search_scholarships(payload: SearchRequest, retriever: ScholarshipRetriever = Depends(get_retriever)):
    """
    Endpoint to search scholarships using natural language and student profile constraints.
    Feeds the frontend 'Search Scholarships' view.
    """
    try:
        # 1. Construct Student Profile object
        profile = StudentProfile(
            nationality=payload.nationality,
            academic_level=payload.academic_level,
            academic_major=payload.academic_major,
            gpa=payload.gpa,
            research_interests=payload.research_interests
        )
        
        # 2. Execute retrieval and re-ranking via hybrid retriever
        docs = retriever.match_scholarships(profile, top_k=payload.top_k)
        
# 3. Format results into cards matching the UI requirements
        cards = []
        for doc in docs:
            metadata = doc.metadata
            # Calculate a clean percentage or use the score directly
            raw_score = float(metadata.get("rerank_score", doc.score if hasattr(doc, 'score') else 0.90))
            
            card = ScholarshipCardResponse(
                title=metadata.get("scholarship_name", "Unknown Scholarship"),
                country=metadata.get("host_country", "Not Specified"),
                academic_level=metadata.get("academic_level", payload.academic_level),
                funding_type=metadata.get("funding_category", "Not Specified"),
                description=doc.page_content[:200] + "...",
                deadline=metadata.get("standardized_deadline", "Not Specified"),
                url=metadata.get("application_link", "#"),
                match_score=raw_score
            )
            cards.append(card)
            
        return SearchResponse(
            total_found=len(cards),
            scholarships=cards
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))