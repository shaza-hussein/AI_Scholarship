from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api.schemas import SearchRequest, SearchResponse, ScholarshipCardResponse
from src.api.dependencies import get_retriever, get_db
from src.rag.retriever import ScholarshipRetriever
from src.schemas.student import StudentProfile as PydanticStudentProfile
from src.database import crud

router = APIRouter(prefix="/api/v1", tags=["Scholarships Search"])

@router.post("/search", response_model=SearchResponse)
def search_scholarships(
    payload: SearchRequest, 
    retriever: ScholarshipRetriever = Depends(get_retriever),
    db: Session = Depends(get_db)
):
    try:
        db_profile = crud.get_profile(db, payload.session_id)
        
        if db_profile:
            profile = PydanticStudentProfile(
                nationality=db_profile.nationality,
                academic_level=db_profile.academic_level,
                academic_major=db_profile.academic_major,
                gpa=db_profile.gpa,
                research_interests=db_profile.research_interests
            )
        else:
            profile = PydanticStudentProfile(
                nationality="Unknown",
                academic_level="Unknown",
                academic_major="Unknown",
                gpa=0.0,
                research_interests=""
            )
            
        docs = retriever.match_scholarships(profile, query=payload.query, top_k=payload.top_k)
        
        cards = []
        for doc in docs:
            metadata = doc.metadata
            raw_score = float(metadata.get("rerank_score", getattr(doc, 'score', 0.90)))
            
            card = ScholarshipCardResponse(
                title=metadata.get("scholarship_name", "Unknown Scholarship"),
                country=metadata.get("host_country", "Not Specified"),
                academic_level=metadata.get("academic_level", profile.academic_level),
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