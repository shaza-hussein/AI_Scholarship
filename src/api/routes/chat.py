from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api.dependencies import get_generator, get_retriever, get_db
from src.rag.generator import ScholarshipGenerator
from src.rag.retriever import ScholarshipRetriever
from src.schemas.chat import ChatRequest, ChatResponse
from src.schemas.student import StudentProfile
from src.database import crud

router = APIRouter(prefix="/api/v1", tags=["AI Chat Assistant"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    payload: ChatRequest,
    generator: ScholarshipGenerator = Depends(get_generator),
    retriever: ScholarshipRetriever = Depends(get_retriever),
    db: Session = Depends(get_db)
):
    try:
        crud.save_chat_message(db=db, session_id=payload.session_id, role="user", content=payload.message)
        
        db_profile = crud.get_profile(db, payload.session_id)
        profile_dict = {}
        profile_obj = StudentProfile(nationality="Unknown", academic_level="Bachelor", academic_major="Unknown", gpa=0.0)
        
        if db_profile:
            profile_dict = {
                "nationality": db_profile.nationality,
                "academic_level": db_profile.academic_level,
                "academic_major": db_profile.academic_major,
                "gpa": db_profile.gpa,
                "research_interests": db_profile.research_interests
            }
            profile_obj = StudentProfile(**profile_dict)

        chat_history = crud.get_chat_history(db, payload.session_id, limit=6)
        
        retrieved_docs = retriever.match_scholarships(profile=profile_obj, query=payload.message, top_k=4)
        
        ai_reply = generator.chat_orchestrator(
            profile_data=profile_dict,
            chat_history=chat_history,
            retrieved_docs=retrieved_docs,
            user_message=payload.message,
            intent=payload.intent
        )
        
        crud.save_chat_message(db=db, session_id=payload.session_id, role="assistant", content=ai_reply)
        
        return ChatResponse(reply=ai_reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))