from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from src.api.dependencies import get_generator, get_db
from src.rag.generator import ScholarshipGenerator
from src.services.pdf_service import extract_text_from_pdf_bytes
from src.database import crud
from src.api.schemas import CVAnalyzeResponse

router = APIRouter(prefix="/api/v1", tags=["CV Analysis"])

@router.post("/analyze-cv", response_model=CVAnalyzeResponse)
async def analyze_student_cv(
    cv_file: UploadFile = File(...),
    session_id: str = Form(...),
    generator: ScholarshipGenerator = Depends(get_generator),
    db: Session = Depends(get_db)
):
    """
    Endpoint to upload a PDF CV, extract text in-memory, 
    parse it into a StudentProfile JSON using Groq Llama-3,
    and save it to the database linked to the user's session_id.
    """
    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        pdf_bytes = await cv_file.read()
        
        cv_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if len(cv_text) < 50:
            raise HTTPException(status_code=400, detail="CV text is too short or unreadable.")
            
        profile_dict = generator.parse_cv_text(cv_text)
        
        saved_profile = crud.create_or_update_profile(db=db, session_id=session_id, profile_data=profile_dict)
        
        return CVAnalyzeResponse(
            status="success",
            session_id=saved_profile.session_id,
            profile={
                "nationality": saved_profile.nationality,
                "academic_level": saved_profile.academic_level,
                "academic_major": saved_profile.academic_major,
                "gpa": saved_profile.gpa,
                "research_interests": saved_profile.research_interests
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))