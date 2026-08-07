from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from src.api.dependencies import get_generator
from src.rag.generator import ScholarshipGenerator
from src.schemas.student import StudentProfile
from src.services.pdf_service import extract_text_from_pdf_bytes

router = APIRouter(prefix="/api/v1", tags=["CV Analysis"])

@router.post("/analyze-cv", response_model=StudentProfile)
async def analyze_student_cv(
    cv_file: UploadFile = File(...), 
    generator: ScholarshipGenerator = Depends(get_generator)
):
    """
    Endpoint to upload a PDF CV, extract text in-memory, 
    and use Groq Llama-3 to parse it into a StudentProfile JSON.
    """
    if not cv_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # 1. Read file bytes directly into memory
        pdf_bytes = await cv_file.read()
        
        # 2. Extract text using our util function
        cv_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if len(cv_text) < 50:
            raise HTTPException(status_code=400, detail="CV text is too short or unreadable.")
            
        # 3. Parse text using the AI Generator
        profile_dict = generator.parse_cv_text(cv_text)
        
        # 4. Validate and return using Pydantic Schema
        return StudentProfile(**profile_dict)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))