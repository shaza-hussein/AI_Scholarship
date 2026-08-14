import os
import duckdb
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api.dependencies import get_generator, get_db
from src.rag.generator import ScholarshipGenerator
from src.schemas.sop import SOPGenerateRequest, SOPResponse
from src.database import crud

router = APIRouter(prefix="/api/v1", tags=["Document Generator"])

def get_scholarship_details(scholarship_name: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    duckdb_path = os.path.join(project_root, "db", "analytical.duckdb")
    
    try:
        conn = duckdb.connect(duckdb_path)
        query = "SELECT scholarship_name, application_process FROM scholarships WHERE scholarship_name = ?"
        result = conn.execute(query, [scholarship_name]).fetchone()
        
        if result:
            return f"Scholarship Name: {result[0]}\nApplication Process/Details: {result[1]}"
        
        return f"Scholarship Name: {scholarship_name}\nDetails not found."
    except Exception as e:
        print(f"Database error: {e}")
        return scholarship_name
    finally:
        if 'conn' in locals():
            conn.close()

@router.post("/generate-document", response_model=SOPResponse)
async def generate_application_document(
    payload: SOPGenerateRequest,
    generator: ScholarshipGenerator = Depends(get_generator),
    db: Session = Depends(get_db)
):
    try:
        db_profile = crud.get_profile(db, payload.session_id)
        if not db_profile:
            raise HTTPException(
                status_code=404, 
                detail="Student profile not found. Please analyze CV first."
            )
        
        profile_dict = {
            "nationality": db_profile.nationality,
            "academic_level": db_profile.academic_level,
            "academic_major": db_profile.academic_major,
            "gpa": db_profile.gpa,
            "research_interests": db_profile.research_interests,

            "target_countries": db_profile.target_countries,
            "skills": db_profile.skills,
            "age": db_profile.age,
        }
        
        scholarship_details = get_scholarship_details(payload.target_scholarship)
        
        generated_text = generator.generate_sop(
            profile_data=profile_dict,
            scholarship_details=scholarship_details,
            additional_notes=payload.additional_notes,
            tone=payload.tone,
            document_type=payload.document_type
        )
        
        return SOPResponse(generated_text=generated_text)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))