from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.api.dependencies import get_db
from src.api.schemas import ProfileUpdateRequest
from src.database import crud

router = APIRouter(prefix="/api/v1", tags=["Student Profile"])

@router.post("/update-profile")
def update_student_profile(payload: ProfileUpdateRequest, db: Session = Depends(get_db)):
    """
    Endpoint to handle manual form submissions or profile updates.
    Overwrites the existing data for the given session_id or creates a new one.
    """
    try:
        profile_data = payload.profile.model_dump(exclude_unset=True)
        
        saved_profile = crud.create_or_update_profile(
            db=db, 
            session_id=payload.session_id, 
            profile_data=profile_data
        )
        
        return {
            "status": "success",
            "message": "Profile saved successfully.",
            "session_id": saved_profile.session_id,
            "profile": {
                "nationality": saved_profile.nationality,
                "academic_level": saved_profile.academic_level,
                "academic_major": saved_profile.academic_major,
                "gpa": saved_profile.gpa,
                "research_interests": saved_profile.research_interests
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{session_id}")
def get_student_profile(session_id: str, db: Session = Depends(get_db)):
    """
    Endpoint to fetch the current profile data to populate the frontend form on page reload.
    """
    db_profile = crud.get_profile(db, session_id)
    if not db_profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
        
    return {
        "nationality": db_profile.nationality,
        "academic_level": db_profile.academic_level,
        "academic_major": db_profile.academic_major,
        "gpa": db_profile.gpa,
        "research_interests": db_profile.research_interests
    }