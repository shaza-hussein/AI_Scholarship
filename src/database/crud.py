from sqlalchemy.orm import Session
from src.database import models

def create_or_update_profile(db: Session, session_id: str, profile_data: dict):
    """
    Searches for the student profile based on the session_id. 
    If found, it updates it; if not found, it creates a new one.
    """
    profile = db.query(models.StudentProfile).filter(models.StudentProfile.session_id == session_id).first()
    
    if not profile:
        profile = models.StudentProfile(session_id=session_id)
        db.add(profile)
    
    # Update fields with the extracted data
    profile.nationality = profile_data.get("nationality", profile.nationality)
    profile.academic_level = profile_data.get("academic_level", profile.academic_level)
    profile.academic_major = profile_data.get("academic_major", profile.academic_major)
    profile.gpa = profile_data.get("gpa", profile.gpa)
    profile.research_interests = profile_data.get("research_interests", profile.research_interests)
    
    db.commit()
    db.refresh(profile)
    return profile

def get_profile(db: Session, session_id: str):
    """Retrieves the student data using the session identifier"""
    return db.query(models.StudentProfile).filter(models.StudentProfile.session_id == session_id).first()