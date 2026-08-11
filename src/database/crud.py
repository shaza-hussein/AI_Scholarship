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



def get_chat_history(db: Session, session_id: str, limit: int = 6):
    messages = db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.created_at.desc()).limit(limit).all()
    
    return messages[::-1]

def save_chat_message(db: Session, session_id: str, role: str, content: str):
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not session:
        session = models.ChatSession(id=session_id, user_session_id=session_id)
        db.add(session)
        db.commit()
        
    message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message