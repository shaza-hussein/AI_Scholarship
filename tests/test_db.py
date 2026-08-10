from src.database.database import SessionLocal
from src.database.models import StudentProfile

def test_database():
    print("Testing Database Connection...")
    db = SessionLocal()
    
    try:
        # 1. Create a dummy session ID and profile
        test_session_id = "test-session-12345"
        
        # Check if it already exists to prevent duplicate errors if run twice
        existing_profile = db.query(StudentProfile).filter(StudentProfile.session_id == test_session_id).first()
        
        if not existing_profile:
            new_profile = StudentProfile(
                session_id=test_session_id,
                nationality="Syrian",
                academic_level="Master",
                academic_major="Artificial Intelligence",
                gpa=3.9
            )
            db.add(new_profile)
            db.commit()
            db.refresh(new_profile)
            print(f"SUCCESS: Profile created with ID {new_profile.id}")
        else:
            print("Profile already exists in the database.")
            
        # 2. Retrieve the data
        retrieved = db.query(StudentProfile).filter(StudentProfile.session_id == "test-session-999").first()
        print(f"RETRIEVED DATA: Major -> {retrieved.academic_major}, GPA -> {retrieved.gpa}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    test_database()