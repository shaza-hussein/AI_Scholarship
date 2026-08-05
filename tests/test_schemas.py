"""
Schema Validation Test Script.
Verifies that the StudentProfile correctly standardizes inputs 
and catches invalid data.
"""

import sys
import os
from pydantic import ValidationError

# Adjust path to import from src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.schemas.student import StudentProfile

def test_student_profile():
    print(">>> Test 1: Valid Input & Auto-Standardization <<<")
    try:
        # Intentionally using lowercase and non-standard synonyms
        student = StudentProfile(
            nationality="syria", 
            academic_level="msc",  # Should be converted to 'Master'
            academic_major="Artificial Intelligence",
            gpa=3.8,
            target_countries=["germany", "netherlands"],
            skills=["Python", "Machine Learning"]
        )
        print("Status: SUCCESS (Data Accepted)")
        print(f"Original input 'syria' -> Converted to: '{student.nationality}'")
        print(f"Original input 'msc' -> Converted to: '{student.academic_level}'")
        print(f"Original input ['germany', ...] -> Converted to: {student.target_countries}\n")
    except ValidationError as e:
        print(f"Failed Test 1: {e}\n")

    print(">>> Test 2: Invalid Input (Testing Strict Validation) <<<")
    try:
        # Intentionally inserting an invalid GPA (> 4.0)
        invalid_student = StudentProfile(
            nationality="Egypt",
            academic_level="Bachelor",
            academic_major="Engineering",
            gpa=4.5  # This should trigger an error
        )
        print("Status: FAILED (System accepted invalid data!)")
    except ValidationError as e:
        print("Status: SUCCESS (System caught the invalid data)")
        print(f"Error Details:\n{e}")

if __name__ == "__main__":
    test_student_profile()