"""
Data Schemas for Student Profile using Pydantic V2.

This module enforces strict data typing and standardization. It acts as the 
first line of defense, ensuring that any payload (whether from the UI form 
or the CV parsing service) is clean, valid, and correctly formatted before 
hitting the retrieval engines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

class StudentProfile(BaseModel):
    """
    Comprehensive data contract for the student matching process.
    Divided into deterministic fields (for SQL filtering) and semantic fields 
    (for Vector similarity).
    """
    
    # --- Deterministic Fields (DuckDB Filtering) ---
    
    nationality: str = Field(
        ..., 
        description="The country of citizenship.",
        examples=["Syria", "Egypt", "Germany"]
    )
    
    academic_level: str = Field(
        ..., 
        description="The target degree the student is applying for.",
        examples=["Master", "PhD", "Bachelor"]
    )
    
    academic_major: str = Field(
        ..., 
        description="The primary field of study or intended major.",
        examples=["Computer Science", "Artificial Intelligence"]
    )
    
    gpa: float = Field(
        ..., 
        ge=0.0, 
        le=4.0, 
        description="Grade Point Average scaled out of 4.0.",
        examples=[3.8]
    )
    
    target_countries: Optional[List[str]] = Field(
        default_factory=list, 
        description="A list of preferred countries for the scholarship.",
        examples=[["Germany", "Netherlands"]]
    )
    
    # --- Semantic Fields (ChromaDB Vector Search) ---
    
    research_interests: Optional[str] = Field(
        default=None, 
        description="Free text describing specific research focus or thesis topics.",
        examples=["Deep learning architectures for tabular data generation."]
    )
    
    skills: Optional[List[str]] = Field(
        default_factory=list, 
        description="Technical tools, programming languages, or soft skills.",
        examples=[["Python", "PyTorch", "DuckDB"]]
    )
    
    # --- Optional Demographics ---
    
    age: Optional[int] = Field(
        default=None, 
        ge=16, 
        le=65, 
        description="Age of the applicant, useful for scholarships with age limits."
    )

    # --- Configuration ---
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra='ignore'
    )

    # --- Validators (Standardization Logic) ---

    @field_validator('nationality')
    @classmethod
    def standardize_nationality(cls, v: str) -> str:
        """
        Capitalizes the first letter of each word to match database strings.
        Example: 'syria' -> 'Syria'
        """
        return v.title()

    @field_validator('academic_level')
    @classmethod
    def standardize_level(cls, v: str) -> str:
        """
        Maps various user inputs to a standardized set of academic levels.
        This prevents queries from failing due to synonym mismatches.
        """
        v_lower = v.lower()
        if 'bachelor' in v_lower or 'undergrad' in v_lower or 'bsc' in v_lower:
            return 'Bachelor'
        elif 'master' in v_lower or 'postgrad' in v_lower or 'msc' in v_lower:
            return 'Master'
        elif 'phd' in v_lower or 'doctorate' in v_lower:
            return 'PhD'
        elif 'postdoc' in v_lower:
            return 'Postdoc'
        
        raise ValueError(
            f"Unrecognized academic level: '{v}'. "
            "Must map to Bachelor, Master, PhD, or Postdoc."
        )

    @field_validator('target_countries')
    @classmethod
    def standardize_target_countries(cls, v: List[str]) -> List[str]:
        """
        Applies title-casing to all countries in the target list.
        """
        return [country.title() for country in v]