from fastapi import APIRouter, HTTPException
import duckdb
import urllib.parse
from src.api.schemas import ScholarshipDetailResponse
import pandas as pd

# تأكد من وضع مسار قاعدة بيانات DuckDB الصحيح الخاص بك
DUCKDB_PATH = "db/analytical.duckdb" 

router = APIRouter(prefix="/api/v1", tags=["Scholarships Deatils"])

@router.get("/scholarships/{scholarship_name}", response_model=ScholarshipDetailResponse)
def get_scholarship_details(scholarship_name: str):
    """
    Endpoint to fetch full scholarship details from DuckDB using its name.
    """
    # فك تشفير الاسم في حال كان يحتوي على مسافات أو رموز من الرابط
    decoded_name = urllib.parse.unquote(scholarship_name)
    
    try:
        # الاتصال بقاعدة البيانات
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
        
        # استخدام الاستعلام الآمن (Parameterized Query) لمنع SQL Injection
        query = """
            SELECT 
                scholarship_name, host_country, academic_level, academic_major,
                funding_category, funding_amount, standardized_deadline,
                description, scholarship_details, eligibility, 
                application_process, application_link
            FROM scholarships 
            WHERE scholarship_name = ?
        """
        
       
        result = conn.execute(query, [decoded_name]).fetchdf()
        conn.close()
        
        if result.empty:
            raise HTTPException(status_code=404, detail="Scholarship not found")
            
        
        row = result.iloc[0].to_dict()
        
        
        for key, value in row.items():
            if pd.isna(value):
                row[key] = "Not Specified" if key == "funding_amount" else ""
                
        return ScholarshipDetailResponse(**row)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))