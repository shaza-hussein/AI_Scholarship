from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_generator
from src.rag.generator import ScholarshipGenerator
from src.schemas.sop import SOPGenerateRequest, SOPResponse

router = APIRouter(prefix="/api/v1", tags=["Document Generator"])

@router.post("/generate-document", response_model=SOPResponse)
async def generate_application_document(
    payload: SOPGenerateRequest,
    generator: ScholarshipGenerator = Depends(get_generator)
):
    """
    Endpoint to generate customized documents (Motivation Letter, SOP, etc.).
    Supports customization of tone and document type.
    """
    try:
        profile_dict = payload.student_profile.model_dump()
        
        generated_text = generator.generate_sop(
            profile_data=profile_dict,
            scholarship_details=payload.scholarship_details,
            additional_notes=payload.additional_notes,
            tone=payload.tone,
            document_type=payload.document_type
        )
        
        return SOPResponse(generated_text=generated_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))