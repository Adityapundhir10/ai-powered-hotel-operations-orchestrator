from fastapi import APIRouter, File, UploadFile
from app.schemas import DocumentExtractionResponse
from app.ml.document_extractor import HotelDocumentExtractor

router = APIRouter()
extractor = HotelDocumentExtractor()


@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    payload = await file.read()
    result = extractor.extract(payload, file.filename or "document.bin")
    return DocumentExtractionResponse(
        document_type=result.document_type,
        fields=result.fields,
        field_confidences=result.field_confidences,
        average_confidence=result.average_confidence,
        backend=result.backend,
        warnings=result.warnings,
    )
