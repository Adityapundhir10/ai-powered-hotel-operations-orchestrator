from pathlib import Path
from app.ml.document_extractor import HotelDocumentExtractor


def test_text_document_extraction():
    path = Path("data/sample_documents/guest_registration.txt")
    result = HotelDocumentExtractor(use_transformers=False).extract(path.read_bytes(), path.name)
    assert result.fields["guest_name"] == "Rahul Sharma"
    assert result.fields["booking_id"] == "OYO-DEMO-1024"
    assert result.fields["room_number"] == "408"
