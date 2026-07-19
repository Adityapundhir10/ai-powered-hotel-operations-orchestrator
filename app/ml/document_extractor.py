from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io
import re

from PIL import Image
from app.config import settings
from app.ml.layoutlm_adapter import LayoutLMv3Adapter


@dataclass
class ExtractionResult:
    document_type: str
    fields: dict[str, Any]
    field_confidences: dict[str, float]
    backend: str
    warnings: list[str]

    @property
    def average_confidence(self) -> float:
        values = list(self.field_confidences.values())
        return round(sum(values) / len(values), 4) if values else 0.0


class HotelDocumentExtractor:
    """OCR + LayoutLMv3-capable document extractor.

    LayoutLMv3 requires task-specific fine-tuning for reliable field extraction.
    This project exposes the production integration and includes a deterministic
    regex fallback for sample registration forms and invoices.
    """

    def __init__(self, use_transformers: bool | None = None):
        self.use_transformers = settings.use_transformers if use_transformers is None else use_transformers
        self.layout_adapter = LayoutLMv3Adapter(settings.resolve(settings.layoutlm_model_path))

    def _ocr(self, payload: bytes, filename: str) -> tuple[str, str, list[str]]:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".csv", ".json"}:
            return payload.decode("utf-8", errors="replace"), "text", []
        try:
            import pytesseract
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            text = pytesseract.image_to_string(image, lang="eng")
            return text, "Tesseract OCR", []
        except Exception as exc:
            return "", "OCR unavailable", [f"OCR could not run: {type(exc).__name__}"]

    @staticmethod
    def _regex_fields(text: str) -> tuple[str, dict[str, Any], dict[str, float]]:
        patterns = {
            "guest_name": r"(?:guest\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{2,60})",
            "booking_id": r"(?:booking|reservation)\s*(?:id|no\.?|number)?\s*[:\-]\s*([A-Z0-9-]{4,30})",
            "room_number": r"room\s*(?:no\.?|number)?\s*[:\-]\s*([A-Z0-9-]{1,10})",
            "check_in": r"check[ -]?in\s*[:\-]\s*([0-9]{2,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})",
            "check_out": r"check[ -]?out\s*[:\-]\s*([0-9]{2,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})",
            "amount": r"(?:total|amount)\s*[:\-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.\d{1,2})?)",
            "phone": r"(?:phone|mobile)\s*[:\-]\s*(\+?[0-9][0-9 -]{7,16})",
            "email": r"(?:email)\s*[:\-]\s*([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})",
        }
        fields, confidences = {}, {}
        for name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[name] = match.group(1).strip()
                confidences[name] = 0.90
        doc_type = "hotel_invoice" if "amount" in fields else "guest_registration_form"
        return doc_type, fields, confidences

    def extract(self, payload: bytes, filename: str) -> ExtractionResult:
        text, ocr_backend, warnings = self._ocr(payload, filename)
        doc_type, fields, confidences = self._regex_fields(text)
        backend = f"{ocr_backend} + regex"
        if self.use_transformers and self.layout_adapter.available and Path(filename).suffix.lower() not in {".txt", ".csv", ".json"}:
            layout_tokens = self.layout_adapter.predict(payload)
            for token in layout_tokens:
                if token.label.startswith("B-") and token.label[2:].lower() not in fields:
                    key = token.label[2:].lower()
                    fields[key] = token.text
                    confidences[key] = round(token.confidence, 4)
            backend += " + LayoutLMv3"
        elif self.use_transformers:
            warnings.append("LayoutLMv3 requires a local fine-tuned token-classification checkpoint.")
            backend += " + LayoutLMv3 interface"
        if not fields:
            warnings.append("No supported fields were detected.")
        return ExtractionResult(doc_type, fields, confidences, backend, warnings)
