from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io

from PIL import Image


@dataclass(frozen=True)
class LayoutToken:
    text: str
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class LayoutLMv3Adapter:
    """Inference adapter for a fine-tuned LayoutLMv3 token classifier.

    The adapter performs OCR word/box extraction with Tesseract, normalizes
    bounding boxes to LayoutLM's 0..1000 coordinate space, and runs a local
    fine-tuned LayoutLMv3 checkpoint. Base weights are not bundled.
    """

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        self.processor: Any | None = None
        self.model: Any | None = None
        if self.checkpoint.exists():
            try:
                from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
                self.processor = AutoProcessor.from_pretrained(str(self.checkpoint), apply_ocr=False)
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(self.checkpoint))
                self.model.eval()
            except Exception:
                self.processor = None
                self.model = None

    @property
    def available(self) -> bool:
        return self.processor is not None and self.model is not None

    @staticmethod
    def _normalize_box(box: tuple[int, int, int, int], width: int, height: int) -> list[int]:
        x0, y0, x1, y1 = box
        return [
            int(1000 * x0 / max(width, 1)),
            int(1000 * y0 / max(height, 1)),
            int(1000 * x1 / max(width, 1)),
            int(1000 * y1 / max(height, 1)),
        ]

    def predict(self, payload: bytes) -> list[LayoutToken]:
        if not self.available:
            return []
        import pytesseract
        import torch
        from pytesseract import Output

        image = Image.open(io.BytesIO(payload)).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        words, boxes = [], []
        for i, text in enumerate(data["text"]):
            clean = text.strip()
            if not clean:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(clean)
            boxes.append(self._normalize_box((x, y, x + w, y + h), image.width, image.height))
        if not words:
            return []
        encoded = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
        probabilities = logits.softmax(-1)[0]
        predicted = probabilities.argmax(-1)
        word_ids = encoded.word_ids(batch_index=0)
        output: list[LayoutToken] = []
        seen = set()
        for token_index, word_index in enumerate(word_ids):
            if word_index is None or word_index in seen or word_index >= len(words):
                continue
            seen.add(word_index)
            label_id = int(predicted[token_index])
            label = self.model.config.id2label.get(label_id, str(label_id))
            confidence = float(probabilities[token_index, label_id])
            output.append(LayoutToken(words[word_index], label, confidence, tuple(boxes[word_index])))
        return output
