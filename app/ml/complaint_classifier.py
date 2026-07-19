from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class ComplaintPrediction:
    intent: str
    severity: str
    confidence: float
    language: str
    backend: str


class MultilingualComplaintClassifier:
    """DistilBERT-capable multilingual classifier with a safe local fallback.

    For production, point DISTILBERT_MODEL_PATH to a fine-tuned
    distilbert-base-multilingual-cased sequence-classification checkpoint.
    """

    INTENT_KEYWORDS = {
        "maintenance": ["leak", "water", "ac", "air conditioner", "broken", "light", "power", "door", "toilet", "नल", "पानी", "बिजली"],
        "housekeeping": ["dirty", "clean", "towel", "bedsheet", "smell", "trash", "गंदा", "सफाई", "तौलिया"],
        "noise": ["noise", "loud", "music", "shouting", "noisy", "शोर"],
        "billing": ["bill", "charge", "refund", "payment", "invoice", "पैसे", "बिल"],
        "food": ["food", "breakfast", "restaurant", "cold meal", "room service", "खाना", "नाश्ता"],
        "safety": ["fire", "smoke", "threat", "unsafe", "injury", "gas", "आग", "धुआं", "खतरा"],
        "service": ["staff", "rude", "delay", "reception", "check-in", "check out", "कर्मचारी", "देरी"],
    }
    CRITICAL = ["fire", "smoke", "gas leak", "electrocution", "injury", "threat", "flood", "आग", "धुआं"]
    HIGH = ["leak", "no power", "locked", "unsafe", "broken door", "water everywhere", "बिजली", "पानी"]

    def __init__(self, model_path: str | Path | None = None, use_transformers: bool | None = None):
        self.model_path = Path(model_path or settings.resolve(settings.distilbert_model_path))
        self.use_transformers = settings.use_transformers if use_transformers is None else use_transformers
        self.pipeline: Any | None = None
        self.id2label: dict[int, str] = {}
        if self.use_transformers and self.model_path.exists():
            try:
                from transformers import pipeline
                self.pipeline = pipeline("text-classification", model=str(self.model_path), tokenizer=str(self.model_path), top_k=None)
            except Exception:
                self.pipeline = None

    @staticmethod
    def detect_language(text: str, supplied: str = "auto") -> str:
        if supplied and supplied != "auto":
            return supplied
        return "hi" if re.search(r"[\u0900-\u097F]", text) else "en"

    def _rule_prediction(self, text: str, language: str) -> ComplaintPrediction:
        normalized = text.lower()
        scores = {intent: sum(keyword in normalized for keyword in words) for intent, words in self.INTENT_KEYWORDS.items()}
        intent = max(scores, key=scores.get)
        top_score = scores[intent]
        if top_score == 0:
            intent = "service"
        if any(word in normalized for word in self.CRITICAL):
            severity = "critical"
            confidence = 0.94
        elif any(word in normalized for word in self.HIGH):
            severity = "high"
            confidence = 0.88
        elif any(word in normalized for word in ["again", "third time", "hours", "very", "बहुत"]):
            severity = "medium"
            confidence = 0.80
        else:
            severity = "low"
            confidence = 0.72 if top_score else 0.58
        return ComplaintPrediction(intent, severity, confidence, language, "rule-based fallback")

    def predict(self, text: str, language: str = "auto") -> ComplaintPrediction:
        detected = self.detect_language(text, language)
        if self.pipeline is None:
            return self._rule_prediction(text, detected)
        try:
            output = self.pipeline(text)
            candidates = output[0] if output and isinstance(output[0], list) else output
            best = max(candidates, key=lambda item: item["score"])
            label = str(best["label"]).lower()
            if "__" in label:
                intent, severity = label.split("__", 1)
            else:
                intent, severity = label, "medium"
            return ComplaintPrediction(intent, severity, float(best["score"]), detected, "DistilBERT")
        except Exception:
            return self._rule_prediction(text, detected)
