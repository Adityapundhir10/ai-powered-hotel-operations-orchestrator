import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pathlib import Path
from datetime import date
import json
import pandas as pd

from app.ml.occupancy_forecaster import OccupancyForecaster
from app.ml.complaint_classifier import MultilingualComplaintClassifier

ROOT = Path(__file__).resolve().parents[1]


def evaluate_classifier():
    samples = [
        ("Water is leaking from the bathroom ceiling", "maintenance", "high"),
        ("The room is dirty and towels are missing", "housekeeping", "low"),
        ("There is smoke near the electrical panel", "safety", "critical"),
        ("My bill has an incorrect extra charge", "billing", "low"),
        ("बहुत शोर है और मैं सो नहीं पा रहा", "noise", "medium"),
    ]
    classifier = MultilingualComplaintClassifier(use_transformers=False)
    predictions = [classifier.predict(text) for text, _, _ in samples]
    intent_accuracy = sum(p.intent == expected for p, (_, expected, _) in zip(predictions, samples)) / len(samples)
    severity_accuracy = sum(p.severity == expected for p, (_, _, expected) in zip(predictions, samples)) / len(samples)
    return {"intent_accuracy": intent_accuracy, "severity_accuracy": severity_accuracy, "samples": len(samples), "backend": predictions[0].backend}


def main():
    results = {"classifier": evaluate_classifier()}
    model_path = ROOT / "models" / "occupancy_xgb.joblib"
    if model_path.exists():
        results["forecast_model"] = {"loaded": True, "backend": OccupancyForecaster(model_path).backend}
    else:
        results["forecast_model"] = {"loaded": False, "instruction": "Run python scripts/train_forecaster.py"}
    output = ROOT / "artifacts" / "evaluations" / "model_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
