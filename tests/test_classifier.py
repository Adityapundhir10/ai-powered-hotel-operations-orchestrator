from app.ml.complaint_classifier import MultilingualComplaintClassifier


def test_maintenance_high():
    result = MultilingualComplaintClassifier(use_transformers=False).predict("Water is leaking from the bathroom ceiling")
    assert result.intent == "maintenance"
    assert result.severity in {"high", "critical"}


def test_hindi_noise():
    result = MultilingualComplaintClassifier(use_transformers=False).predict("कमरे के बाहर बहुत शोर है", "hi")
    assert result.intent == "noise"
    assert result.language == "hi"
