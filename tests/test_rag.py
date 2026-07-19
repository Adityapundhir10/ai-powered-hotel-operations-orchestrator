from app.rag.hybrid_retriever import HybridRetriever


def test_water_leak_retrieval():
    hits = HybridRetriever(use_transformers=False).search("urgent bathroom water leak", top_k=2)
    assert hits
    assert hits[0].document.document_id == "engineering_water_leak"
