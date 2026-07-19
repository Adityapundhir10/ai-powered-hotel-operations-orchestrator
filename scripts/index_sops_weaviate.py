import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.weaviate_store import WeaviateSOPStore


def main():
    retriever = HybridRetriever(use_transformers=True)
    if retriever.embedding_model is None:
        raise RuntimeError("Install requirements-full.txt and enable transformer models.")
    vectors = retriever.embedding_model.encode(
        [doc.title + " " + doc.text for doc in retriever.documents], normalize_embeddings=True
    ).tolist()
    store = WeaviateSOPStore()
    store.connect()
    try:
        store.ensure_collection()
        count = store.upsert(retriever.documents, vectors)
        print(f"Indexed {count} SOPs into Weaviate")
    finally:
        store.close()


if __name__ == "__main__":
    main()
