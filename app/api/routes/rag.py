from fastapi import APIRouter
from app.schemas import RetrievalRequest, RetrievalHit
from app.rag.hybrid_retriever import HybridRetriever

router = APIRouter()
retriever = HybridRetriever()


@router.post("/search", response_model=list[RetrievalHit])
def search(request: RetrievalRequest):
    hits = retriever.search(request.query, request.top_k, request.metadata_filter)
    return [
        RetrievalHit(
            document_id=hit.document.document_id,
            title=hit.document.title,
            text=hit.document.text,
            metadata=hit.document.metadata,
            bm25_score=hit.bm25_score,
            vector_score=hit.vector_score,
            fused_score=hit.fused_score,
            rerank_score=hit.rerank_score,
        )
        for hit in hits
    ]
