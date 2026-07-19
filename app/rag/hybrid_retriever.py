from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings


@dataclass(frozen=True)
class SOPDocument:
    document_id: str
    title: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class SearchHit:
    document: SOPDocument
    bm25_score: float
    vector_score: float
    fused_score: float
    rerank_score: float | None = None


class SimpleBM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.df: dict[str, int] = {}
        for doc in documents:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
        self.n = len(documents)

    def score(self, query_tokens: list[str], index: int) -> float:
        doc = self.documents[index]
        frequencies: dict[str, int] = {}
        for token in doc:
            frequencies[token] = frequencies.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            df = self.df.get(token, 0)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            tf = frequencies.get(token, 0)
            denominator = tf + self.k1 * (1 - self.b + self.b * len(doc) / max(self.avgdl, 1))
            score += idf * (tf * (self.k1 + 1) / denominator) if denominator else 0
        return score


class HybridRetriever:
    """BM25 + vector fusion with optional MiniLM and cross-encoder reranking."""

    def __init__(self, documents: list[SOPDocument] | None = None, use_transformers: bool | None = None):
        self.use_transformers = settings.use_transformers if use_transformers is None else use_transformers
        self.documents = documents or self._load_sops(settings.resolve(settings.sop_directory))
        self.tokenized = [self._tokenize(doc.title + " " + doc.text) for doc in self.documents]
        self.bm25 = SimpleBM25(self.tokenized)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        corpus = [doc.title + " " + doc.text for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus) if corpus else None
        self.embedding_model: Any | None = None
        self.cross_encoder: Any | None = None
        self.document_embeddings: np.ndarray | None = None
        if self.use_transformers and corpus:
            try:
                from sentence_transformers import SentenceTransformer, CrossEncoder
                self.embedding_model = SentenceTransformer(settings.minilm_model_name)
                self.cross_encoder = CrossEncoder(settings.cross_encoder_model_name)
                self.document_embeddings = self.embedding_model.encode(corpus, normalize_embeddings=True)
            except Exception:
                self.embedding_model = None
                self.cross_encoder = None
                self.document_embeddings = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _load_sops(directory: Path) -> list[SOPDocument]:
        documents: list[SOPDocument] = []
        if not directory.exists():
            return documents
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            title = raw.splitlines()[0].lstrip("# ") if raw.splitlines() else path.stem
            metadata = {"team": path.stem.split("_")[0], "source": "synthetic_sop", "filename": path.name}
            documents.append(SOPDocument(path.stem, title, raw, metadata))
        return documents

    def _vector_scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.array([])
        if self.embedding_model is not None and self.document_embeddings is not None:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            return np.asarray(query_embedding @ self.document_embeddings.T)[0]
        query_vector = self.vectorizer.transform([query])
        return cosine_similarity(query_vector, self.tfidf_matrix)[0]

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        minimum, maximum = float(values.min()), float(values.max())
        if maximum - minimum < 1e-12:
            return np.zeros_like(values) if maximum == 0 else np.ones_like(values)
        return (values - minimum) / (maximum - minimum)

    def search(self, query: str, top_k: int = 3, metadata_filter: dict[str, str] | None = None) -> list[SearchHit]:
        if not self.documents:
            return []
        query_tokens = self._tokenize(query)
        raw_bm25 = np.array([self.bm25.score(query_tokens, i) for i in range(len(self.documents))], dtype=float)
        raw_vector = self._vector_scores(query)
        bm25 = self._normalize(raw_bm25)
        vector = self._normalize(raw_vector)
        fused = 0.45 * bm25 + 0.55 * vector
        candidates = list(range(len(self.documents)))
        filters = metadata_filter or {}
        candidates = [
            i for i in candidates
            if all(self.documents[i].metadata.get(k) == v for k, v in filters.items())
        ]
        candidates.sort(key=lambda i: float(fused[i]), reverse=True)
        candidates = candidates[: max(top_k * 3, top_k)]
        rerank_scores: dict[int, float] = {}
        if self.cross_encoder is not None and candidates:
            pairs = [(query, self.documents[i].title + " " + self.documents[i].text) for i in candidates]
            predictions = self.cross_encoder.predict(pairs)
            rerank_scores = {i: float(score) for i, score in zip(candidates, predictions)}
            candidates.sort(key=lambda i: rerank_scores[i], reverse=True)
        return [
            SearchHit(
                document=self.documents[i],
                bm25_score=round(float(raw_bm25[i]), 6),
                vector_score=round(float(raw_vector[i]), 6),
                fused_score=round(float(fused[i]), 6),
                rerank_score=round(rerank_scores[i], 6) if i in rerank_scores else None,
            )
            for i in candidates[:top_k]
        ]
