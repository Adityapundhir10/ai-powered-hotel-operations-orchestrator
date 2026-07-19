from __future__ import annotations
from typing import Any
from app.config import settings
from app.rag.hybrid_retriever import SOPDocument


class WeaviateSOPStore:
    """Optional Weaviate v4 adapter using self-provided vectors."""

    COLLECTION = "HotelSOP"

    def __init__(self):
        self.client: Any | None = None

    def connect(self):
        if not settings.use_weaviate:
            return None
        import weaviate
        from weaviate.connect import ConnectionParams
        self.client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=settings.weaviate_http_host,
                http_port=settings.weaviate_http_port,
                http_secure=False,
                grpc_host=settings.weaviate_http_host,
                grpc_port=settings.weaviate_grpc_port,
                grpc_secure=False,
            )
        )
        self.client.connect()
        return self.client

    def ensure_collection(self):
        if self.client is None:
            return
        from weaviate.classes.config import Configure, Property, DataType
        if not self.client.collections.exists(self.COLLECTION):
            self.client.collections.create(
                name=self.COLLECTION,
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="team", data_type=DataType.TEXT),
                ],
            )

    def upsert(self, documents: list[SOPDocument], vectors: list[list[float]]):
        if self.client is None:
            return 0
        collection = self.client.collections.use(self.COLLECTION)
        with collection.batch.dynamic() as batch:
            for document, vector in zip(documents, vectors):
                batch.add_object(
                    properties={
                        "document_id": document.document_id,
                        "title": document.title,
                        "text": document.text,
                        "team": document.metadata.get("team", "operations"),
                    },
                    vector=vector,
                )
        return len(documents)

    def hybrid_search(self, query: str, top_k: int = 3):
        if self.client is None:
            return []
        collection = self.client.collections.use(self.COLLECTION)
        response = collection.query.hybrid(query=query, alpha=0.55, limit=top_k)
        return [obj.properties for obj in response.objects]

    def close(self):
        if self.client is not None:
            self.client.close()
