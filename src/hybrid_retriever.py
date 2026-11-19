from typing import Any, Callable

from langchain_core.documents import Document

from utils.logger import get_logger

logger = get_logger(__name__)


class FilteredHybridRetriever:
    def __init__(
        self,
        vectorstore,
        sparse_retriever,
        weights: list[float] | None = None,
        dense_k: int = 5,
        sparse_k: int = 5,
        sparse_candidates: int = 50,
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.vectorstore = vectorstore
        self.sparse_retriever = sparse_retriever
        self.weights = weights or [0.6, 0.4]
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.sparse_candidates = max(sparse_candidates, sparse_k)
        self.top_k = top_k
        self.rrf_k = rrf_k

        self.sparse_retriever.k = self.sparse_candidates

    def invoke(
        self,
        query: str,
        where: dict[str, Any] | None = None,
        predicate: Callable[[dict], bool] | None = None,
    ) -> list[Document]:
        dense_documents = self._dense_search(query, where)
        sparse_documents = self._sparse_search(query, predicate)

        if where is not None and not dense_documents and not sparse_documents:
            logger.warning("Metadata filter returned no documents, retrying without it")
            return self.invoke(query)

        return self._fuse(dense_documents, sparse_documents)

    def _dense_search(
        self, query: str, where: dict[str, Any] | None
    ) -> list[Document]:
        try:
            return self.vectorstore.similarity_search(
                query, k=self.dense_k, filter=where
            )
        except Exception as error:
            logger.warning("Dense search with metadata filter failed: %s", error)
            return []

    def _sparse_search(
        self, query: str, predicate: Callable[[dict], bool] | None
    ) -> list[Document]:
        documents = self.sparse_retriever.invoke(query)
        if predicate is not None:
            documents = [
                document for document in documents if predicate(document.metadata)
            ]
        return documents[: self.sparse_k]

    def _fuse(
        self, dense_documents: list[Document], sparse_documents: list[Document]
    ) -> list[Document]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        for weight, results in zip(self.weights, [dense_documents, sparse_documents]):
            for rank, document in enumerate(results):
                key = document.metadata.get("id") or document.page_content
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank)
                documents.setdefault(key, document)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [documents[key] for key in ranked[: self.top_k]]
