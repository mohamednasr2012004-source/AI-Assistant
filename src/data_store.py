import shutil
from pathlib import Path

import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    SPARSE_TOP_K,
)
from utils.metadata import METADATA_FIELDS, to_primitive


class StoreBuilder:
    def __init__(
        self,
        data: pd.DataFrame,
        persist_dir: str = PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.data = data
        self.persist_dir = Path(persist_dir)
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.sparse_retriever = None

    def _build_documents(self) -> list[Document]:
        documents = []

        for _, row in self.data.iterrows():
            metadata = {}
            for field in METADATA_FIELDS:
                if field not in row:
                    continue
                value = row[field]
                if pd.isna(value):
                    continue
                metadata[field] = to_primitive(value)

            documents.append(Document(page_content=str(row["text"]), metadata=metadata))

        return documents

    def _split_documents(self) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(self._build_documents())

    def build_and_save_vectorstore(self, overwrite: bool = False):
        if self.persist_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Vector store already exists at {self.persist_dir}. "
                    "Delete it or rebuild with overwrite=True."
                )
            shutil.rmtree(self.persist_dir)

        documents = self._split_documents()
        if not documents:
            raise ValueError("No documents to index")

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def load_vector_store(self) -> Chroma:
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_dir}. Run pipeline/build_pipeline.py first."
            )

        return Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding,
        )

    def build_sparse_retriever(self) -> BM25Retriever:
        documents = self._split_documents()
        if not documents:
            raise ValueError("No documents to index")

        self.sparse_retriever = BM25Retriever.from_documents(documents)
        self.sparse_retriever.k = SPARSE_TOP_K
        return self.sparse_retriever
