from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import load_config

CONFIG = load_config()


class RagRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

        retrieved = []
        for doc, score in results_with_scores:
            similarity = 1 / (1 + score)
            if similarity >= score_threshold:
                retrieved.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(similarity, 4),
                    }
                )
        return retrieved


def get_embeddings() -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=CONFIG.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embeddings. Install dependencies with: "
            "uv pip install -r requirements.txt"
        ) from exc


def resolve_vectordb_dir() -> Path:
    if CONFIG.vectordb_dir.exists():
        return CONFIG.vectordb_dir
    if CONFIG.legacy_vectordb_dir.exists():
        return CONFIG.legacy_vectordb_dir
    return CONFIG.vectordb_dir


def load_retriever() -> RagRetriever:
    vectordb_dir = resolve_vectordb_dir()
    if not vectordb_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {vectordb_dir}. Run the notebook embedding cell to create it."
        )

    vectorstore = FAISS.load_local(
        str(vectordb_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return RagRetriever(vector_store=vectorstore)
