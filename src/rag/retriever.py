import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import load_config

CONFIG = load_config()
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]{3,}\b")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall((text or "").lower()))


def _lexical_overlap(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0

    # Query-centric overlap keeps reranking stable for short factual questions.
    return len(query_tokens & content_tokens) / len(query_tokens)


class RagRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.candidate_multiplier = 3
        self.semantic_weight = 0.8

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=candidate_k)

        retrieved = []
        for doc, score in results_with_scores:
            semantic_similarity = 1 / (1 + score)
            lexical_similarity = _lexical_overlap(query, doc.page_content)
            grounded_score = (
                self.semantic_weight * semantic_similarity
                + (1 - self.semantic_weight) * lexical_similarity
            )
            if grounded_score >= score_threshold:
                retrieved.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(grounded_score, 4),
                        "semantic_score": round(semantic_similarity, 4),
                        "lexical_score": round(lexical_similarity, 4),
                    }
                )

        retrieved.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return retrieved[:top_k]


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
