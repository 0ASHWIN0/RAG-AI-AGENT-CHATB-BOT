import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    embedding_model: str
    ollama_model: str
    ollama_num_predict: int
    ollama_num_ctx: int
    rag_context_chars: int
    default_top_k: int
    default_score_threshold: float
    vectordb_dir: Path
    legacy_vectordb_dir: Path
    fallback_answer: str


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def load_config() -> AppConfig:
    vectordb_default = os.getenv("RAG_VECTORDB_DIR")
    vectordb_legacy = os.getenv("RAG_VECTORDB_LEGACY_DIR")

    return AppConfig(
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_num_predict=_env_int("OLLAMA_NUM_PREDICT", 140),
        ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", 2048),
        rag_context_chars=_env_int("RAG_CONTEXT_CHARS", 1000),
        default_top_k=_env_int("RAG_TOP_K_DEFAULT", 4),
        default_score_threshold=_env_float("RAG_SCORE_THRESHOLD_DEFAULT", 0.30),
        vectordb_dir=Path(vectordb_default) if vectordb_default else PROJECT_ROOT / "vectordb" / "faiss_index",
        legacy_vectordb_dir=Path(vectordb_legacy) if vectordb_legacy else PROJECT_ROOT / "faiss_index",
        fallback_answer=os.getenv("RAG_FALLBACK_ANSWER", "I don't know based on the available documents."),
    )
