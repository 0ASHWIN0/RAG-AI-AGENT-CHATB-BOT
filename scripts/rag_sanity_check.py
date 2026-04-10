import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from src.core.config import load_config
    from src.llm.ollama_client import load_ollama_client
    from src.rag.pipeline import ask_rag
    from src.rag.retriever import load_retriever

    config = load_config()
    query = "What is this document about?"

    print("[sanity] Loading retriever...")
    try:
        retriever = load_retriever()
    except Exception as exc:
        print(f"[sanity] FAIL: retriever load error: {exc}")
        return 1

    print("[sanity] Loading local model client...")
    try:
        llm_client = load_ollama_client()
    except Exception as exc:
        print(f"[sanity] FAIL: model client load error: {exc}")
        return 2

    print("[sanity] Running RAG query...")
    try:
        result = ask_rag(
            query=query,
            retriever=retriever,
            mistral_client=llm_client,
            top_k=config.default_top_k,
            score_threshold=config.default_score_threshold,
        )
    except Exception as exc:
        print(f"[sanity] FAIL: ask_rag error: {exc}")
        return 3

    answer = (result.get("answer") or "").strip()
    retrieved_count = int(result.get("retrieved_count") or 0)

    if not answer:
        print("[sanity] FAIL: empty answer")
        return 4

    leaked_markers = (
        "answer only from the provided context",
        "if the context is insufficient",
        "question",
        "context",
    )
    lowered = answer.lower()
    if any(marker in lowered for marker in leaked_markers):
        print("[sanity] FAIL: prompt instructions leaked into answer")
        print(f"[sanity] Answer: {answer}")
        return 5

    print("[sanity] PASS")
    print(f"[sanity] Retrieved chunks: {retrieved_count}")
    print(f"[sanity] Fallback answer used: {answer == config.fallback_answer}")
    print(f"[sanity] Answer preview: {answer[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
