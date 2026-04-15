import re
from typing import Any

from src.core.config import load_config
from src.llm.ollama_client import complete_with_ollama
from src.prompts.templates import build_rag_prompt
from src.rag.retriever import RagRetriever

CONFIG = load_config()
FALLBACK_ANSWER = CONFIG.fallback_answer


def build_sources(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": (item.get("metadata") or {}).get("source", "unknown"),
            "page": (item.get("metadata") or {}).get("page", "NA"),
            "score": item.get("score", "NA"),
        }
        for item in retrieved_docs
    ]


def clean_answer_text(answer_text: str) -> str:
    if not answer_text:
        return FALLBACK_ANSWER

    text = answer_text.strip()
    banned_prefixes = (
        "you are a helpful rag assistant",
        "answer only from the provided context",
        "if the context is insufficient",
        "keep the answer concise and factual",
        "question:",
        "context:",
    )
    filtered_lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith(banned_prefixes):
            continue
        filtered_lines.append(line)

    text = "\n".join(filtered_lines).strip()
    text = re.sub(r"^final answer:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^answer:\s*", "", text, flags=re.IGNORECASE).strip()

    if not text:
        return FALLBACK_ANSWER
    return text


def format_context(retrieved_docs: list[dict[str, Any]]) -> str:
    if not retrieved_docs:
        return "No context found."

    blocks = []
    for idx, item in enumerate(retrieved_docs, 1):
        metadata = item.get("metadata", {}) or {}
        text = (item.get("content", "") or "")[: CONFIG.rag_context_chars]
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append(
            f"[Chunk {idx}]\n"
            f"Source: {metadata.get('source', 'unknown')}\n"
            f"Page: {metadata.get('page', 'NA')}\n"
            f"Grounded Score: {item.get('score', 'NA')}\n"
            f"Semantic Score: {item.get('semantic_score', 'NA')}\n"
            f"Lexical Score: {item.get('lexical_score', 'NA')}\n"
            f"Content: {text}\n"
        )
    return "\n".join(blocks)


def prepare_rag_request(
    query: str,
    retriever: RagRetriever,
    top_k: int = 4,
    score_threshold: float = 0.25,
) -> dict[str, Any]:
    retrieved_docs = retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)

    if not retrieved_docs:
        return {
            "answer": FALLBACK_ANSWER,
            "prompt": None,
            "sources": [],
            "retrieved_count": 0,
        }

    context_text = format_context(retrieved_docs)
    prompt = build_rag_prompt(query, context_text, FALLBACK_ANSWER)

    return {
        "answer": None,
        "prompt": prompt,
        "sources": build_sources(retrieved_docs),
        "retrieved_count": len(retrieved_docs),
    }


def ask_rag(
    query: str,
    retriever: RagRetriever,
    mistral_client: Any,
    top_k: int = 4,
    score_threshold: float = 0.25,
) -> dict[str, Any]:
    request = prepare_rag_request(
        query=query,
        retriever=retriever,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    if request["prompt"] is None:
        return {
            "answer": request["answer"],
            "sources": request["sources"],
            "retrieved_count": request["retrieved_count"],
        }

    raw_answer = complete_with_ollama(mistral_client, request["prompt"])
    answer_text = clean_answer_text(raw_answer)

    return {
        "answer": answer_text,
        "sources": request["sources"],
        "retrieved_count": request["retrieved_count"],
    }
