from typing import Any

from langchain_ollama import ChatOllama

from src.core.config import load_config

CONFIG = load_config()
OLLAMA_MODEL = CONFIG.ollama_model


def load_ollama_client() -> ChatOllama:
    try:
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            num_predict=CONFIG.ollama_num_predict,
            num_ctx=CONFIG.ollama_num_ctx,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Ollama client. Make sure Ollama is installed and running."
        ) from exc


def extract_text_from_response(res: Any) -> str:
    content = getattr(res, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts) if parts else str(content)
    return str(content if content is not None else res)


def complete_with_ollama(ollama_client: ChatOllama, prompt: str) -> str:
    res = ollama_client.invoke(prompt)
    return extract_text_from_response(res)
