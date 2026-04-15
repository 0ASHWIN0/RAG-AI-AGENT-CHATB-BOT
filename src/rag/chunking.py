import re
from typing import Any

from langchain_core.documents import Document

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []

    sentences = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(normalized) if segment.strip()]
    return sentences or [normalized]


def _tail_for_overlap(sentences: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or not sentences:
        return []

    kept: list[str] = []
    current_size = 0
    for sentence in reversed(sentences):
        candidate_size = len(sentence) + (1 if kept else 0)
        if kept and current_size + candidate_size > overlap_chars:
            break
        kept.append(sentence)
        current_size += candidate_size

    kept.reverse()
    return kept


def sentence_chunks_from_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if not current_sentences:
            current_sentences.append(sentence)
            current_len = sentence_len
            continue

        projected_len = current_len + 1 + sentence_len
        if projected_len <= chunk_size:
            current_sentences.append(sentence)
            current_len = projected_len
            continue

        chunks.append(" ".join(current_sentences))
        overlap_seed = _tail_for_overlap(current_sentences, chunk_overlap)
        current_sentences = overlap_seed.copy()
        current_len = len(" ".join(current_sentences)) if current_sentences else 0

        if current_len and current_len + 1 + sentence_len <= chunk_size:
            current_sentences.append(sentence)
            current_len = current_len + 1 + sentence_len
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_len = sentence_len

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def sentence_chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    chunked_documents: list[Document] = []

    for document in documents:
        source_text = document.page_content or ""
        chunks = sentence_chunks_from_text(source_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, chunk_text in enumerate(chunks, start=1):
            metadata: dict[str, Any] = dict(document.metadata or {})
            metadata["chunk_index"] = idx
            metadata["chunking"] = "sentence"
            chunked_documents.append(Document(page_content=chunk_text, metadata=metadata))

    return chunked_documents
