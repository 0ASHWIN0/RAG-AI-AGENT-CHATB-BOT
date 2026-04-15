from langchain_core.documents import Document

from src.rag.chunking import sentence_chunk_documents, sentence_chunks_from_text, split_into_sentences


def test_split_into_sentences_uses_sentence_boundaries() -> None:
    text = "First sentence. Second sentence! Third one?"

    sentences = split_into_sentences(text)

    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_sentence_chunks_from_text_keeps_sentence_units() -> None:
    text = "A short sentence. Another short sentence. Final short sentence."

    chunks = sentence_chunks_from_text(text, chunk_size=40, chunk_overlap=10)

    assert chunks
    assert all(chunk.endswith((".", "!", "?")) for chunk in chunks)


def test_sentence_chunk_documents_adds_sentence_chunking_metadata() -> None:
    docs = [
        Document(
            page_content="One. Two. Three. Four.",
            metadata={"source": "doc.pdf", "page": 1},
        )
    ]

    chunks = sentence_chunk_documents(docs, chunk_size=12, chunk_overlap=4)

    assert chunks
    assert all(chunk.metadata.get("chunking") == "sentence" for chunk in chunks)
