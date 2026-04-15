from src.rag.pipeline import FALLBACK_ANSWER, clean_answer_text, format_context, prepare_rag_request


def test_clean_answer_text_removes_instruction_lines() -> None:
    raw = (
        "Answer ONLY from the provided context.\n"
        "If the context is insufficient, say exactly: 'I don't know based on the available documents.'\n"
        "The contract renews annually."
    )

    cleaned = clean_answer_text(raw)

    assert cleaned == "The contract renews annually."


def test_clean_answer_text_handles_empty_value() -> None:
    assert clean_answer_text("") == FALLBACK_ANSWER


class StubRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        return self.results


def test_prepare_rag_request_builds_prompt_and_sources() -> None:
    retriever = StubRetriever(
        [
            {
                "content": "RAG combines retrieval with generation.",
                "metadata": {"source": "doc.pdf", "page": 2},
                "score": 0.82,
            }
        ]
    )

    request = prepare_rag_request("What is RAG?", retriever=retriever, top_k=3, score_threshold=0.2)

    assert request["prompt"] is not None
    assert "What is RAG?" in request["prompt"]
    assert request["retrieved_count"] == 1
    assert request["sources"] == [{"source": "doc.pdf", "page": 2, "score": 0.82}]


def test_prepare_rag_request_returns_fallback_without_documents() -> None:
    retriever = StubRetriever([])

    request = prepare_rag_request("Unknown", retriever=retriever)

    assert request == {
        "answer": FALLBACK_ANSWER,
        "prompt": None,
        "sources": [],
        "retrieved_count": 0,
    }


def test_format_context_includes_grounding_scores() -> None:
    context = format_context(
        [
            {
                "content": "RAG combines retrieval and generation.",
                "metadata": {"source": "doc.pdf", "page": 4},
                "score": 0.77,
                "semantic_score": 0.7,
                "lexical_score": 0.9,
            }
        ]
    )

    assert "[Chunk 1]" in context
    assert "Grounded Score" in context
    assert "Semantic Score" in context
    assert "Lexical Score" in context
