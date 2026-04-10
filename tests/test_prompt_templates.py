from src.prompts.templates import build_rag_prompt


def test_build_rag_prompt_contains_expected_sections() -> None:
    prompt = build_rag_prompt(
        question="What is RAG?",
        context="RAG combines retrieval with generation.",
        fallback_answer="I don't know based on the available documents.",
    )

    assert "QUESTION" in prompt
    assert "CONTEXT" in prompt
    assert "FINAL ANSWER" in prompt
    assert "What is RAG?" in prompt


def test_build_rag_prompt_includes_fallback_rule() -> None:
    fallback = "I don't know based on the available documents."
    prompt = build_rag_prompt("Q", "C", fallback)

    assert fallback in prompt
