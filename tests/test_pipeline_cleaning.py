from src.rag.pipeline import FALLBACK_ANSWER, clean_answer_text


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
