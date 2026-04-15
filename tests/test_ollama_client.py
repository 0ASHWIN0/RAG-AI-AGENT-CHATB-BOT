from src.llm.ollama_client import extract_text_from_response, stream_with_ollama


class Chunk:
    def __init__(self, content):
        self.content = content


class StubClient:
    def stream(self, prompt: str):
        assert prompt == "prompt"
        return [Chunk("Hello"), Chunk(" "), Chunk("world")]


def test_extract_text_from_response_handles_string_content() -> None:
    assert extract_text_from_response(Chunk("hello")) == "hello"


def test_stream_with_ollama_yields_text_chunks() -> None:
    chunks = list(stream_with_ollama(StubClient(), "prompt"))

    assert chunks == ["Hello", " ", "world"]
