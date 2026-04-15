from src.rag.retriever import RagRetriever


class FakeDoc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    def similarity_search_with_score(self, query: str, k: int):
        # First item has better vector score but weak lexical overlap.
        # Second item has slightly worse vector score but strong lexical overlap.
        return [
            (FakeDoc("General overview of systems.", {"source": "a.pdf", "page": 1}), 0.2),
            (FakeDoc("RAG retrieval and generation pipeline details.", {"source": "b.pdf", "page": 2}), 0.4),
        ]


def test_retrieve_applies_grounded_reranking_and_scores() -> None:
    retriever = RagRetriever(vector_store=FakeVectorStore())
    results = retriever.retrieve("What is RAG retrieval pipeline?", top_k=2, score_threshold=0.0)

    assert len(results) == 2
    assert "semantic_score" in results[0]
    assert "lexical_score" in results[0]
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_respects_score_threshold() -> None:
    retriever = RagRetriever(vector_store=FakeVectorStore())
    results = retriever.retrieve("unrelated query", top_k=2, score_threshold=0.95)

    assert results == []
