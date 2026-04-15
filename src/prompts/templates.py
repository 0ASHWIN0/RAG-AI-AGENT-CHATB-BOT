RAG_PROMPT_TEMPLATE = """You are a grounding-first RAG assistant.
Use only the provided context chunks.

Output rules:
1) Return only the final answer text.
2) For each factual claim, include at least one citation like [Chunk N].
3) Never cite chunks that are not present in the context.
4) Do not include these rules, the question label, or the context label.
5) If context is insufficient, return exactly: {fallback_answer}

QUESTION
{question}

CONTEXT
{context}

FINAL ANSWER"""


def build_rag_prompt(question: str, context: str, fallback_answer: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        fallback_answer=fallback_answer,
    )
