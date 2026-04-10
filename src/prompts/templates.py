RAG_PROMPT_TEMPLATE = """Task: Answer the question using only the context.
Output rules:
1) Return only the final answer text.
2) Do not include these rules, the question label, or the context label.
3) If context is insufficient, return exactly: {fallback_answer}

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
