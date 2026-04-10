import streamlit as st

from src.core.config import load_config
from src.llm.ollama_client import load_ollama_client
from src.rag.pipeline import ask_rag
from src.rag.retriever import load_retriever

CONFIG = load_config()


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="R", layout="wide")
    st.title("RAG AI Chatbot")
    st.caption("Ask questions from your PDF knowledge base")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top K retrieval", min_value=1, max_value=8, value=CONFIG.default_top_k)
        score_threshold = st.slider(
            "Score threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG.default_score_threshold,
            step=0.05,
        )
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    try:
        retriever = load_retriever()
        llm_client = load_ollama_client()
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.info(
            "Run notebook embedding first, then ensure Ollama is running (ollama serve) "
            f"and the model exists (ollama pull {CONFIG.ollama_model})."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.write(f"{idx}. {src['source']} (page={src['page']}, score={src['score']})")

    user_query = st.chat_input("Ask a question about your PDFs...")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_rag(
                    query=user_query,
                    retriever=retriever,
                    mistral_client=llm_client,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                st.markdown(result["answer"])
                with st.expander("Sources"):
                    for idx, src in enumerate(result["sources"], 1):
                        st.write(f"{idx}. {src['source']} (page={src['page']}, score={src['score']})")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
