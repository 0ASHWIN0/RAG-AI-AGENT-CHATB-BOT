import streamlit as st

from src.core.config import load_config
from src.llm.ollama_client import load_ollama_client, stream_with_ollama
from src.rag.pipeline import clean_answer_text, prepare_rag_request
from src.rag.retriever import load_retriever

CONFIG = load_config()


def render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for idx, src in enumerate(sources, 1):
            st.write(f"{idx}. {src['source']} (page={src['page']}, score={src['score']})")


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))


def get_vectorstore_stats(retriever: object) -> dict[str, str]:
    stats = {
        "index_chunks": "NA",
        "docstore_items": "NA",
        "vectordb_dir": str(CONFIG.vectordb_dir),
    }

    vector_store = getattr(retriever, "vector_store", None)
    if vector_store is None:
        return stats

    index = getattr(vector_store, "index", None)
    ntotal = getattr(index, "ntotal", None)
    if ntotal is not None:
        stats["index_chunks"] = str(ntotal)

    docstore = getattr(vector_store, "docstore", None)
    store_map = getattr(docstore, "_dict", None)
    if isinstance(store_map, dict):
        stats["docstore_items"] = str(len(store_map))

    return stats


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

    stats = get_vectorstore_stats(retriever)
    with st.sidebar:
        st.divider()
        st.subheader("Runtime")
        st.caption(f"Model: {CONFIG.ollama_model}")
        st.caption(f"Index chunks: {stats['index_chunks']}")
        st.caption(f"Docstore items: {stats['docstore_items']}")
        st.caption(f"Vector DB: {stats['vectordb_dir']}")

    ensure_session_state()
    render_chat_history()

    user_query = st.chat_input("Ask a question about your PDFs...")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            request = prepare_rag_request(
                query=user_query,
                retriever=retriever,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            if request["prompt"] is None:
                answer_text = request["answer"]
                st.markdown(answer_text)
            else:
                response_placeholder = st.empty()
                stream_status = st.status("Streaming response...", expanded=False)
                raw_answer = ""
                for chunk in stream_with_ollama(llm_client, request["prompt"]):
                    raw_answer += chunk
                    response_placeholder.markdown(f"{raw_answer}▌")

                stream_status.update(label="Response complete", state="complete")

                answer_text = clean_answer_text(raw_answer)
                response_placeholder.markdown(answer_text)

            render_sources(request["sources"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": request["sources"],
                }
            )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
