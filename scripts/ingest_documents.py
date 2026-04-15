import re
from pathlib import Path
from typing import Any, List

import fitz
from langchain_core.documents import Document

from src.rag.chunking import sentence_chunk_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
VECTORSTORE_DIR = PROJECT_ROOT / "vectordb" / "faiss_index"


def clean_text(text: str) -> str:
    """Clean PDF text by removing HTML, header/footer noise, and extra whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"(?m)^\s*page\s*\d+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*\d+\s*of\s*\d+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\s*(confidential|draft|www\.|https?://).*?$", "", text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_pdf_documents(data_dir: Path) -> List[Document]:
    """Load PDF text from the data directory as page-level LangChain Documents."""
    documents: List[Document] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        if not pdf_path.is_file():
            continue

        with fitz.open(pdf_path) as pdf:
            for page_index in range(len(pdf)):
                raw_text = pdf[page_index].get_text()
                cleaned_text = clean_text(raw_text)
                if not cleaned_text:
                    continue
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_index + 1,
                            "path": str(pdf_path),
                        },
                    )
                )

    return documents


def sentence_aware_chunk_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Chunk documents by sentence boundaries with overlap-aware packing."""
    return sentence_chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_vector_store(chunks: List[Document]) -> Any:
    """Create and persist a FAISS vector store from chunked documents."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print(f"Loading PDFs from {DATA_DIR}")
    documents = load_pdf_documents(DATA_DIR)
    print(f"Loaded {len(documents)} page-level documents")

    if not documents:
        raise RuntimeError("No PDF text found. Check the data directory and PDF content.")

    print("Creating sentence-aware chunks...")
    chunks = sentence_aware_chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    vectorstore = create_vector_store(chunks)
    print(f"Saved FAISS vector store to: {VECTORSTORE_DIR}")
    print(f"Total vectors stored: {vectorstore.index.ntotal}")


if __name__ == "__main__":
    main()
