# RAG AI Agent Chat Bot

A Retrieval-Augmented Generation (RAG) AI chatbot built with Python, LangChain, FAISS, Ollama, and Streamlit. This project ingests PDF documents, processes them into chunks, stores embeddings in a vector database, and serves a chat interface grounded on your PDF knowledge base.

## Features

- **Document Ingestion**: Load and parse PDF files from a specified directory using PyMuPDF
- **Text Chunking**: Split documents into manageable chunks with configurable size and overlap
- **Vector Search**: FAISS-backed retrieval pipeline for similarity search over chunked documents
- **Local Generation (Ollama)**: Answer generation using a local Ollama model
- **Streamlit Chat UI**: Browser-based chatbot interface with source display
- **Modular Design**: Clean separation of retriever, generator, pipeline, notebooks, and app layer
- **Virtual Environment**: Isolated Python environment with uv package management

## Installation

### Prerequisites
- Python 3.12+
- uv (Python package manager)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-ai-anget-chat-bot
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or source .venv/Scripts/activate in Git Bash
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

### Data Processing
1. Place your PDF documents in the `data` directory (create it if it doesn't exist)
2. Open `notebooks/document.ipynb` in VS Code
3. Run the cells in order:
   - Load PDF documents from the data directory
   - Create chunks from the loaded documents
   - Build and save the FAISS vector database

The notebook saves vector data for the application to load later.

### Running the Application
```bash
python main.py
```

### Run Streamlit Chatbot
```bash
python -m streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

### Requirements Before Running the Chatbot
1. Start Ollama locally:
   ```bash
   ollama serve
   ```
2. Pull the model once:
   ```bash
   ollama pull llama3.1:8b
   ```
3. The FAISS index must already exist. Generate it first from the notebook.

### Quick Sanity Check
Run this after setup to validate retrieval, generation, and output guardrails:

```bash
python scripts/rag_sanity_check.py
```

### Runtime Configuration (Environment Variables)
Optional values you can set before launching the app:

```env
OLLAMA_MODEL=llama3.1:8b
OLLAMA_NUM_PREDICT=140
OLLAMA_NUM_CTX=2048
RAG_TOP_K_DEFAULT=4
RAG_SCORE_THRESHOLD_DEFAULT=0.30
RAG_CONTEXT_CHARS=1000
RAG_VECTORDB_DIR=./vectordb/faiss_index
RAG_VECTORDB_LEGACY_DIR=./faiss_index
```

## Project Structure

```
rag-ai-anget-chat-bot/
├── app.py                  # Streamlit chatbot entry point
├── main.py                 # Optional Python entry point
├── pyproject.toml          # Project configuration
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Input PDF documents
├── chunks/                 # Processed chunk outputs
├── vectordb/               # Saved vector database
├── notebooks/
│   └── document.ipynb      # Preprocessing, tuning, and evaluation notebook
├── src/
│   ├── core/
│   │   └── config.py       # Centralized runtime configuration
│   ├── llm/
│   │   └── ollama_client.py # Local Ollama client integration
│   ├── prompts/
│   │   └── templates.py    # Prompt templates and guardrails
│   ├── rag/
│   │   ├── pipeline.py     # End-to-end RAG pipeline
│   │   └── retriever.py    # Retrieval logic and FAISS loading
│   └── ui/
│       └── streamlit_app.py # Streamlit UI implementation
├── scripts/
│   └── rag_sanity_check.py # End-to-end validation script
├── tests/
│   └── ...                 # Unit tests for config, prompts, and pipeline
└── .venv/                  # Virtual environment (created by uv)
```

## Dependencies

- `langchain`: Core framework for LLM applications
- `langchain-community`: Community integrations
- `faiss-cpu`: Local vector database for semantic retrieval
- `langchain-ollama`: Ollama LangChain integration
- `streamlit`: Chatbot web UI
- `python-dotenv`: Environment variable loading
- `pypdf`: PDF processing
- `pymupdf`: Alternative PDF loader
- `uv`: Fast Python package manager

## Development

### Developer Setup
Install dev tooling for tests/linting/type checks:

```bash
uv pip install -e ".[dev]"
```

### Quality Gates
Run these before merging changes:

```bash
ruff check .
ruff format .
pytest
python scripts/rag_sanity_check.py
```

### Continuous Integration
Pull requests and pushes to the main branch run automated Ruff and pytest checks via [CI workflow](.github/workflows/ci.yml).

### Pre-commit Hooks
Enable automated checks on each commit:

```bash
uv pip install pre-commit
pre-commit install
```

### Adding New Features
- Extend the notebook for preprocessing and evaluation
- Implement retrieval or generation changes inside `src/`
- Update the UI in `app.py`
- Add more document loaders as needed

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Keep functions modular and well-documented

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Future Enhancements

- Streaming responses in the Streamlit UI
- Better source citations and answer formatting
- Multi-format document support
- Advanced chunking and reranking strategies
- Evaluation suite for retrieval and answer quality