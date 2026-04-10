from src.core.config import load_config


def test_load_config_defaults() -> None:
    config = load_config()

    assert config.ollama_model
    assert config.default_top_k >= 1
    assert 0.0 <= config.default_score_threshold <= 1.0
    assert config.rag_context_chars > 0


def test_load_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "tinyllama:latest")
    monkeypatch.setenv("RAG_TOP_K_DEFAULT", "3")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD_DEFAULT", "0.45")

    config = load_config()

    assert config.ollama_model == "tinyllama:latest"
    assert config.default_top_k == 3
    assert config.default_score_threshold == 0.45
