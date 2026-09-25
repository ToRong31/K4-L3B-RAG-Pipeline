"""Keep contract tests fully offline even when a developer has API keys in .env."""

import pytest


@pytest.fixture(autouse=True)
def disable_optional_providers(monkeypatch):
    monkeypatch.setenv("QUERY_FORMULATION_ENABLED", "false")
    monkeypatch.setenv("COHERE_RERANK_ENABLED", "false")
    monkeypatch.setenv("HYBRID_WEIGHTED_ENABLED", "false")
    monkeypatch.setenv("MULTI_DENSE_ENABLED", "false")
    monkeypatch.setenv("DENSE_BACKEND", "shared")
