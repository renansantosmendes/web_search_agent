"""Tests of the Langfuse tracing switch, with the Langfuse client replaced."""

import pytest

from tools import tracing


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")


def test_tracing_is_off_without_the_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert tracing.tracing_enabled() is False
    assert tracing.tracing_callbacks() == []


def test_tracing_needs_both_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert tracing.tracing_enabled() is False


def test_tracing_adds_a_langfuse_callback_when_configured(monkeypatch, keys):
    monkeypatch.setattr(tracing, "CallbackHandler", lambda: "langfuse-handler")

    assert tracing.tracing_enabled() is True
    assert tracing.tracing_callbacks() == ["langfuse-handler"]


def test_flush_sends_the_pending_traces_when_configured(monkeypatch, keys):
    sent = []

    class FakeClient:
        def flush(self):
            sent.append("flushed")

    monkeypatch.setattr(tracing, "get_client", lambda: FakeClient())

    tracing.flush_tracing()

    assert sent == ["flushed"]


def test_flush_does_nothing_without_the_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "get_client", lambda: pytest.fail("no client"))

    tracing.flush_tracing()
