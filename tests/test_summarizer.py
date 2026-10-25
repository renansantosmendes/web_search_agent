"""Tests of the chat model used by the news chain."""

from tools.summarizer import MODEL_NAME, make_summary_model


def test_the_model_is_gpt_4o_mini_with_temperature_zero(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = make_summary_model()

    assert model.model_name == MODEL_NAME == "gpt-4o-mini"
    assert model.temperature == 0
