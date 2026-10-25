"""Tests of the news chain, with a fake chat model so no API is called."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tools.news_chain import (
    MIN_TEXT_CHARS,
    NOT_AN_ARTICLE,
    build_news_chain,
    clean_summary,
    has_enough_text,
    normalize_article,
    prepare_inputs,
)

LONG_TEXT = "Menu. " + "O dólar subiu hoje com a expectativa do mercado. " * 10


def make_model(responses: list[str]) -> tuple[FakeListChatModel, list]:
    """Create a fake chat model that records the messages of every call.

    Parameters
    ----------
    responses : list[str]
        Answers returned in order, one per model call.

    Returns
    -------
    tuple[FakeListChatModel, list]
        The model, and the list that receives the messages of each call.
    """
    received: list = []

    class RecordingModel(FakeListChatModel):
        def invoke(self, input, config=None, **kwargs):
            received.append(input.to_messages())
            return super().invoke(input, config, **kwargs)

    return RecordingModel(responses=responses), received


def test_the_chain_extracts_the_article_and_then_summarizes_it():
    model, received = make_model(["Corpo da notícia sobre o dólar.", "O dólar subiu."])

    result = build_news_chain(model).invoke({"title": "Dólar", "text": LONG_TEXT})

    assert result["article"] == "Corpo da notícia sobre o dólar."
    assert result["summary"] == "O dólar subiu."
    assert result["title"] == "Dólar"
    assert len(received) == 2


def test_extraction_receives_the_title_and_the_page_text():
    model, received = make_model(["Corpo.", "Resumo."])

    build_news_chain(model).invoke({"title": "Dólar", "text": LONG_TEXT})

    human = received[0][-1].content
    assert human.startswith("Título: Dólar")
    assert "Texto da página:" in human
    assert "O dólar subiu hoje" in human


def test_summary_receives_only_the_extracted_article():
    model, received = make_model(["Corpo limpo da notícia.", "Resumo."])

    build_news_chain(model).invoke({"title": "T", "text": LONG_TEXT})

    assert received[1][-1].content == "Corpo limpo da notícia."


def test_short_text_skips_the_model_and_gives_no_summary():
    model, received = make_model(["nunca usado"])

    result = build_news_chain(model).invoke({"title": "T", "text": "curto"})

    assert result == {"title": "T", "text": "curto", "article": "", "summary": None}
    assert received == []


def test_page_that_is_not_news_gives_no_summary():
    model, received = make_model([NOT_AN_ARTICLE + "."])

    result = build_news_chain(model).invoke({"title": "Menu", "text": LONG_TEXT})

    assert result["article"] == ""
    assert result["summary"] is None
    assert len(received) == 1


def test_prepare_inputs_strips_the_title_and_cuts_the_text():
    prepared = prepare_inputs({"title": "  Dólar  ", "text": "x" * 9000})
    assert prepared["title"] == "Dólar"
    assert len(prepared["text"]) == 8000


def test_prepare_inputs_reads_missing_values_as_empty():
    assert prepare_inputs({}) == {"title": "", "text": ""}


def test_text_needs_the_minimum_length_to_be_worth_a_call():
    assert has_enough_text({"title": "", "text": "x" * MIN_TEXT_CHARS})
    assert not has_enough_text({"title": "", "text": "x" * (MIN_TEXT_CHARS - 1)})


def test_normalize_article_drops_the_not_an_article_marker():
    assert normalize_article("  " + NOT_AN_ARTICLE + ". ") == ""
    assert normalize_article("\nCorpo da notícia.\n") == "Corpo da notícia."


def test_clean_summary_removes_surrounding_quotes_and_spaces():
    assert clean_summary(' "O dólar caiu." ') == "O dólar caiu."
    assert clean_summary("“Selic sobe.”") == "Selic sobe."
