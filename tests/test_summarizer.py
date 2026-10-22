"""Tests of the summarizer, with a fake chat model so no API is called."""

from tools.summarizer import MAX_INPUT_CHARS, SYSTEM_PROMPT, summarize


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeModel:
    """Records the messages it receives and answers with a fixed sentence."""

    def __init__(self, answer: str = "Frase curta."):
        self.answer = answer
        self.received = []

    def invoke(self, messages):
        self.received.append(messages)
        return FakeMessage(self.answer)


def test_summary_is_the_model_answer_without_surrounding_spaces():
    model = FakeModel(answer="  O dólar caiu.  \n")

    assert summarize("Texto sobre o dólar.", model) == "O dólar caiu."


def test_empty_text_returns_none_without_calling_the_model():
    model = FakeModel()

    assert summarize("   \n ", model) is None
    assert model.received == []


def test_the_system_prompt_and_the_text_are_sent_in_that_order():
    model = FakeModel()

    summarize("Texto da página.", model)

    assert model.received == [
        [("system", SYSTEM_PROMPT), ("human", "Texto da página.")]
    ]


def test_long_text_is_cut_before_being_sent():
    model = FakeModel()

    summarize("a" * (MAX_INPUT_CHARS + 500), model)

    sent_text = model.received[0][1][1]
    assert len(sent_text) == MAX_INPUT_CHARS
