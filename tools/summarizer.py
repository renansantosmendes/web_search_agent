"""Chat model that extracts and summarizes the news of the search results.

The model is used by the news chain (tools/news_chain.py), which the
query_with_summary route runs for each result. The model is created once per
request, with temperature 0, so the same page gives the same article and
summary.
"""

from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"


def make_summary_model() -> ChatOpenAI:
    """Create the chat model used by the news chain.

    Returns
    -------
    ChatOpenAI
        A gpt-4o-mini model with temperature 0.

    Example
    -------
    >>> model = make_summary_model()  # doctest: +SKIP
    """
    return ChatOpenAI(model=MODEL_NAME, temperature=0)
