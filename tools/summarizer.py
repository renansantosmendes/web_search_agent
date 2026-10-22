"""One-sentence summaries of page text, generated with gpt-4o-mini.

Used by the query_with_summary route to give each search result a short,
direct sentence that the user can read at a glance.
"""

from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"
MAX_INPUT_CHARS = 8000

SYSTEM_PROMPT = (
    "Resuma o texto do usuário em apenas uma frase, curtíssima, direta e "
    "chamativa, em português do Brasil. Responda somente com a frase, sem "
    "aspas, rótulos ou explicações."
)


def make_summary_model() -> ChatOpenAI:
    """Create the chat model used for summaries.

    Returns
    -------
    ChatOpenAI
        A gpt-4o-mini model with temperature 0, so the same text gives the
        same summary.
    """
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def summarize(text: str, model: ChatOpenAI) -> str | None:
    """Summarize a text in one short sentence.

    Only the first ``MAX_INPUT_CHARS`` characters of the text are sent to the
    model, to keep the cost of long pages bounded.

    Parameters
    ----------
    text : str
        Text to summarize, usually the ``cleaned_content`` of a result.
    model : ChatOpenAI
        Chat model that answers the request.

    Returns
    -------
    str | None
        The one-sentence summary, or None if the text is empty (no model call
        is made in that case).

    Example
    -------
    >>> summarize("Dólar cai para R$ 5,01 com expectativa eleitoral.", model)
    'O dólar cai para R$ 5,01 com a expectativa eleitoral.'
    """
    text = text.strip()
    if not text:
        return None

    reply = model.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", text[:MAX_INPUT_CHARS]),
        ]
    )
    return reply.content.strip()
