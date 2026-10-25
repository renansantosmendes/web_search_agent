"""Chain that turns the text of a web page into a news article and a summary.

The page text (from Tavily) mixes the news with menus, quotes, ads, footers
and links to other stories. The chain runs in two steps, both with a prompt
and the same chat model:

1. extract the body of the news from the page text;
2. write one short sentence from that body.

A RunnableBranch skips the model when the page has too little text, and skips
the summary when no news body was found. Each step is recorded by Langfuse
when the caller passes the tracing callbacks in the run config.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

MAX_INPUT_CHARS = 8000
MIN_TEXT_CHARS = 300
NOT_AN_ARTICLE = "SEM_NOTICIA"

EXTRACT_SYSTEM_PROMPT = (
    "Você recebe o título e o texto de uma página da web. Extraia somente o "
    "corpo da notícia: o relato dos fatos. Descarte menus, cotações, "
    "anúncios, rodapés, legendas, créditos, listas de outras notícias e links. "
    "Mantenha o texto em português do Brasil, com os parágrafos separados por "
    "uma linha em branco e os subtítulos em markdown (##), quando houver. "
    "Não acrescente nem invente informação. Se a página não for uma notícia, "
    "responda somente: " + NOT_AN_ARTICLE + "."
)

SUMMARY_SYSTEM_PROMPT = (
    "Resuma a notícia em uma única frase curta, direta e informativa, em "
    "português do Brasil. Responda somente com a frase, sem aspas, rótulos ou "
    "explicações."
)

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_SYSTEM_PROMPT),
        ("human", "Título: {title}\n\nTexto da página:\n{text}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", "{article}"),
    ]
)


def prepare_inputs(inputs: dict[str, Any]) -> dict[str, str]:
    """Normalize the title and cut the text to the size sent to the model.

    Parameters
    ----------
    inputs : dict[str, Any]
        Dictionary with the ``title`` and the ``text`` of the page. Missing
        values are read as empty strings.

    Returns
    -------
    dict[str, str]
        Dictionary with the stripped ``title`` and the ``text`` cut to
        ``MAX_INPUT_CHARS``.

    Example
    -------
    >>> prepare_inputs({"title": " Dólar ", "text": "x" * 10})
    {'title': 'Dólar', 'text': 'xxxxxxxxxx'}
    """
    title = str(inputs.get("title") or "").strip()
    text = str(inputs.get("text") or "")
    return {"title": title, "text": text[:MAX_INPUT_CHARS]}


def has_enough_text(inputs: dict[str, str]) -> bool:
    """Tell whether the page has enough text to be worth a model call.

    Parameters
    ----------
    inputs : dict[str, str]
        Output of ``prepare_inputs``.

    Returns
    -------
    bool
        True when the text has at least ``MIN_TEXT_CHARS`` characters.

    Example
    -------
    >>> has_enough_text({"title": "", "text": "curto"})
    False
    """
    return len(inputs["text"].strip()) >= MIN_TEXT_CHARS


def normalize_article(reply: str) -> str:
    """Turn the answer of the extraction step into the article text.

    Parameters
    ----------
    reply : str
        Answer of the model to the extraction prompt.

    Returns
    -------
    str
        The article text, or an empty string when the model reported that
        the page is not a news article.

    Example
    -------
    >>> normalize_article("SEM_NOTICIA.")
    ''
    """
    text = reply.strip()
    if text.upper().startswith(NOT_AN_ARTICLE):
        return ""
    return text


def clean_summary(reply: str) -> str:
    """Remove the spacing and quotes around a one-sentence summary.

    Parameters
    ----------
    reply : str
        Answer of the model to the summary prompt.

    Returns
    -------
    str
        The sentence without surrounding spaces and quotes.

    Example
    -------
    >>> clean_summary(' "O dólar caiu." ')
    'O dólar caiu.'
    """
    return reply.strip().strip("\"'“”‘’").strip()


def build_news_chain(model: BaseChatModel) -> Runnable:
    """Build the chain that extracts the article and summarizes it.

    Parameters
    ----------
    model : BaseChatModel
        Chat model used by both steps.

    Returns
    -------
    Runnable
        Runnable that takes ``{"title": str, "text": str}`` and returns
        ``{"title", "text", "article", "summary"}``. ``article`` is an empty
        string and ``summary`` is None when there is no news body.

    Example
    -------
    >>> chain = build_news_chain(model)  # doctest: +SKIP
    >>> chain.invoke({"title": "Dólar", "text": page_text})  # doctest: +SKIP
    {'title': 'Dólar', 'text': '...', 'article': '...', 'summary': '...'}
    """
    extract = EXTRACT_PROMPT | model | StrOutputParser() | RunnableLambda(normalize_article)
    summarize = (
        RunnableLambda(lambda inputs: {"article": inputs["article"]})
        | SUMMARY_PROMPT
        | model
        | StrOutputParser()
        | RunnableLambda(clean_summary)
    )
    article_branch = RunnableBranch(
        (has_enough_text, extract),
        RunnableLambda(lambda inputs: ""),
    )
    summary_branch = RunnableBranch(
        (lambda inputs: bool(inputs["article"]), summarize),
        RunnableLambda(lambda inputs: None),
    )
    return (
        RunnableLambda(prepare_inputs)
        | RunnablePassthrough.assign(article=article_branch)
        | RunnablePassthrough.assign(summary=summary_branch)
    )
