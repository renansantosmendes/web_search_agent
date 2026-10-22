"""FastAPI wrapper around the web research agent, for deployment on Vercel.

Endpoints:
    GET  /health   liveness check
    POST /chat     send a message, get the agent's reply
    GET  /search   run a Tavily search and return the complete response
    GET  /query_with_summary   the same, with a one-sentence summary per result

Conversations are kept by session_id. Send the session_id returned by the
first call to continue the same conversation.

Limitation: sessions live in the memory of the instance that answered. On
Vercel a new instance (cold start) starts with no sessions, so for durable
history the sessions should move to a database, as finbrain-agent does.
"""

import os
import threading
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from tavily import TavilyClient

from agent import create_agent
from tools.content_cleaner import clean_markdown
from tools.summarizer import make_summary_model, summarize

MAX_SESSIONS = 200  # bound on the sessions kept by one warm instance
MAX_SUMMARY_RESULTS = 10  # results summarized per request, to limit model cost

API_DESCRIPTION = """
Web research agent. It answers questions using web searches (Tavily) and
page reading, and keeps the conversation by `session_id`.

- **`/chat`** talks to the agent. The agent searches the web when it needs
  current information and cites the source URLs in its answer.
- **`/search`** runs a Tavily search directly, without the agent, and returns
  the complete response, including the cleaned text of each page.
- **`/query_with_summary`** returns the same as `/search`, plus a one-sentence
  `summary` of each page, written by gpt-4o-mini.

Sessions are kept in the memory of one server instance. They are lost when
the instance restarts or a new one starts.
"""

app = FastAPI(
    title="Web Research Agent API",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=[
        {"name": "system", "description": "Service status."},
        {"name": "agent", "description": "Chat with the research agent."},
        {"name": "search", "description": "Direct Tavily web search."},
    ],
)

_agent = None
_sessions: dict[str, list] = {}
_lock = threading.Lock()


class ChatRequest(BaseModel):
    """Body of `POST /chat`."""

    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Message to send to the agent, up to 4000 characters.",
        examples=["What is the current USD to BRL exchange rate?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description=(
            "Session to continue. Leave it out to start a new conversation; "
            "the new `session_id` is returned in the response."
        ),
        examples=["3f2b9c1e-6d0a-4a7e-9d3b-2c1f0e9a8b7c"],
    )


class ChatResponse(BaseModel):
    """Reply of `POST /chat`."""

    session_id: str = Field(
        description="Session of the conversation. Send it in the next call "
        "to continue the same conversation."
    )
    reply: str = Field(
        description="Answer of the agent. It may cite the source URLs."
    )


class ErrorResponse(BaseModel):
    """Error body returned by the API."""

    detail: str = Field(description="Short description of the error.")


def get_agent():
    """Return the agent, creating it on first use.

    Returns
    -------
    object
        The compiled deep agent shared by the requests of this instance.
    """
    global _agent
    with _lock:
        if _agent is None:
            _agent = create_agent()
        return _agent


def _text(content) -> str:
    """Plain text of a message content, which may be a string or a list of parts.

    Parameters
    ----------
    content : str | list
        The ``content`` of a model message.

    Returns
    -------
    str
        The text of the message.
    """
    if isinstance(content, str):
        return content
    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
    return "".join(parts)


def _remember(session_id: str, messages: list) -> None:
    """Store the messages of a session, dropping the oldest sessions when full.

    Parameters
    ----------
    session_id : str
        Session to update.
    messages : list
        Full message history of the session after the turn.
    """
    with _lock:
        _sessions.pop(session_id, None)
        _sessions[session_id] = messages
        while len(_sessions) > MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))


@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    response_description="The service is running.",
)
def health() -> dict:
    """Report that the API is running.

    Use it to check that the service is up, for example in uptime monitors.

    Returns
    -------
    dict
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["agent"],
    summary="Send a message to the research agent",
    responses={500: {"model": ErrorResponse, "description": "The agent failed."}},
)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a message, continuing the conversation of the given session.

    The agent has the `web_search` tool (Tavily), a `scrape_page` tool that
    reads a URL, and the skills in the `skills/` folder. It answers in
    Brazilian Portuguese and cites the sources it used.

    To start a conversation, send only `message`. To continue it, send the
    `session_id` returned by the previous call.

    Raises
    ------
    HTTPException
        500 if the agent fails to answer.
    """
    session_id = request.session_id or str(uuid.uuid4())
    with _lock:
        history = list(_sessions.get(session_id, []))
    history.append({"role": "user", "content": request.message})

    try:
        response = get_agent().invoke({"messages": history})
    except Exception:
        raise HTTPException(status_code=500, detail="Agent invocation failed")

    messages = response["messages"]
    _remember(session_id, messages)
    return ChatResponse(session_id=session_id, reply=_text(messages[-1].content))


@app.get(
    "/search",
    tags=["search"],
    summary="Search the web with Tavily",
    responses={502: {"model": ErrorResponse, "description": "Tavily failed."}},
)
def search(
    query: str = Query(
        min_length=1,
        max_length=400,
        description="Search query, from 1 to 400 characters.",
        examples=["exchange rate dollar real today"],
    ),
    max_results: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return, from 1 to 20 (Tavily's limit).",
    ),
) -> dict:
    """Run a Tavily search and return the complete response.

    The search is advanced and returns, in addition to the Tavily answer:

    - `answer`: short answer to the query.
    - `results`: list of pages. Each item has `title`, `url`, `content` (the
      Tavily excerpt), `score`, `favicon`, `raw_content` (the full page text
      in markdown) and `cleaned_content` (the same text without images,
      navigation links and extra blank lines).
    - `images`: related images.
    - `usage`: credits used by the request.
    - `query`, `response_time` and `request_id`: request metadata.

    Raises
    ------
    HTTPException
        502 if the Tavily request fails.
    """
    return _run_search(query, max_results)


@app.get(
    "/query_with_summary",
    tags=["search"],
    summary="Search the web with Tavily and summarize each result",
    responses={
        502: {"model": ErrorResponse, "description": "Tavily or the model failed."}
    },
)
def query_with_summary(
    query: str = Query(
        min_length=1,
        max_length=400,
        description="Search query, from 1 to 400 characters.",
        examples=["exchange rate dollar real today"],
    ),
    max_results: int = Query(
        default=5,
        ge=1,
        description=(
            "Number of results to return. Values above 10 are cut to 10 to "
            "limit the cost of the summaries."
        ),
    ),
) -> dict:
    """Run a Tavily search and add a one-sentence summary to each result.

    The response is the same as `/search`, and each item of `results` also has
    a `summary`: one short, direct sentence written by gpt-4o-mini from the
    `cleaned_content` of the page. The summary is `null` when the page has no
    text. At most 10 results are returned, even if more are requested.

    Raises
    ------
    HTTPException
        502 if the Tavily request or a summary fails.
    """
    response = _run_search(query, min(max_results, MAX_SUMMARY_RESULTS))

    try:
        model = make_summary_model()
        for result in response.get("results", []):
            result["summary"] = summarize(result["cleaned_content"], model)
    except Exception:
        raise HTTPException(status_code=502, detail="Summary generation failed")
    return response


def _run_search(query: str, max_results: int) -> dict:
    """Call Tavily and add the cleaned text of each result.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Number of results to return.

    Returns
    -------
    dict
        The Tavily response, where each result also has ``cleaned_content``.

    Raises
    ------
    HTTPException
        502 if the Tavily request fails.
    """
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer="advanced",
            include_raw_content="markdown",
            include_images=True,
            include_favicon=True,
            include_usage=True,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Search failed")

    for result in response.get("results", []):
        result["cleaned_content"] = clean_markdown(result.get("raw_content") or "")
    return response
