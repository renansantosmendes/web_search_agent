"""FastAPI wrapper around the web research agent, for deployment on Vercel.

Endpoints:
    GET  /health   liveness check
    POST /chat     send a message, get the agent's reply

Conversations are kept by session_id. Send the session_id returned by the
first call to continue the same conversation.

Limitation: sessions live in the memory of the instance that answered. On
Vercel a new instance (cold start) starts with no sessions, so for durable
history the sessions should move to a database, as finbrain-agent does.
"""

import threading
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import create_agent

MAX_SESSIONS = 200  # bound on the sessions kept by one warm instance

app = FastAPI(title="Web Research Agent API")

_agent = None
_sessions: dict[str, list] = {}
_lock = threading.Lock()


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    """Reply of POST /chat."""

    session_id: str
    reply: str


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


@app.get("/health")
def health() -> dict:
    """Report that the API is running.

    Returns
    -------
    dict
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a message, continuing the conversation of the given session.

    Parameters
    ----------
    request : ChatRequest
        The message and, optionally, the session to continue. Without a
        session_id a new session is started.

    Returns
    -------
    ChatResponse
        The session_id to use in the next call and the agent's reply.

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
