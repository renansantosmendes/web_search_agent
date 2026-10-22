"""Tests of the HTTP API with a fake agent, so no model or search API is called."""

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeAgent:
    """Records each call and answers with the number of messages it received."""

    def __init__(self):
        self.calls = []

    def invoke(self, inputs):
        messages = inputs["messages"]
        self.calls.append(len(messages))
        history = list(messages) + [FakeMessage(f"resposta {len(messages)}")]
        return {"messages": history}


class FailingAgent:
    def invoke(self, inputs):
        raise RuntimeError("model unavailable")


@pytest.fixture
def fake_agent(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(app_module, "_agent", agent)
    app_module._sessions.clear()
    yield agent
    app_module._sessions.clear()


@pytest.fixture
def client(fake_agent):
    return TestClient(app)


def test_health_answers_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_starts_a_session_when_none_is_sent(client, fake_agent):
    response = client.post("/chat", json={"message": "oi"})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["reply"] == "resposta 1"
    assert fake_agent.calls == [1]


def test_chat_continues_the_same_session(client, fake_agent):
    first = client.post("/chat", json={"message": "oi"}).json()
    second = client.post(
        "/chat", json={"message": "e agora?", "session_id": first["session_id"]}
    ).json()

    assert second["session_id"] == first["session_id"]
    assert fake_agent.calls == [1, 3]


def test_different_sessions_do_not_share_history(client, fake_agent):
    client.post("/chat", json={"message": "a", "session_id": "one"})
    client.post("/chat", json={"message": "b", "session_id": "two"})

    assert fake_agent.calls == [1, 1]


@pytest.mark.parametrize("body", [{"message": ""}, {}, {"message": "x" * 4001}])
def test_invalid_messages_return_422(client, body):
    assert client.post("/chat", json=body).status_code == 422


def test_agent_failure_returns_500_without_details(monkeypatch, client):
    monkeypatch.setattr(app_module, "_agent", FailingAgent())

    response = client.post("/chat", json={"message": "oi"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Agent invocation failed"


def test_the_oldest_sessions_are_dropped_when_full(client, fake_agent, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_SESSIONS", 2)
    for name in ("a", "b", "c"):
        client.post("/chat", json={"message": "x", "session_id": name})

    assert list(app_module._sessions) == ["b", "c"]
