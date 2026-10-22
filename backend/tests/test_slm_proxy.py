"""POST /slm/answer and GET /slm/status: the proxy to the optional SLM service, and the live free-text path.

Nothing here talks to a real SLM or to Gemini. The SLM service is faked with httpx.MockTransport, the chat model
and the tools of the live pipeline with plain fakes, and every outside connection is refused.
"""

import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as api_app
from api import slm_proxy
from api.schemas import ReportResponse
from orchestrator import config, graph
from retrieval import config as rconfig

client = TestClient(api_app.app)
URL = "http://slm.test:8001"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No backend/.env is read, no key is present, and no connection leaves the machine."""
    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(config.SLM_URL_ENV, raising=False)
    monkeypatch.delenv(config.SLM_FLAG_ENV, raising=False)
    monkeypatch.delenv("CLIMATE_SUITE_MODE", raising=False)
    for var in (*rconfig.API_KEY_ENV_VARS, "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    loopback = {"127.0.0.1", "::1", "localhost"}
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] in loopback:
            return real_connect(sock, address, *args, **kwargs)
        raise AssertionError(f"tried to connect to {address!r}")

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host in loopback or host is None:
            return real_getaddrinfo(host, *args, **kwargs)
        raise AssertionError(f"tried to resolve {host!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


def fake_service(monkeypatch, handler):
    monkeypatch.setattr(slm_proxy, "make_client",
                        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout))


def ok_service(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True, "model": "fake-slm"})
        body = json.loads(request.content)
        seen.append(body)
        refused = "unrelated" in body["question"]
        text = "I cannot answer this from the provided context." if refused else "Rajasthan."
        return httpx.Response(200, json={"answer": text, "refused": refused, "model": "fake-slm",
                                         "latency_ms": 9.9, "n_prompt_tokens": 40, "n_new_tokens": 3,
                                         "request_id": "r"})
    return handler


BODY = {"context": "Barmer is in Rajasthan.", "question": "Which state is Barmer in?"}


# --------------------------------------------------------------------------- #
# /slm/answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [None, "demo", "live"])
def test_an_answer_is_proxied_in_every_mode(monkeypatch, mode):
    if mode:
        monkeypatch.setenv("CLIMATE_SUITE_MODE", mode)
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    seen = []
    fake_service(monkeypatch, ok_service(seen))
    response = client.post("/slm/answer", json=BODY)
    assert response.status_code == 200
    assert response.json() == {"answer": "Rajasthan.", "refused": False, "model": "fake-slm"}
    assert seen[0]["context"] == BODY["context"] and seen[0]["question"] == BODY["question"]
    assert seen[0]["max_new_tokens"] == config.SLM_MAX_NEW_TOKENS


def test_a_refusal_is_passed_on_as_refused_true(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, ok_service([]))
    response = client.post("/slm/answer", json={**BODY, "question": "an unrelated question?"})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert response.json()["answer"].startswith("I cannot answer")


def test_the_response_has_only_answer_refused_and_model(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, ok_service([]))
    assert set(client.post("/slm/answer", json=BODY).json()) == {"answer", "refused", "model"}


def test_no_url_is_a_clear_503_and_no_request_is_made(monkeypatch):
    def boom(timeout):
        raise AssertionError("a client was made although SLM_URL is not set")

    monkeypatch.setattr(slm_proxy, "make_client", boom)
    response = client.post("/slm/answer", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "SLM service is not running; start it and set SLM_URL"


def test_an_empty_url_counts_as_unset(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, "  ")
    assert client.post("/slm/answer", json=BODY).status_code == 503


@pytest.mark.parametrize("body, ok", [
    ({"context": "c" * 4000, "question": "q" * 500}, True),
    ({"context": "c" * 4001, "question": "q"}, False),
    ({"context": "c", "question": "q" * 501}, False),
    ({"context": "", "question": "q"}, False),
    ({"context": "c", "question": ""}, False),
    ({"context": "c"}, False),
    ({"context": "c", "question": "q", "max_new_tokens": 9999}, False),
])
def test_size_limits_and_shape(monkeypatch, body, ok):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    seen = []
    fake_service(monkeypatch, ok_service(seen))
    response = client.post("/slm/answer", json=body)
    assert response.status_code == (200 if ok else 422)
    assert len(seen) == (1 if ok else 0), "an invalid request must not reach the SLM service"


def test_an_unreachable_service_is_a_503(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)

    def refuse(request):
        raise httpx.ConnectError("refused")

    fake_service(monkeypatch, refuse)
    assert client.post("/slm/answer", json=BODY).status_code == 503


def test_a_timeout_is_a_504(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)

    def slow(request):
        raise httpx.ReadTimeout("slow")

    fake_service(monkeypatch, slow)
    assert client.post("/slm/answer", json=BODY).status_code == 504


def test_the_call_has_a_timeout(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    timeouts = []
    real = slm_proxy.make_client

    def spy(timeout):
        timeouts.append(timeout)
        return httpx.Client(transport=httpx.MockTransport(ok_service([])), timeout=timeout)

    monkeypatch.setattr(slm_proxy, "make_client", spy)
    client.post("/slm/answer", json=BODY)
    assert timeouts == [slm_proxy.CALL_TIMEOUT_S] and real is not spy


@pytest.mark.parametrize("status, expected", [(413, 413), (503, 503), (500, 503), (422, 503)])
def test_service_errors_become_short_messages(monkeypatch, status, expected):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, lambda request: httpx.Response(status, json={"detail": "stack trace here"}))
    response = client.post("/slm/answer", json=BODY)
    assert response.status_code == expected
    assert "stack trace" not in response.text


def test_a_malformed_answer_is_a_502(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, lambda request: httpx.Response(200, json={"nothing": 1}))
    assert client.post("/slm/answer", json=BODY).status_code == 502


def test_nothing_is_logged_or_stored(monkeypatch, caplog):
    import inspect

    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, ok_service([]))
    with caplog.at_level("DEBUG"):
        client.post("/slm/answer", json={"context": "SECRET-PASSAGE", "question": "SECRET-QUESTION?"})
    assert "SECRET" not in caplog.text
    source = inspect.getsource(slm_proxy)
    assert "logging" not in source and "open(" not in source and "write_text" not in source


def test_the_proxy_makes_no_gemini_call():
    import inspect

    source = inspect.getsource(slm_proxy)
    assert "invoke_with_backoff" not in source and "get_chat_model" not in source and "CALL_TALLY" not in source


# --------------------------------------------------------------------------- #
# /slm/status
# --------------------------------------------------------------------------- #
def test_status_without_a_url():
    body = client.get("/slm/status").json()
    assert body["configured"] is False and body["available"] is False
    assert body["message"] == "SLM service is not running; start it and set SLM_URL"


def test_status_with_a_running_service(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, ok_service([]))
    assert client.get("/slm/status").json() == {"configured": True, "available": True, "model": "fake-slm", "message": ""}


def test_status_with_a_url_but_no_service(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)

    def refuse(request):
        raise httpx.ConnectError("refused")

    fake_service(monkeypatch, refuse)
    body = client.get("/slm/status").json()
    assert body["configured"] is True and body["available"] is False


def test_status_with_a_model_still_loading(monkeypatch):
    monkeypatch.setenv(config.SLM_URL_ENV, URL)
    fake_service(monkeypatch, lambda request: httpx.Response(200, json={"status": "loading", "model_loaded": False}))
    assert client.get("/slm/status").json()["available"] is False


# --------------------------------------------------------------------------- #
# Live mode: free text goes through POST /report (fake chat model, fake tools, no network)
# --------------------------------------------------------------------------- #
class FakeTool:
    args = {"region": 1, "month": 1, "crop": 1, "query": 1, "k": 1}

    def __init__(self, result):
        self.result, self.calls = result, []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


def test_live_free_text_runs_the_pipeline_and_returns_a_rendered_report(monkeypatch):
    monkeypatch.setenv("CLIMATE_SUITE_MODE", "live")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")   # a placeholder: the model and tools are faked
    question = "Will it be dry in Barmer next month?"
    drought = {"region": "barmer", "forecast_anchor_month": "2024-12-01", "predicted_values": [0.1, 0.2, 0.3],
               "horizon_confidence": [{"horizon": "t+1", "month": "2025-01-01", "skill_score": 0.2053,
                                       "method": "direct", "label": "validated"}]}
    tools = {"forecast_drought_risk": FakeTool(drought),
             "forecast_heat_stress_risk": FakeTool({"region": "barmer", "forecast_available": False,
                                                    "note": "no usable skill"}),
             "retrieve_context": FakeTool([{"source": "Doc A", "citation": "https://example.org/a",
                                            "text": "A passage.", "section": "s", "score": 0.5}])}
    monkeypatch.setattr(graph, "TOOLS_BY_NAME", tools)
    monkeypatch.setattr(graph, "fetch_outlooks", lambda: {"outlooks": []})
    monkeypatch.setattr(graph, "fetch_external_sources", lambda **k: {"sources": []})
    prompts = []

    class Reply:
        def __init__(self, content="", tool_calls=()):
            self.content, self.tool_calls = content, list(tool_calls)

    def fake_invoke(model, messages, attempts=3):
        if model == "router":
            return Reply()                       # no tool calls: the pipeline falls back to its defaults
        prompts.append(messages)
        return Reply("## Summary\n\nThe skill score at the first horizon is 0.2053.")

    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: "router" if tools else "writer")
    monkeypatch.setattr(graph, "invoke_with_backoff", fake_invoke)

    response = client.post("/report", json={"request": question})
    assert response.status_code == 200
    body = ReportResponse(**response.json()).model_dump()
    assert body["request"] == question
    assert body["report"].startswith("## Summary") and "0.2053" in body["report"]
    assert body["grounding"]["status"] == "clean" and body["grounding"]["grounded"] is True
    assert body["meta"]["mode"] == "live" and body["meta"]["saved_example"] is False
    assert "forecast_drought_risk" in body["tools_called"]
    assert body["horizon_confidence"][0]["label"] == "validated"
    assert body["slm"] is None, "the SLM step is off by default"
    assert question in json.dumps(prompts), "the question reached the (fake) writer"
