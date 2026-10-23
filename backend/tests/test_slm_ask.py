"""POST /slm/ask: the SLM answers from the project's own material (a report plus matching source passages).

The SLM service is faked with httpx.MockTransport; the corpus search is replaced by fixed passages; no Gemini, no network.
"""

import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as api_app
from api import demo, slm_context, slm_proxy
from orchestrator import config
from retrieval import config as rconfig

client = TestClient(api_app.app)
REAL_CORPUS_PASSAGES = slm_context.corpus_passages      # kept before the fixture replaces it
PASSAGES = [{"id": "a", "source": "IMD heat wave FAQ", "text": "A heat wave is declared when the maximum temperature reaches a threshold. " * 4},
            {"id": "b", "source": "SPI guide", "text": "SPI-3 is a three-month standardised precipitation index. " * 4}]


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)
    monkeypatch.setenv(config.SLM_URL_ENV, "http://slm.test:8001")
    monkeypatch.delenv("CLIMATE_SUITE_MODE", raising=False)
    monkeypatch.setattr(slm_context, "corpus_passages", lambda q, n=slm_context.PASSAGES: PASSAGES)
    real_connect = socket.socket.connect

    def guarded(sock, address, *a, **k):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
            return real_connect(sock, address, *a, **k)
        raise AssertionError(f"tried to connect to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)


def fake_service(monkeypatch, seen, refuse=False):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = "I cannot answer this from the provided context." if refuse else "From the report: t+1 is validated."
        return httpx.Response(200, json={"answer": text, "refused": refuse, "model": "fake-slm"})

    monkeypatch.setattr(slm_proxy, "make_client",
                        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout))


def test_a_saved_report_and_corpus_passages_form_the_context(monkeypatch):
    seen = []
    fake_service(monkeypatch, seen)
    response = client.post("/slm/ask", json={"question": "Is the next month forecast reliable?", "report_id": "drought_one_month"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] and body["refused"] is False and body["model"] == "fake-slm"
    assert [s["kind"] for s in body["sources"]] == ["report", "corpus", "corpus"]
    context = seen[0]["context"]
    assert len(context) <= slm_proxy.MAX_CONTEXT_CHARS
    assert "[report:" in context and "[corpus: IMD heat wave FAQ]" in context
    assert "label: validated" in context, "the forecast labels of the report reach the model"
    assert seen[0]["question"] == "Is the next month forecast reliable?"


def test_report_text_from_the_page_is_used(monkeypatch):
    seen = []
    fake_service(monkeypatch, seen)
    response = client.post("/slm/ask", json={"question": "q?", "report_text": "The page's own report text."})
    assert response.status_code == 200
    assert "The page's own report text." in seen[0]["context"]


def test_a_refusal_passes_through(monkeypatch):
    fake_service(monkeypatch, [], refuse=True)
    body = client.post("/slm/ask", json={"question": "q?", "report_id": "drought_one_month"}).json()
    assert body["refused"] is True


def test_works_with_the_corpus_alone(monkeypatch):
    seen = []
    fake_service(monkeypatch, seen)
    body = client.post("/slm/ask", json={"question": "what is SPI-3?"}).json()
    assert [s["kind"] for s in body["sources"]] == ["corpus", "corpus"]


def test_no_material_is_a_clear_404_and_the_service_is_not_called(monkeypatch):
    monkeypatch.setattr(slm_context, "corpus_passages", lambda q, n=4: [])
    fake_service(monkeypatch, (calls := []))
    response = client.post("/slm/ask", json={"question": "q?"})
    assert response.status_code == 404 and "no report or source corpus" in response.json()["detail"].lower()
    assert calls == []


def test_unknown_report_id_is_a_404(monkeypatch):
    fake_service(monkeypatch, [])
    assert client.post("/slm/ask", json={"question": "q?", "report_id": "nope"}).status_code == 404


def test_unset_url_is_the_same_503(monkeypatch):
    monkeypatch.delenv(config.SLM_URL_ENV)
    response = client.post("/slm/ask", json={"question": "q?", "report_id": "drought_one_month"})
    assert response.status_code == 503 and response.json()["detail"] == slm_proxy.UNSET_MESSAGE


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "q" * 501}, {"question": "q", "report_text": "x" * 20001},
                                  {"question": "q", "extra": 1}, {}])
def test_limits_and_shape(monkeypatch, body):
    seen = []
    fake_service(monkeypatch, seen)
    assert client.post("/slm/ask", json=body).status_code == 422 and seen == []


def test_the_context_never_exceeds_the_limit_and_says_when_it_cut(monkeypatch):
    seen = []
    fake_service(monkeypatch, seen)
    long = {"question": "q?", "report_text": "R" * 20000}
    body = client.post("/slm/ask", json=long).json()
    assert len(seen[0]["context"]) <= slm_proxy.MAX_CONTEXT_CHARS
    assert body["truncated"] is True


def test_build_context_without_a_corpus_or_report():
    out = slm_context.build_context("q", 4000, passages=[])
    assert out == {"context": "", "sources": [], "truncated": False}


def test_build_context_labels_every_part_and_respects_the_limit():
    out = slm_context.build_context("q", 600, report_text="Report sentence.", passages=PASSAGES)
    assert len(out["context"]) <= 600
    assert out["context"].startswith("[report: report text]\nReport sentence.")
    assert all(s["kind"] in ("report", "corpus") for s in out["sources"])


def test_demo_mode_needs_no_key_and_no_gemini(monkeypatch):
    import inspect

    monkeypatch.setenv("CLIMATE_SUITE_MODE", "demo")
    seen = []
    fake_service(monkeypatch, seen)
    assert client.post("/slm/ask", json={"question": "q?", "report_id": "drought_one_month"}).status_code == 200
    source = inspect.getsource(slm_context) + inspect.getsource(slm_proxy)
    assert "get_chat_model" not in source and "invoke_with_backoff" not in source and "embed" not in source.lower().replace("embedding", "")


def test_nothing_is_logged(monkeypatch, caplog):
    fake_service(monkeypatch, [])
    with caplog.at_level("DEBUG"):
        client.post("/slm/ask", json={"question": "SECRET-QUESTION?", "report_text": "SECRET-REPORT"})
    assert "SECRET" not in caplog.text


def test_saved_example_files_exist_for_the_test():
    assert demo.load_example("drought_one_month") is not None


def test_corpus_passages_unwraps_the_bm25_hit_shape(monkeypatch):
    """The bug this test exists for: bm25_search returns {id, bm25_score, rank, chunk}, with the passage under "chunk";
    reading "text" at the top level made the corpus part of the context always empty."""
    from retrieval import lexical

    hits = [{"id": "a", "bm25_score": 3.2, "rank": 1, "chunk": {"id": "a", "source": "IMD FAQ", "text": "A heat wave is ..."}},
            {"id": "b", "bm25_score": 1.1, "rank": 2, "chunk": {"id": "b", "source": "Empty", "text": "  "}}]
    monkeypatch.setattr(lexical, "bm25_search", lambda q, n=4: hits)
    out = REAL_CORPUS_PASSAGES("what is a heat wave?")
    assert [c["source"] for c in out] == ["IMD FAQ"]


def test_corpus_passages_is_empty_not_an_error_when_the_corpus_is_missing(monkeypatch):
    from retrieval import lexical

    def missing(q, n=4):
        raise FileNotFoundError("chunks.jsonl")

    monkeypatch.setattr(lexical, "bm25_search", missing)
    assert REAL_CORPUS_PASSAGES("q?") == []


@pytest.mark.skipif(not rconfig.CHUNKS_PATH.exists(), reason="the corpus (retrieval/chunks.jsonl) is not built here")
def test_the_real_corpus_gives_passages_for_a_plain_question():
    out = REAL_CORPUS_PASSAGES("What is a heat wave?")
    assert out and all(c["text"].strip() for c in out)


@pytest.mark.parametrize("mode", ["demo", "live"])
def test_the_saved_reports_are_listed_in_every_mode(monkeypatch, mode):
    monkeypatch.setenv("CLIMATE_SUITE_MODE", mode)
    body = client.get("/slm/reports").json()
    assert [r["id"] for r in body["reports"]] == [item["id"] for item in demo.load_index()]
    assert all(r["label"] for r in body["reports"])
