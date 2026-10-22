"""The optional small-language-model passage check: default off, fake SLM, cannot change the report.

Nothing here talks to a real SLM or to Gemini. The SLM service is faked with httpx.MockTransport.
"""

import json

import httpx
import pytest

from api import app as api_app
from api.schemas import ReportResponse
from orchestrator import config, graph, slm

CHUNKS = [
    {"source": "Doc A", "citation": "https://example.org/a", "text": "Passage A says the plains threshold is 40 C."},
    {"source": "Doc B", "citation": "https://example.org/b", "text": "Passage B is about something else."},
    {"source": "Doc C", "citation": "https://example.org/c", "text": "Passage C is never asked about."},
]
ON = {config.SLM_FLAG_ENV: "on", config.SLM_URL_ENV: "http://slm.test:8001"}


def fake_service(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_handler(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), body))
        refused = "something else" in body["context"]
        text = "I cannot answer this from the provided context." if refused else "40 C in the plains."
        return httpx.Response(200, json={"answer": text, "refused": refused, "model": "fake-slm", "latency_ms": 12.5,
                                         "n_prompt_tokens": 77, "n_new_tokens": 5, "request_id": "r"})
    return handler


# --------------------------------------------------------------------------- #
# The switch: default off
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value, expected", [(None, False), ("off", False), ("", False), ("maybe", False),
                                             ("on", True), (" ON ", True), ("1", True), ("true", True)])
def test_the_step_is_off_unless_asked(value, expected):
    env = {} if value is None else {config.SLM_FLAG_ENV: value}
    assert slm.enabled(env) is expected


def test_off_adds_nothing_and_calls_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the SLM was asked although the step is off")

    monkeypatch.setattr(slm, "enabled", lambda env=None: False)
    monkeypatch.setattr(slm, "passage_answers", boom)
    state = {"request": "q", "retrieved_chunks": CHUNKS, "warnings": ["w"]}
    assert graph.slm_passage(state) == {}, "no key is added to the state when the step is off"


def test_the_node_sits_between_type_c_and_synthesis():
    edges = {(e.source, e.target) for e in graph.build_graph().get_graph().edges}
    assert ("fetch_type_c", "slm_passage") in edges and ("slm_passage", "synthesize") in edges
    assert ("fetch_type_c", "synthesize") not in edges


# --------------------------------------------------------------------------- #
# Asking the SLM
# --------------------------------------------------------------------------- #
def test_it_asks_about_the_first_passages_only_and_records_answers_and_refusals():
    calls = []
    result = slm.passage_answers("At what temperature is a heat wave declared?", CHUNKS, env=ON,
                                 client=fake_service(ok_handler(calls)))
    assert len(calls) == config.SLM_MAX_PASSAGES == 2
    assert calls[0][0] == "http://slm.test:8001/answer"
    assert calls[0][1] == {"context": CHUNKS[0]["text"], "question": "At what temperature is a heat wave declared?",
                           "max_new_tokens": config.SLM_MAX_NEW_TOKENS}
    assert result["enabled"] is True and result["model"] == "fake-slm" and result["skipped_reason"] is None
    a, b = result["answers"]
    assert (a["source"], a["answer"], a["refused"]) == ("Doc A", "40 C in the plains.", False)
    assert (b["source"], b["refused"]) == ("Doc B", True)
    assert a["latency_ms"] == 12.5 and a["n_prompt_tokens"] == 77
    assert "not used to write the report" in result["note"]


def test_a_missing_url_or_missing_passages_is_a_clear_skip_not_an_error():
    assert slm.passage_answers("q", CHUNKS, env={config.SLM_FLAG_ENV: "on"})["skipped_reason"] == "SLM_URL is not set"
    result = slm.passage_answers("q", [{"text": "  "}], env=ON, client=fake_service(ok_handler([])))
    assert result["skipped_reason"] == "no retrieved passage to ask about" and result["answers"] == []


def test_a_too_long_passage_is_recorded_and_the_next_one_is_still_asked():
    def handler(request):
        body = json.loads(request.content)
        if "Passage A" in body["context"]:
            return httpx.Response(413, json={"detail": "prompt has 2000 tokens"})
        return httpx.Response(200, json={"answer": "x", "refused": False, "model": "fake-slm"})

    result = slm.passage_answers("q", CHUNKS, env=ON, client=fake_service(handler))
    assert result["answers"][0]["error"] == "this passage is too long for the SLM" and result["answers"][0]["answer"] is None
    assert result["answers"][1]["answer"] == "x" and result["skipped_reason"] is None


@pytest.mark.parametrize("handler, reason", [
    (lambda r: httpx.Response(503, json={"detail": "backend error"}), "the SLM service answered HTTP 503"),
    (lambda r: httpx.Response(422, json={"detail": "bad"}), "the SLM service answered HTTP 422"),
    (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")), "the SLM service did not answer (ConnectError)"),
    (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), "the SLM service did not answer (ReadTimeout)"),
])
def test_a_failing_service_stops_the_step_without_raising(handler, reason):
    result = slm.passage_answers("q", CHUNKS, env=ON, client=fake_service(handler))
    assert result["skipped_reason"] == reason and result["answers"] == []


def test_the_total_time_limit_stops_the_step():
    ticks = iter([0, 0, config.SLM_TOTAL_DEADLINE_S + 1, config.SLM_TOTAL_DEADLINE_S + 2])
    calls = []
    result = slm.passage_answers("q", CHUNKS, env=ON, client=fake_service(ok_handler(calls)), clock=lambda: next(ticks))
    assert len(calls) == 1 and result["skipped_reason"] == "the total time limit was reached"


# --------------------------------------------------------------------------- #
# In the graph and the API: separate, cannot change the report
# --------------------------------------------------------------------------- #
def test_the_node_records_the_result_and_a_warning_when_skipped(monkeypatch):
    monkeypatch.setattr(slm, "enabled", lambda env=None: True)
    monkeypatch.setattr(slm, "passage_answers",
                        lambda q, c, **k: {"enabled": True, "answers": [], "skipped_reason": "SLM_URL is not set",
                                           "model": None, "note": slm.NOTE})
    out = graph.slm_passage({"request": "q", "retrieved_chunks": CHUNKS, "warnings": ["earlier"]})
    assert out["slm"]["skipped_reason"] == "SLM_URL is not set"
    assert out["warnings"] == ["earlier", "SLM passage check skipped: SLM_URL is not set"]


def test_the_synthesis_prompt_is_identical_with_and_without_the_slm_result(monkeypatch):
    seen = []

    class FakeModel:
        pass

    def fake_invoke(model, messages, attempts=3):
        seen.append(messages)

        class R:
            content = "A report."
        return R()

    monkeypatch.setattr(graph, "get_chat_model", lambda tools=False: FakeModel())
    monkeypatch.setattr(graph, "invoke_with_backoff", fake_invoke)
    base = {"request": "q", "tool_outputs": {"t": {"x": 1}}, "retrieved_chunks": CHUNKS, "attempts": 0, "warnings": []}
    graph.synthesize(dict(base))
    graph.synthesize({**base, "slm": {"enabled": True, "answers": [{"answer": "SECRET-SLM-TEXT 0.42"}]}})
    assert seen[0] == seen[1]
    assert "SECRET-SLM-TEXT" not in json.dumps(seen[1])


def test_an_slm_answer_is_not_a_source_for_the_number_checker():
    state = {"report": "The threshold is 0.42.", "tool_outputs": {}, "retrieved_chunks": [],
             "type_c": {}, "external": {}, "request": "q", "attempts": 1,
             "slm": {"answers": [{"answer": "The threshold is 0.42."}]}}
    result = graph.verify_grounding(state)["grounding"]
    assert result["grounded"] is False and "0.42" in result["unverified_numbers"]


def test_the_response_carries_the_slm_field_only_when_the_step_ran():
    state = {"report": "text", "grounding": {"grounded": True, "total_checked": 0}, "tool_outputs": {}, "tool_calls": []}
    assert api_app.build_response(state, "x")["slm"] is None
    ran = {"enabled": True, "model": "fake-slm", "answers": [], "skipped_reason": None, "note": slm.NOTE}
    body = api_app.build_response({**state, "slm": ran}, "x")
    assert body["slm"] == ran
    assert ReportResponse(**body).slm["model"] == "fake-slm"


def test_saved_demo_files_have_no_slm_field_and_stay_valid():
    from api import demo

    for item in demo.load_index():
        saved = demo.load_example(item["id"])
        assert ReportResponse(**saved).slm is None


def test_the_step_costs_no_gemini_quota():
    import inspect

    source = inspect.getsource(slm)
    assert "invoke_with_backoff" not in source and "get_chat_model" not in source and "CALL_TALLY" not in source
