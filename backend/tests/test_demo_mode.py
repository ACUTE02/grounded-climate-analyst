"""Demo mode: saved examples, no key, no network, no Gemini call.

CLIMATE_SUITE_MODE=demo is the default. In demo mode the API answers only the saved
examples in backend/demo/ and refuses everything else with a plain 409.
"""

import json
import re
import socket

import pytest
from fastapi.testclient import TestClient

from api import app as api_app
from api import demo, quota
from api.schemas import ReportRequest, ReportResponse
from evaluation.checker_eval import CASES
from orchestrator.grounding import check_grounding
from retrieval import config as rconfig

client = TestClient(api_app.app)

KEY_VARS = (*rconfig.API_KEY_ENV_VARS, "OPENAI_API_KEY")


@pytest.fixture
def demo_mode(monkeypatch, tmp_path):
    """Demo mode with no key anywhere, no .env file and every connection refused."""
    monkeypatch.delenv(demo.MODE_ENV, raising=False)
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)   # no backend/.env is read

    loopback = {"127.0.0.1", "::1", "localhost"}   # the test client's own event loop
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def guarded_connect(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and address[0] in loopback:
            return real_connect(sock, address, *args, **kwargs)
        raise AssertionError(f"demo mode tried to connect to {address!r}")

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host in loopback or host is None:
            return real_getaddrinfo(host, *args, **kwargs)
        raise AssertionError(f"demo mode tried to resolve {host!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    def no_graph(*args, **kwargs):
        raise AssertionError("demo mode ran the live pipeline")

    monkeypatch.setattr("orchestrator.graph.analyse", no_graph)


def _index():
    return json.loads(demo.INDEX_PATH.read_text(encoding="utf-8"))["examples"]


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #
def test_demo_is_the_default_mode(demo_mode):
    assert demo.mode() == "demo"
    assert demo.mode_problem() is None


@pytest.mark.parametrize("value, expected", [("live", "live"), (" LIVE ", "live"),
                                             ("demo", "demo")])
def test_the_variable_selects_the_mode(demo_mode, monkeypatch, value, expected):
    monkeypatch.setenv(demo.MODE_ENV, value)
    assert demo.mode() == expected


def test_an_unknown_value_falls_back_to_demo_and_says_so(demo_mode, monkeypatch):
    monkeypatch.setenv(demo.MODE_ENV, "livee")
    assert demo.mode() == "demo"
    assert "livee" in demo.mode_problem()
    note = client.get("/health").json()["note"]
    assert "livee" in note


def test_the_mode_is_read_at_request_time(demo_mode, monkeypatch):
    assert client.get("/health").json()["mode"] == "demo"
    monkeypatch.setenv(demo.MODE_ENV, "live")
    assert client.get("/health").json()["mode"] == "live"


# --------------------------------------------------------------------------- #
# The saved files
# --------------------------------------------------------------------------- #
def test_the_index_lists_the_six_examples():
    ids = [item["id"] for item in _index()]
    assert ids == ["drought_one_month", "drought_no_skill_horizon", "heat_null_result",
                   "crop_impact_wheat", "grounding_catch", "out_of_scope"]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("item", _index(), ids=lambda i: i["id"])
def test_every_saved_file_is_a_valid_response_with_a_truthful_meta(item):
    ReportRequest(**item["body"])
    data = json.loads((demo.REPORTS_DIR / item["file"]).read_text(encoding="utf-8"))
    response = ReportResponse(**data)
    meta = response.meta
    assert meta is not None
    assert meta.mode == "demo" and meta.saved_example is True
    assert meta.example_id == item["id"]
    assert meta.how_made == item["kind"]
    assert meta.report_generated == bool(response.report.strip()) == item["has_report"]
    assert meta.generated_at and meta.banner
    assert response.request == item["body"]["request"]
    assert response.quota == {}, "the live quota must not be baked into a saved file"
    # an answer with no report must say so in the grounding field too
    if not response.report.strip():
        assert response.grounding.status == "not_generated"
        assert response.grounding.report_missing is True
    # the derived fields agree with the tool outputs they came from
    state = {"tool_outputs": response.tool_outputs}
    assert [h.model_dump() for h in response.horizon_confidence] == \
        api_app._horizon_confidence(state)


def test_drought_examples_keep_the_labels_distinct():
    for example_id in ("drought_one_month", "drought_no_skill_horizon"):
        data = json.loads((demo.REPORTS_DIR / f"{example_id}.json").read_text(
            encoding="utf-8"))
        labels = [h["label"] for h in data["horizon_confidence"]]
        assert labels[0] == "validated" and labels[1] == "weak/directional"
        assert labels[2].startswith("no skill")
        assert [h["reliable"] for h in data["horizon_confidence"]] == [True, False, False]
        assert any(m["what"].startswith("reliable drought forecast")
                   for m in data["missing_data"])


def test_heat_example_states_that_there_is_no_forecast():
    data = json.loads((demo.REPORTS_DIR / "heat_null_result.json").read_text(
        encoding="utf-8"))
    assert data["tool_outputs"]["forecast_heat_stress_risk"]["forecast_available"] is False
    assert any(m["what"] == "heat stress forecast" for m in data["missing_data"])


def test_grounding_catch_is_the_real_checker_verdict_on_the_fixture():
    data = json.loads((demo.REPORTS_DIR / "grounding_catch.json").read_text(
        encoding="utf-8"))
    case = next(c for c in CASES if c.id == "fabricated_skill_score")
    verdict = check_grounding(case.report, case.tool_outputs, case.chunks or [])
    assert data["grounding"]["status"] == "warning"
    assert data["grounding"]["unverified_numbers"] == verdict["unverified_numbers"] \
        == list(case.fabricated)
    assert data["grounding"]["total_checked"] == verdict["total_checked"]
    assert case.report in data["report"]
    assert data["meta"]["how_made"] == "constructed_fixture"
    assert "made wrong on purpose" in data["meta"]["banner"]
    assert data["meta"]["details"]["related_real_case"]["attempts"]


def test_out_of_scope_refusals_come_from_the_real_validators():
    from crop_impact import config as cconfig
    from forecasting import config as fconfig

    data = json.loads((demo.REPORTS_DIR / "out_of_scope.json").read_text(
        encoding="utf-8"))
    reasons = " | ".join(m["reason"] for m in data["missing_data"])
    with pytest.raises(ValueError) as region_error:
        fconfig.check_region("jaisalmer")
    with pytest.raises(ValueError) as crop_error:
        cconfig.check_crop("mustard")
    assert str(region_error.value) in reasons and str(crop_error.value) in reasons
    assert len(data["missing_data"]) == 3


# The names are written in pieces on purpose, so that scanning the source tree for
# these words does not match this test file itself.
_NAMES = ("cl" + "aude", "anthr" + "opic", "co-auth" + "ored", "PROJECT" + "_LOG",
          "EVALUATION" + r"\.md", "internal" + "_docs", "project" + "_log")
PRIVATE = re.compile("|".join(_NAMES) + r"|[\w.+-]+@[\w-]+\.[\w.]+|\+" + "91|"
                     r"AI" + r"za[0-9A-Za-z_-]{20,}", re.I)


def test_saved_files_carry_no_private_names_emails_phones_or_keys():
    for path in demo.DEMO_DIR.rglob("*"):
        if path.is_file():
            assert not PRIVATE.search(path.read_text(encoding="utf-8")), path.name


# --------------------------------------------------------------------------- #
# Endpoints in demo mode
# --------------------------------------------------------------------------- #
def test_examples_in_demo_mode_are_the_saved_ones(demo_mode):
    body = client.get("/examples").json()
    assert body["mode"] == "demo"
    assert [e["id"] for e in body["examples"]] == [i["id"] for i in _index()]
    assert all(e["saved"] is True for e in body["examples"])
    assert "no Gemini calls" in body["cost_warning"]
    assert body["quota"]["calls_used_today"] == 0


def test_a_saved_example_is_served_whole(demo_mode):
    for item in _index():
        response = client.get(f"/examples/{item['id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["mode"] == "demo" and body["meta"]["saved_example"] is True
        assert body["request"] == item["body"]["request"]
        assert body["quota"]["note"].startswith("Demo mode makes no Gemini calls")


def test_an_unknown_example_is_a_404(demo_mode):
    assert client.get("/examples/nothing_here").status_code == 404


def test_report_returns_the_saved_example_for_its_own_request(demo_mode):
    item = _index()[0]
    response = client.post("/report", json=item["body"])
    assert response.status_code == 200
    assert response.json()["meta"]["example_id"] == item["id"]


def test_matching_ignores_case_and_spacing_but_not_fields(demo_mode):
    item = _index()[0]
    shouted = {**item["body"], "request": "  " + item["body"]["request"].upper() + " "}
    assert client.post("/report", json=shouted).status_code == 200
    other_region = {**item["body"], "region": "barmer"}
    assert client.post("/report", json=other_region).status_code == 409


def test_free_text_is_refused_with_a_plain_message(demo_mode):
    response = client.post("/report", json={"request": "Will it rain in Delhi?"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "demo_free_text_disabled"
    assert detail["message"] == demo.FREE_TEXT_MESSAGE
    assert "own Gemini key" in detail["message"]
    assert detail["examples"] == [i["id"] for i in _index()]


def test_health_in_demo_mode_is_ok_without_the_index_models_or_key(demo_mode):
    body = client.get("/health").json()
    assert body["mode"] == "demo"
    assert body["status"] == "ok"
    assert body["api_key_present"] is False
    assert body["quota"]["calls_used_today"] == 0


def test_quota_in_demo_mode_uses_nothing(demo_mode):
    body = client.get("/quota").json()
    assert body == quota.demo_status()
    assert set(body) == set(quota.status().keys())


def test_demo_mode_makes_no_gemini_call_and_opens_no_connection(demo_mode):
    """Dead network, no key, graph trapped: every demo path still answers, and the
    Gemini call tally does not move."""
    from orchestrator.graph import CALL_TALLY

    before = dict(CALL_TALLY)
    item = _index()[1]
    assert client.get("/health").status_code == 200
    assert client.get("/examples").status_code == 200
    assert client.get(f"/examples/{item['id']}").status_code == 200
    assert client.post("/report", json=item["body"]).status_code == 200
    assert client.post("/report", json={"request": "anything else"}).status_code == 409
    assert client.get("/quota").status_code == 200
    assert dict(CALL_TALLY) == before


# --------------------------------------------------------------------------- #
# Live mode keeps working and says what it is
# --------------------------------------------------------------------------- #
def test_live_answers_carry_a_live_meta(monkeypatch, tmp_path):
    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)
    state = {"report": "text", "grounding": {"grounded": True, "total_checked": 0},
             "tool_outputs": {}, "tool_calls": []}
    meta = api_app.build_response(state, "x")["meta"]
    assert meta["mode"] == "live" and meta["saved_example"] is False
    assert meta["how_made"] == "live_generation" and meta["report_generated"] is True


def test_live_start_problems_name_what_is_missing(monkeypatch, tmp_path):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rconfig, "CHUNKS_PATH", tmp_path / "chunks.jsonl")
    problems = " ".join(demo.live_start_problems())
    assert "no Gemini key" in problems
    assert "retrieval.build" in problems


def test_the_network_guard_itself_works(demo_mode):
    """Positive control: an outside connection from inside these tests would fail."""
    with pytest.raises(AssertionError):
        socket.create_connection(("example.com", 80), timeout=1)


@pytest.mark.parametrize("item", _index(), ids=lambda i: i["id"])
def test_the_evidence_note_appears_only_when_the_data_was_refetched(item):
    data = json.loads((demo.REPORTS_DIR / item["file"]).read_text(encoding="utf-8"))
    meta = data["meta"]
    assert meta["data_source"] in ("snapshot", "refetched", None)
    if meta["data_source"] == "refetched":
        assert demo.EVIDENCE_NOTE in meta["banner"]
    else:
        assert demo.EVIDENCE_NOTE not in (meta["banner"] or "")
    assert "Evidence page" in demo.EVIDENCE_NOTE


def test_meta_carries_the_note_only_for_refetched_data(monkeypatch, tmp_path):
    from forecasting import snapshot

    monkeypatch.setattr(rconfig, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot, "data_source", lambda: "refetched")
    meta = demo.make_meta(how_made="live_generation", tool_outputs={}, report="x",
                          mode_="live", used_models=True)
    assert meta["banner"] == demo.EVIDENCE_NOTE and meta["data_source"] == "refetched"
    monkeypatch.setattr(snapshot, "data_source", lambda: "snapshot")
    meta = demo.make_meta(how_made="live_generation", tool_outputs={}, report="x",
                          mode_="live", used_models=True)
    assert meta["banner"] is None and meta["data_source"] == "snapshot"


@pytest.mark.parametrize("example_id, region", [("drought_one_month", "rajasthan"),
                                                ("drought_no_skill_horizon", "barmer")])
def test_saved_drought_numbers_equal_the_published_frozen_manifest(example_id, region):
    """Built from the data snapshot, the generated numbers equal the frozen evidence."""
    from forecasting import config as fconfig

    frozen = json.loads(fconfig.HORIZON_MANIFEST_PATH.read_text(encoding="utf-8"))
    data = json.loads((demo.REPORTS_DIR / f"{example_id}.json").read_text(encoding="utf-8"))
    saved = [(h["horizon"], h["skill_score"], h["label"], h["method"])
             for h in data["horizon_confidence"]]
    published = [(h["horizon"], h["skill_score"], h["label"], h["method"])
                 for h in frozen["regions"][region]]
    assert saved == published
