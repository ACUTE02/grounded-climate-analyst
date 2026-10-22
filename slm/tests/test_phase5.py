"""Tests for Phase 5, stage 1: the shared prompt function, the FastAPI service with a fake backend, the backend interface, the
vLLM smoke script helpers, cell P5-0, bundle. CPU only. No torch, transformers, peft, vllm."""
import json
import logging
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vllm_smoke
from climate_slm import chat
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase5_config, system_prompt
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import metrics as M
from climate_slm.evaluation.evaluate import decoding_params
from climate_slm.evaluation.scoring import load_items
from serving import app as appmod
from serving import backends
from serving.backends import FakeBackend, HFServingBackend, make_backend
from serving.prompting import build_messages
from serving.schemas import MAX_NEW_TOKENS, AnswerRequest
from _release import needs_training_data

REAL5 = load_phase5_config()


def with_test_models(cfg: dict) -> dict:
    """The published config has one model, loaded by its Hugging Face id. Tests that run several models or a local adapter
    directory add two fake entries that exist only here (no real adapter, no real path)."""
    models = dict(cfg["models"])
    models["sft_0.5b"] = {"label": "0.5B SFT", "adapter": "outputs/test-adapters/a", "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                          "parity_eval": "test_a"}
    models["dpo2_0.5b"] = {"label": "0.5B DPO v2", "adapter": "outputs/test-adapters/b", "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                           "parity_eval": "test_b"}
    return {**cfg, "models": models, "parity": {**cfg["parity"], "models": ["sft_0.5b", "dpo2_0.5b", "sft_1.5b"]},
            "bench": {**cfg["bench"], "models": ["sft_0.5b", "sft_1.5b", "dpo2_0.5b"]}}


CFG5, CFG1, ECFG = with_test_models(REAL5), load_config(), load_eval_config()
REF = CFG1["refusal_text"]
SRC = REPO_ROOT / "serving"


def _client(backend=None, **kw):
    backend = backend or FakeBackend("sft_0.5b")
    return TestClient(appmod.create_app(backend, CFG5, CFG1), **kw), backend


def _post(client, **body):
    base = {"context": "Water is wet. Fire is hot.", "question": "What is water?"}
    return client.post("/answer", json={**base, **body})


# ---- prompt building: identical to the data ---------------------------------------------------------------------------------
@needs_training_data
def test_build_messages_equals_every_val_and_train_row():
    for split in ("val", "train"):
        for it in load_items(REPO_ROOT / ECFG["paths"]["splits"][split]):
            context, question, _ = parse_example(it)
            assert build_messages(context, question, CFG1) == it["prompt"], it["meta"]["id"]


def test_build_messages_shape_and_exactness():
    m = build_messages("  ctx with spaces \n", "Q?")
    assert [x["role"] for x in m] == ["system", "user"] and m[0]["content"] == system_prompt(CFG1)
    assert m[1]["content"] == "Context:\n  ctx with spaces \n\n\nQuestion: Q?"            # nothing stripped, nothing cut
    assert REF in m[0]["content"]


def test_chat_template_goes_through_the_shared_helper():
    b = FakeBackend()
    m = build_messages("a b", "c d")
    text = chat.prompt_text(backends.qwen_style_template, m)
    assert text.endswith("<|im_start|>assistant\n") and b.count_prompt_tokens(m) == len(text.split())


# ---- config ---------------------------------------------------------------------------------------------------------------------
def test_config_lists_the_published_model_and_the_limits():
    mods = REAL5["models"]
    assert set(mods) == {"sft_1.5b"}, "only the 1.5B SFT adapter is published"
    entry = mods["sft_1.5b"]
    assert entry["adapter_id"] == "GHOSTBHAIYA/climate-slm-1.5b-sft" and entry["adapter_revision"] is None
    assert entry["base_model_id"] == "Qwen/Qwen2.5-1.5B-Instruct" and entry["lora_rank"] == 16 and entry["lora_alpha"] == 32
    assert "adapter" not in entry, "no local adapter path in the published config"
    assert entry["parity_eval"] == "q15_lora_fp16_e3"
    text = (REPO_ROOT / "configs" / "phase5.yaml").read_text(encoding="utf-8")
    assert "outputs/adapters" not in text and "sft_0.5b" not in text and "dpo2_0.5b" not in text
    s = CFG5["service"]
    assert s["max_new_tokens"] == MAX_NEW_TOKENS == 256 and s["max_model_len"] == 2048 and s["max_prompt_tokens"] == 1792
    assert s["cors"] is False and s["host"] == "127.0.0.1" and CFG5["default_model"] in mods
    assert CFG5["vllm"]["dtype"] == "half" and CFG5["vllm"]["max_lora_rank"] == 16
    assert CFG5["mlflow"]["experiment_name"] == "phase5_serving" and CFG5["bench"]["concurrency"] == [1, 4, 16] and CFG5["bench"]["repeats"] == 3
    assert set(REAL5["bench"]["models"]) == set(mods)
    appmod.check_service_config(CFG5)


def test_saved_parity_files_exist_when_outputs_are_here():
    for k, v in CFG5["models"].items():
        f = REPO_ROOT / "outputs" / f"eval_{v['parity_eval']}_val.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            assert d["split"] == "val" and d["summary"]["n_items"] == 53, k


def test_decoding_is_the_evaluators_greedy_mode():
    d = decoding_params(ECFG, "greedy")
    assert d["do_sample"] is False and d["repetition_penalty"] == 1.0 and d["max_new_tokens"] == CFG5["service"]["max_new_tokens"]
    # the service config does not carry its own copy of the prompt, the refusal text or the decoding settings
    keys = set()
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                keys.add(k)
                walk(v)
    walk(CFG5)
    assert not keys & {"system_prompt", "refusal_text", "repetition_penalty", "do_sample", "temperature"}


def test_config_checks_reject_a_wrong_service_block():
    import copy
    for key, value in (("max_new_tokens", 512), ("max_prompt_tokens", 2000), ("cors", True)):
        bad = copy.deepcopy(CFG5)
        bad["service"][key] = value
        with pytest.raises(ValueError):
            appmod.check_service_config(bad)


# ---- validation ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", [
    {"context": "", "question": "q"}, {"context": "   \n", "question": "q"}, {"context": "c", "question": ""},
    {"context": "c", "question": " \t"}, {"question": "q"}, {"context": "c"}, {}, {"context": 5, "question": "q"},
    {"context": "c", "question": "q", "extra": 1}, {"context": "c", "question": "q", "max_new_tokens": 0},
    {"context": "c", "question": "q", "max_new_tokens": 257}, {"context": "c", "question": "q", "max_new_tokens": -3},
    {"context": "c", "question": "q", "max_new_tokens": "many"}])
def test_bad_requests_give_422_and_never_reach_the_backend(body):
    c, b = _client()
    r = c.post("/answer", json=body)
    assert r.status_code == 422 and b.calls == []


def test_request_with_no_json_body_is_422():
    c, _ = _client()
    assert c.post("/answer", content=b"not json", headers={"content-type": "application/json"}).status_code == 422


def test_max_new_tokens_bounds_are_passed_to_the_backend():
    c, b = _client()
    assert _post(c).status_code == 200 and b.calls[-1][1] == 256                      # default
    for n in (1, 7, 256):
        assert _post(c, max_new_tokens=n).status_code == 200 and b.calls[-1][1] == n


def test_a_long_answer_is_cut_at_max_new_tokens_and_counted():
    b = FakeBackend("sft_0.5b", responder=lambda m: " ".join(["w"] * 50))
    c, _ = _client(b)
    r = _post(c, max_new_tokens=5).json()
    assert r["n_new_tokens"] == 5 and r["answer"] == "w w w w w"


def test_model_name_must_match_the_served_model():
    c, _ = _client()
    assert _post(c, model="sft_0.5b").status_code == 200
    r = _post(c, model="sft_1.5b")
    assert r.status_code == 422 and "sft_0.5b" in r.json()["detail"]
    assert _post(c, model=None).status_code == 200


def test_context_longer_than_the_token_limit_gives_413_with_the_numbers():
    c, b = _client()
    limit = CFG5["service"]["max_prompt_tokens"]
    base = b.count_prompt_tokens(build_messages("w", "q", CFG1))
    ok_ctx = " ".join(["w"] * (limit - base + 1))                      # exactly at the limit (the first "w" is counted in base)
    assert b.count_prompt_tokens(build_messages(ok_ctx, "q", CFG1)) == limit
    assert c.post("/answer", json={"context": ok_ctx, "question": "q"}).status_code == 200
    n_calls = len(b.calls)
    r = c.post("/answer", json={"context": ok_ctx + " w", "question": "q"})
    assert r.status_code == 413 and f"{limit + 1} tokens" in r.json()["detail"] and str(limit) in r.json()["detail"]
    assert "never cut silently" in r.json()["detail"] and len(b.calls) == n_calls                  # the backend did not see the prompt


# ---- the answer ------------------------------------------------------------------------------------------------------------------------
def test_refusal_flag_on_the_exact_refusal_text_and_on_a_normal_answer():
    texts = {REF: True, "It is blue.": False, "I cannot answer this from the provided context. But maybe...": False,
             REF.upper(): True, REF.rstrip("."): True}
    for text, expected in texts.items():
        c, _ = _client(FakeBackend("sft_0.5b", responder=lambda m, t=text: t))
        r = _post(c).json()
        assert r["refused"] is expected, text
        assert r["refused"] == M.is_refusal_strict(r["answer"], REF)                                  # the evaluator's detector, nothing else


def test_response_fields_and_request_id():
    c, b = _client(FakeBackend("sft_0.5b", responder=lambda m: "Water is wet today."))
    r = _post(c)
    j = r.json()
    assert set(j) == {"answer", "refused", "model", "n_prompt_tokens", "n_new_tokens", "latency_ms", "request_id"}
    assert j["answer"] == "Water is wet today." and j["refused"] is False and j["model"] == "sft_0.5b"
    assert j["n_new_tokens"] == 4 and j["n_prompt_tokens"] == b.count_prompt_tokens(build_messages("Water is wet. Fire is hot.", "What is water?", CFG1))
    assert isinstance(j["latency_ms"], float) and j["latency_ms"] >= 0
    assert j["request_id"] == r.headers["x-request-id"] and len(j["request_id"]) == 32
    assert _post(c).json()["request_id"] != j["request_id"]


def test_the_backend_gets_exactly_the_evaluator_messages():
    c, b = _client()
    _post(c, context="Ctx.", question="Qu?")
    assert b.calls[-1][0] == build_messages("Ctx.", "Qu?", CFG1) and b.calls[-1][0][0]["content"] == system_prompt(CFG1)


def test_fake_end_to_end_on_val_rows():
    items = load_items(REPO_ROOT / ECFG["paths"]["splits"]["val"])
    seen = {}

    def responder(messages):
        seen[messages[1]["content"]] = True
        return REF if "refuse" in messages[1]["content"].lower() else "ok"
    c, b = _client(FakeBackend("sft_0.5b", responder=responder))
    for it in items:
        context, question, _ = parse_example(it)
        r = c.post("/answer", json={"context": context, "question": question})
        assert r.status_code in (200, 413)
        if r.status_code == 200:
            assert b.calls[-1][0] == it["prompt"]
    assert len(b.calls) >= 40


# ---- errors, health, logging ------------------------------------------------------------------------------------------------------------
def test_backend_error_gives_503_with_a_short_message_and_no_trace(caplog):
    secret = RuntimeError("CUDA out of memory while handling SECRET-CONTEXT-WORDS")
    c, _ = _client(FakeBackend("sft_0.5b", fail_with=secret), raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger="serving"):
        r = _post(c, context="SECRET-CONTEXT-WORDS here", question="SECRET-QUESTION?")
    assert r.status_code == 503 and r.json() == {"detail": "backend error"}
    assert "Traceback" not in r.text and "CUDA" not in r.text and "x-request-id" in r.headers
    logs = caplog.text
    assert "RuntimeError" in logs and "SECRET" not in logs and "CUDA" not in logs


def test_token_counting_error_is_a_503_too():
    class Broken(FakeBackend):
        def count_prompt_tokens(self, messages):
            raise ValueError("tokenizer exploded")
    c, _ = _client(Broken("sft_0.5b"))
    r = _post(c)
    assert r.status_code == 503 and r.json() == {"detail": "backend error"}


def test_unhandled_error_is_a_short_500_without_a_stack_trace():
    class Weird(FakeBackend):
        @property
        def model_name(self):
            raise ZeroDivisionError("boom")

        @model_name.setter
        def model_name(self, v):
            pass
    c, _ = _client(Weird("x"), raise_server_exceptions=False)
    r = c.get("/health")
    assert r.status_code == 500 and r.json() == {"detail": "internal error"} and "Traceback" not in r.text


def test_not_loaded_backend_is_503_and_health_says_loading():
    c, _ = _client(FakeBackend("sft_0.5b", loaded=False))
    assert _post(c).status_code == 503
    h = c.get("/health").json()
    assert h["status"] == "loading" and h["model_loaded"] is False


def test_health_fields():
    c, _ = _client()
    r = c.get("/health")
    h = r.json()
    assert r.status_code == 200 and h["status"] == "ok" and h["model_loaded"] is True and h["backend"] == "fake" and h["model"] == "sft_0.5b"
    assert h["models"] == sorted(CFG5["models"]) and {"python", "fastapi", "pydantic", "torch", "transformers", "vllm"} <= set(h["versions"])
    assert h["versions"]["fastapi"] and h["versions"]["torch"] is None                           # the laptop has no torch
    assert "x-request-id" in r.headers


def test_logs_have_ids_and_counts_but_never_the_text(caplog):
    c, _ = _client(FakeBackend("sft_0.5b", responder=lambda m: "UNIQUE-ANSWER-TEXT"))
    with caplog.at_level(logging.INFO, logger="serving"):
        r = _post(c, context="UNIQUE-CONTEXT-TEXT is here", question="UNIQUE-QUESTION-TEXT?")
    j = r.json()
    logs = caplog.text
    assert j["request_id"] in logs and "model=sft_0.5b" in logs and "latency_ms=" in logs and "n_prompt_tokens=" in logs and "n_new_tokens=" in logs
    assert "UNIQUE" not in logs
    assert "UNIQUE" not in "".join(rec.getMessage() for rec in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="serving"):
        c.post("/answer", json={"context": "UNIQUE-CONTEXT-TEXT " * 5000, "question": "q"})                    # 413 path
    assert "UNIQUE" not in caplog.text


def test_no_cors_and_no_secrets():
    app = appmod.create_app(FakeBackend("sft_0.5b"), CFG5, CFG1)
    assert not any("CORS" in str(m.cls) for m in app.user_middleware)
    r = TestClient(app).options("/answer", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers
    for f in list(SRC.glob("*.py")) + [REPO_ROOT / "configs" / "phase5.yaml", REPO_ROOT / "scripts" / "vllm_smoke.py"]:
        low = f.read_text(encoding="utf-8").lower()
        for word in ("api_key", "apikey", "password", "secret_key", "hf_token", "bearer ", "authorization"):
            assert word not in low, (f.name, word)
    assert "allow_origins" not in "".join(f.read_text(encoding="utf-8") for f in SRC.glob("*.py"))


# ---- backend selection and the interface ----------------------------------------------------------------------------------------------------
def test_backend_and_model_selection_from_the_environment():
    assert appmod.choose_backend_name(CFG5, {}) == "hf" and appmod.choose_backend_name(CFG5, {"PHASE5_BACKEND": "fake"}) == "fake"
    assert appmod.choose_model_key(CFG5, {}) == "sft_1.5b" and appmod.choose_model_key(CFG5, {"PHASE5_MODEL": "sft_0.5b"}) == "sft_0.5b"
    assert set(backends.BACKEND_NAMES) == {"vllm", "hf", "fake"}


def test_make_backend():
    f = make_backend("fake", "sft_1.5b", CFG5)
    assert isinstance(f, FakeBackend) and f.model_name == "sft_1.5b" and f.info() == {"backend": "fake", "model": "sft_1.5b", "loaded": True}
    h = make_backend("hf", "dpo2_0.5b", CFG5, ECFG)
    assert isinstance(h, HFServingBackend) and not h.loaded and h.info()["base_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert h.adapter.replace("\\", "/").endswith("test-adapters/b")
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("tgi", "sft_0.5b", CFG5)
    with pytest.raises(ValueError, match="unknown model"):
        make_backend("fake", "nope", CFG5)
    v = make_backend("vllm", "sft_1.5b", CFG5)                                   # stage 2: the vllm backend exists (not loaded here)
    assert v.name == "vllm" and not v.loaded and v.base_model_id == "Qwen/Qwen2.5-1.5B-Instruct" and v.max_batch == 16
    assert h.max_batch == 8 and h.window_ms == CFG5["batching"]["window_ms"]


def test_app_built_from_the_environment_loads_the_backend_on_startup(monkeypatch):
    monkeypatch.setenv("PHASE5_BACKEND", "fake")
    monkeypatch.setenv("PHASE5_MODEL", "sft_1.5b")
    app = appmod.create_app(cfg5=CFG5, cfg1=CFG1)
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["model"] == "sft_1.5b" and h["backend"] == "fake" and h["model_loaded"] is True


def test_no_gpu_library_is_imported_at_module_level():
    for f in SRC.glob("*.py"):
        head = f.read_text(encoding="utf-8").split("def ")[0].split("class ")[0]
        for lib in ("torch", "transformers", "peft", "vllm", "bitsandbytes"):
            assert f"\nimport {lib}" not in head and f"\nfrom {lib}" not in head, (f.name, lib)
    assert not {"torch", "vllm", "transformers", "peft"} & set(sys.modules)


def test_schema_request_keeps_the_text_unchanged():
    r = AnswerRequest(context="  a  ", question=" b ")
    assert r.context == "  a  " and r.question == " b " and r.max_new_tokens is None and r.model is None


# ---- the vLLM smoke script (helpers only; the vLLM part runs on Colab) ------------------------------------------------------------------------
def test_smoke_prompts_are_the_first_val_rows_with_the_shared_function():
    rows = vllm_smoke.first_val_prompts(3, ECFG, CFG1)
    val = load_items(REPO_ROOT / ECFG["paths"]["splits"]["val"])[:3]
    assert [r["id"] for r in rows] == [v["meta"]["id"] for v in val] and [r["messages"] for r in rows] == [v["prompt"] for v in val]
    assert CFG5["smoke"]["n_prompts"] == 3 and CFG5["smoke"]["split"] == "val"


def test_compare_to_saved():
    rows = [{"id": "a", "kind": "answerable"}, {"id": "b", "kind": "refusal"}, {"id": "c", "kind": "refusal"}, {"id": "zz", "kind": "answerable"}]
    saved = [{"id": "a", "output": "Same."}, {"id": "b", "output": REF}, {"id": "c", "output": "Invented."}]
    c = vllm_smoke.compare_to_saved(rows, ["Same. ", REF, REF, "x"], saved, REF)
    assert c["n"] == 4 and c["identical"] == 2
    p = {x["id"]: x for x in c["per_item"]}
    assert p["a"]["identical"] and not p["a"]["refused_now"] and p["b"]["identical"] and p["b"]["refused_now"] and p["b"]["refused_saved"]
    assert not p["c"]["identical"] and p["c"]["refused_now"] and p["c"]["refused_saved"] is False
    assert p["zz"]["saved_found"] is False and p["zz"]["refused_saved"] is None


def test_smoke_step_reports_pass_and_fail(capsys):
    report = {"steps": []}
    assert vllm_smoke.step(report, "ok step", lambda: 5) == 5
    assert vllm_smoke.step(report, "bad step", lambda: 1 / 0) is None
    assert [s["status"] for s in report["steps"]] == ["PASS", "FAIL"] and "ZeroDivisionError" in report["steps"][1]["error"]
    out = capsys.readouterr().out
    assert "[PASS] ok step" in out and "[FAIL] bad step" in out


def test_smoke_finish_writes_the_result_file(capsys):
    out = "outputs/_smoke_check_tmp.json"
    p = REPO_ROOT / out
    try:
        assert vllm_smoke.finish({"steps": [{"step": "x", "status": "PASS"}]}, out, 0) == 0
        assert json.loads(p.read_text(encoding="utf-8"))["ok"] is True
        assert vllm_smoke.finish({"steps": [{"step": "x", "status": "FAIL"}]}, out, 0) == 1
        assert json.loads(p.read_text(encoding="utf-8"))["ok"] is False
        assert "hf backend" in capsys.readouterr().out
        assert vllm_smoke.finish({"steps": [{"step": "x", "status": "PASS"}]}, out, 1) == 1          # a non-zero code is never "ok"
    finally:
        p.unlink(missing_ok=True)


def test_smoke_script_structure():
    src = (REPO_ROOT / "scripts" / "vllm_smoke.py").read_text(encoding="utf-8")
    assert "sft_test" not in src and "\"test\"" not in src and "split test" not in src
    assert src.index("import vllm") < src.index("from vllm import LLM") and src.count("from vllm") == 2          # imported after the step, inside main
    assert "enable_lora" in src and "LoRARequest" in src and "temperature=0.0" in src and "repetition_penalty=1.0" in src
    assert "build_messages" in src and "chat.prompt_text" in src
    assert "hf backend" in src and "remove every sentence about vLLM" in src


# ---- notebook cell, bundle, requirements ---------------------------------------------------------------------------------------------------
def _cells():
    nb = json.loads((REPO_ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    return nb["cells"]


def test_notebook_p5_0_cell_starts_with_the_cd():
    code = ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]
    p50 = [c for c in code if "# P5-0." in c.splitlines()[:2][-1] or c.startswith("# P5-0.")]
    assert len(p50) == 1 and p50[0].splitlines()[0] == "%cd /content/climate_slm_project"            # the cwd is set before anything else
    c = p50[0]
    assert len(c.splitlines()) <= 15
    assert "pip install -q vllm" in c and "scripts/vllm_smoke.py" in c and "phase4_start.zip" in c
    assert "requirements-gpu.txt" not in c and "pip uninstall" not in c and "--upgrade" not in c         # the training environment is not touched
    assert "test" not in c.lower() and "pip install -q torch" not in c


def test_other_cells_are_unchanged_in_number_and_kind():
    code = ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]
    ids = {k: [c for c in code if c.lstrip().startswith(k)] for k in ("# P3-", "# P4-", "# P4b-")}
    assert len(ids["# P3-"]) == 11 and len(ids["# P4-"]) == 11 and len(ids["# P4b-"]) == 11


def test_bundle_has_the_service_the_config_and_the_smoke_script():
    assert {"path": "serving", "endings": [".py"]} in ECFG["bundle"]["folders"] and "scripts/vllm_smoke.py" in ECFG["bundle"]["files"]
    z = REPO_ROOT / ECFG["paths"]["colab_bundle"]
    if z.exists():
        names = zipfile.ZipFile(z).namelist()
        for n in ("serving/__init__.py", "serving/app.py", "serving/backends.py", "serving/schemas.py", "serving/prompting.py",
                  "configs/phase5.yaml", "scripts/vllm_smoke.py", "data/instruction/sft_val.jsonl"):
            assert n in names, n
        assert not any(x.startswith("outputs/") for x in names)


def test_requirements_serve_lists_laptop_test_needs_and_no_gpu_library():
    lines = [x.strip() for x in (REPO_ROOT / "requirements-serve.txt").read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    names = {x.split("==")[0].lower() for x in lines}
    assert {"fastapi", "uvicorn", "httpx", "pydantic", "pytest", "pyyaml"} <= names
    assert not names & {"torch", "vllm", "transformers", "peft", "bitsandbytes"}


def test_main_binds_to_the_configured_host_unless_the_environment_overrides_it(monkeypatch):
    import types
    seen = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, host, port, log_level: seen.update(host=host, port=port)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("PHASE5_BACKEND", "fake")
    monkeypatch.setenv("PHASE5_MODEL", "sft_1.5b")
    monkeypatch.delenv("PHASE5_HOST", raising=False)
    assert appmod.main() == 0 and seen["host"] == "127.0.0.1"
    monkeypatch.setenv("PHASE5_HOST", "0.0.0.0")
    assert appmod.main() == 0 and seen["host"] == "0.0.0.0"
