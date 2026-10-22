"""Tests for Phase 5, stage 2: micro-batching, the vllm and hf backends (with fake engines), the service client and launcher, the parity
script, the benchmark functions and script, the report and MLflow logging (fake mlflow), collect sets, notebook cells P5-1..P5-6.
CPU only. No torch, transformers, peft, vllm: their modules are replaced by small fakes where a backend needs them."""
import copy
import json
import socket
import sys
import threading
import time
import types
import zipfile
from concurrent.futures import Future
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_serving as B
import parity_check as P
import phase5_check as C5
import phase5_report as R
from climate_slm import chat
from climate_slm.config import REPO_ROOT, load_config, load_eval_config, load_phase5_config
from climate_slm.data.sft import parse_example
from climate_slm.evaluation import collect, report
from climate_slm.evaluation.scoring import load_items
from serving import app as appmod
from serving import backends, client as sc
from serving.backends import FakeBackend, VLLMServingBackend, make_backend
from serving.batching import MicroBatcher
from serving.prompting import build_messages

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
ROOT = REPO_ROOT


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- micro-batching -----------------------------------------------------------------------------------------------------------------
def test_batcher_returns_results_in_order_and_groups_requests_that_arrive_together():
    seen = []

    def run(batch):
        seen.append(list(batch))
        return [x * 10 for x in batch]
    b = MicroBatcher(run, max_batch=4, window_ms=200)
    futs = [b.submit(i) for i in range(6)]
    assert [f.result(timeout=5) for f in futs] == [0, 10, 20, 30, 40, 50]
    b.close()
    assert sum(len(x) for x in seen) == 6 and max(len(x) for x in seen) <= 4 and seen[0] == [0, 1, 2, 3]
    st = b.stats()
    assert st["n_batches"] == len(seen) and st["max_batch_size"] == 4 and st["mean_batch_size"] == pytest.approx(6 / len(seen))


def test_batcher_single_request_does_not_wait_longer_than_the_window():
    b = MicroBatcher(lambda batch: batch, max_batch=8, window_ms=20)
    t0 = time.perf_counter()
    assert b.submit("x").result(timeout=5) == "x"
    assert time.perf_counter() - t0 < 2.0
    b.close()


def test_batcher_error_reaches_every_waiting_request_and_the_worker_goes_on():
    calls = []

    def run(batch):
        calls.append(batch)
        if "bad" in batch:
            raise RuntimeError("engine failed")
        return batch
    b = MicroBatcher(run, max_batch=8, window_ms=100)
    f1, f2 = b.submit("bad"), b.submit("ok")
    for f in (f1, f2):
        with pytest.raises(RuntimeError, match="engine failed"):
            f.result(timeout=5)
    assert b.submit("fine").result(timeout=5) == "fine"                              # the worker survived
    b.close()


def test_batcher_per_item_exception_and_wrong_result_count():
    b = MicroBatcher(lambda batch: [ValueError("one bad") if x == 1 else x for x in batch], max_batch=4, window_ms=100)
    f0, f1 = b.submit(0), b.submit(1)
    assert f0.result(timeout=5) == 0
    with pytest.raises(ValueError, match="one bad"):
        f1.result(timeout=5)
    b.close()
    b2 = MicroBatcher(lambda batch: [], max_batch=4, window_ms=10)
    with pytest.raises(RuntimeError, match="returned 0 results"):
        b2.submit(1).result(timeout=5)
    b2.close()


def test_batcher_close_never_leaves_a_request_waiting():
    gate = threading.Event()

    def run(batch):
        gate.wait(5)
        return batch
    b = MicroBatcher(run, max_batch=1, window_ms=1)
    running = b.submit("a")                                                          # in the engine
    queued = b.submit("b")                                                           # queued before close: still answered
    time.sleep(0.05)
    stray = Future()
    b._q.put(None)                                                                   # the stop signal, then a request behind it:
    b._q.put(("late", stray))                                                        # what the old race in close() could leave
    threading.Timer(0.2, gate.set).start()
    assert running.result(timeout=5) == "a" and queued.result(timeout=5) == "b"
    b._thread.join(timeout=5)
    b.close()
    with pytest.raises(RuntimeError, match="closed before this request ran"):
        stray.result(timeout=5)                                                      # failed, not hanging
    with pytest.raises(RuntimeError, match="closed"):
        b.submit("after")


def test_batcher_many_threads_and_close():
    b = MicroBatcher(lambda batch: [x + 1 for x in batch], max_batch=16, window_ms=5)
    out = []
    ts = [threading.Thread(target=lambda i=i: out.append(b.submit(i).result(timeout=10))) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(out) == list(range(1, 41))
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        b.submit(1)
    b.close()                                                                         # closing twice is fine
    with pytest.raises(ValueError):
        MicroBatcher(lambda x: x, 0, 5)


# ---- fake engines for the real backends -------------------------------------------------------------------------------------------------
class FakeTok:
    def __init__(self):
        self.in_use = 0
        self.max_in_use = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return backends.qwen_style_template(msgs, add_generation_prompt)

    def __call__(self, text, add_special_tokens=False):
        self.in_use += 1
        self.max_in_use = max(self.max_in_use, self.in_use)
        time.sleep(0.001)
        self.in_use -= 1
        return {"input_ids": text.split()}


def install_fake_vllm(monkeypatch, answer="Water is wet."):
    log = {"llm_args": None, "generate": []}
    tok = FakeTok()

    class SamplingParams:
        def __init__(self, temperature, max_tokens, repetition_penalty):
            self.temperature, self.max_tokens, self.repetition_penalty = temperature, max_tokens, repetition_penalty

    class LoRARequest:
        def __init__(self, name, lora_int_id, path):
            self.name, self.id, self.path = name, lora_int_id, path

    class Out:
        def __init__(self, prompt, text, n, reason):
            self.prompt_token_ids = prompt.split()
            self.outputs = [types.SimpleNamespace(text=text, token_ids=list(range(n)), finish_reason=reason)]

    class LLM:
        def __init__(self, **kw):
            log["llm_args"] = kw

        def get_tokenizer(self):
            return tok

        def generate(self, prompts, params, lora_request=None, use_tqdm=True):
            log["generate"].append({"prompts": prompts, "params": params, "lora": lora_request, "use_tqdm": use_tqdm})
            return [Out(p, f"  {answer}  ", 3, "length" if sp.max_tokens == 1 else "stop") for p, sp in zip(prompts, params)]

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = LoRARequest
    vllm.lora, lora.request = lora, req
    for n, m in (("vllm", vllm), ("vllm.lora", lora), ("vllm.lora.request", req)):
        monkeypatch.setitem(sys.modules, n, m)
    trf = types.ModuleType("transformers")
    trf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda model_id: tok)
    monkeypatch.setitem(sys.modules, "transformers", trf)
    return log, tok


def test_vllm_backend_load_arguments_and_greedy_generation(monkeypatch):
    log, _ = install_fake_vllm(monkeypatch)
    b = make_backend("vllm", "sft_0.5b", CFG5)
    assert not b.loaded
    b.load()
    a = log["llm_args"]
    assert a == {"model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "half", "max_model_len": 2048, "gpu_memory_utilization": 0.85,
                 "enable_lora": True, "max_lora_rank": 16}
    assert b.loaded
    m = build_messages("ctx", "q?", CFG1)
    g = b.generate(m, 77)
    c = log["generate"][0]
    assert c["prompts"] == [chat.prompt_text(backends.qwen_style_template, m)] and c["use_tqdm"] is False
    sp = c["params"][0]
    assert (sp.temperature, sp.max_tokens, sp.repetition_penalty) == (0.0, 77, 1.0)                  # greedy, evaluator settings
    assert c["lora"].name == "sft_0.5b" and c["lora"].id == 1 and c["lora"].path.replace("\\", "/").endswith("test-adapters/a")
    assert g.text == "Water is wet." and g.n_new_tokens == 3 and g.n_prompt_tokens == len(chat.prompt_text(backends.qwen_style_template, m).split())
    assert g.hit_max_new_tokens is False and b.generate(m, 1).hit_max_new_tokens is True            # finish_reason "length"
    assert b.count_prompt_tokens(m) == g.n_prompt_tokens
    b.close()


def test_vllm_backend_batches_concurrent_requests_into_one_engine_call(monkeypatch):
    log, _ = install_fake_vllm(monkeypatch)
    b = VLLMServingBackend("sft_0.5b", "adapter", "base", CFG5["vllm"], 2048, window_ms=300, max_batch=16)
    b.load()
    m = build_messages("ctx", "q?", CFG1)
    futs = [b._batcher.submit((m, n)) for n in (5, 6, 7, 8)]
    res = [f.result(timeout=5) for f in futs]
    assert len(res) == 4 and len(log["generate"]) == 1 and len(log["generate"][0]["prompts"]) == 4
    assert [sp.max_tokens for sp in log["generate"][0]["params"]] == [5, 6, 7, 8]                    # one sampling setting per request
    assert b.info()["max_batch"] == 16 and b.info()["n_batches"] == 1 and b.info()["mean_batch_size"] == 4
    b.close()


def test_token_counting_is_serialised_with_a_lock(monkeypatch):
    _, tok = install_fake_vllm(monkeypatch)
    b = make_backend("vllm", "sft_0.5b", CFG5)
    b.load()
    m = build_messages("a b c", "d?", CFG1)
    ts = [threading.Thread(target=b.count_prompt_tokens, args=(m,)) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert tok.max_in_use == 1                                                                         # never two threads in the tokenizer
    b.close()


def test_backend_not_loaded_raises_a_backend_error():
    b = make_backend("vllm", "sft_0.5b", CFG5)
    with pytest.raises(backends.BackendError):
        b.generate(build_messages("c", "q", CFG1), 5)


def test_hf_backend_uses_the_evaluator_backend_in_batches_grouped_by_max_new_tokens(monkeypatch):
    install_fake_vllm(monkeypatch)                                                                     # only for the fake tokenizer
    calls = []

    class FakeEval:
        def __init__(self, cfg, model_arg, decoding, seed, base_model_id):
            calls.append(("init", model_arg, decoding, seed, base_model_id))
            self.params = {"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0}
            self.tok = FakeTok()

        def _prompt_text(self, messages):
            return chat.prompt_text(backends.qwen_style_template, messages)

        def generate(self, prompts):
            calls.append(("generate", len(prompts), self.params["max_new_tokens"], self.params["repetition_penalty"]))
            return [{"text": f"answer {len(prompts)}", "n_new_tokens": 2, "hit_max_new_tokens": False} for _ in prompts]

    import climate_slm.evaluation.evaluate as ev
    monkeypatch.setattr(ev, "HFBackend", FakeEval)
    downloads = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda repo_id, revision=None: downloads.append((repo_id, revision)) or "/cache/climate-slm-adapter"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    b = make_backend("hf", "sft_1.5b", CFG5, ECFG)
    assert not b.loaded and b.max_batch == 8 and downloads == [], "building the backend downloads nothing"
    b.load()
    assert downloads == [("GHOSTBHAIYA/climate-slm-1.5b-sft", None)]                    # the adapter is fetched by its Hugging Face id at load
    assert b.loaded and calls[0][2:] == ("greedy", None, "Qwen/Qwen2.5-1.5B-Instruct") and calls[0][1] == "/cache/climate-slm-adapter"
    m = build_messages("ctx", "q", CFG1)
    b._batcher.window_s = 0.3
    futs = [b._batcher.submit((m, n)) for n in (256, 256, 32)]
    res = [f.result(timeout=5) for f in futs]
    gens = [c for c in calls if c[0] == "generate"]
    assert sorted((g[1], g[2]) for g in gens) == [(1, 32), (2, 256)] and all(g[3] == 1.0 for g in gens)    # grouped by max_new_tokens
    assert [r.text for r in res] == ["answer 2", "answer 2", "answer 1"] and res[0].n_prompt_tokens == len(b._backend._prompt_text(m).split())
    assert b._backend.params["max_new_tokens"] == 256                                                  # the evaluator settings are restored
    b.close()


def test_make_backend_real_backends_import_nothing_heavy_until_load():
    for name in ("vllm", "hf"):
        b = make_backend(name, "dpo2_0.5b", CFG5, ECFG)
        assert not b.loaded and b.adapter.replace("\\", "/").endswith("test-adapters/b")
        assert b.info()["backend"] == name and b.info()["max_batch"] == CFG5["batching"]["max_batch"][name]
    assert "vllm" not in sys.modules or isinstance(sys.modules["vllm"], types.ModuleType)


# ---- the service: start-up time and batching statistics in /health --------------------------------------------------------------------------------
def test_health_reports_startup_seconds_and_batching(monkeypatch):
    monkeypatch.setenv("PHASE5_BACKEND", "fake")
    monkeypatch.setenv("PHASE5_MODEL", "sft_0.5b")
    app = appmod.create_app(cfg5=CFG5, cfg1=CFG1)
    with TestClient(app) as c:
        h = c.get("/health").json()
    assert h["model_loaded"] is True and h["startup_seconds"] is not None and h["startup_seconds"] >= 0 and h["batching"] is None
    ready = TestClient(appmod.create_app(FakeBackend("sft_0.5b"), CFG5, CFG1)).get("/health").json()
    assert ready["startup_seconds"] is None                                                            # a backend handed in ready was not loaded here


def test_health_has_batch_stats_for_a_batched_backend(monkeypatch):
    install_fake_vllm(monkeypatch)
    b = VLLMServingBackend("sft_0.5b", "adapter", "base", CFG5["vllm"], 2048, window_ms=5, max_batch=4)
    b.load()
    c = TestClient(appmod.create_app(b, CFG5, CFG1))
    assert c.post("/answer", json={"context": "ctx", "question": "q?"}).status_code == 200
    h = c.get("/health").json()
    assert h["backend"] == "vllm" and h["batching"]["n_batches"] == 1 and h["batching"]["mean_batch_size"] == 1
    b.close()


# ---- client, GPU memory, launcher ------------------------------------------------------------------------------------------------------------
def test_service_client_answer_and_errors():
    http = TestClient(appmod.create_app(FakeBackend("sft_0.5b"), CFG5, CFG1))
    c = sc.ServiceClient(http)
    assert c.health()["model"] == "sft_0.5b"
    r = c.answer("ctx here", "q?", max_new_tokens=9, model="sft_0.5b")
    assert r["answer"] == "fake answer" and r["n_new_tokens"] == 2
    for kwargs, status in (({"context": " ", "question": "q"}, 422), ({"context": "c", "question": "q", "model": "other"}, 422)):
        with pytest.raises(sc.ServiceError) as e:
            c.answer(**kwargs)
        assert e.value.status == status
    with pytest.raises(sc.ServiceError) as e:
        c.answer(" ".join(["w"] * 5000), "q")
    assert e.value.status == 413 and "never cut silently" in e.value.detail


def test_parse_nvidia_smi_and_gpu_sampler():
    rows = sc.parse_nvidia_smi("Tesla T4, 15360, 1021\nTesla T4, 15360, 15000\n\nbad line\n")
    assert rows == [{"name": "Tesla T4", "total_mib": 15360, "used_mib": 1021}, {"name": "Tesla T4", "total_mib": 15360, "used_mib": 15000}]
    assert sc.parse_nvidia_smi("") == []
    values = iter([1000, 3000, 2000] + [1500] * 1000)
    s = sc.GpuSampler(0.01, lambda: [{"name": "x", "total_mib": 1, "used_mib": next(values)}]).start()
    time.sleep(0.15)
    out = s.stop()
    assert out["peak_used_mib"] == 3000 and out["min_used_mib"] == 1000 and out["n_samples"] >= 3
    none = sc.GpuSampler(0.01, lambda: None).start()
    time.sleep(0.05)
    assert none.stop() == {"peak_used_mib": None, "min_used_mib": None, "n_samples": 0}


def test_server_process_with_the_fake_backend_end_to_end(tmp_path):
    import httpx
    port = free_port()
    srv = sc.ServerProcess("fake", "sft_1.5b", CFG5, port, tmp_path / "server.log")
    assert srv.base_url == f"http://127.0.0.1:{port}" and "PHASE5_BACKEND=fake" in srv.command_text() and "serving.app" in srv.command_text()
    srv.start()
    try:
        with httpx.Client(base_url=srv.base_url, timeout=30) as http:
            health, wall = srv.wait_healthy(lambda: sc.ServiceClient(http, 5.0), 60, 0.2)
            assert health["backend"] == "fake" and health["model_loaded"] and health["startup_seconds"] is not None and wall > 0
            r = sc.ServiceClient(http).answer("Water is wet.", "What is water?")
            assert r["answer"] == "fake answer" and len(r["request_id"]) == 32
    finally:
        srv.stop(10)
    assert srv.proc.poll() is not None and (tmp_path / "server.log").exists()
    log = (tmp_path / "server.log").read_text(encoding="utf-8", errors="replace")
    assert "Water is wet" not in log                                                                   # the service log has no context text


def test_server_process_that_dies_is_reported_with_its_log(tmp_path):
    srv = sc.ServerProcess("fake", "sft_1.5b", CFG5, free_port(), tmp_path / "dead.log", python=sys.executable,
                           extra_env={"PHASE5_MODEL": "no_such_model"})
    srv.start()
    try:
        with pytest.raises(sc.ServiceError, match="exited with code"):
            srv.wait_healthy(lambda: sc.ServiceClient(types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(OSError("down"))), 1.0), 60, 0.2)
    finally:
        srv.stop(5)
    assert "unknown model" in srv.log_tail()


# ---- parity ----------------------------------------------------------------------------------------------------------------------------------------
SAVED = ROOT / "outputs" / "eval_lora_fp16_lr2e-4_e2_val.json"
needs_outputs = pytest.mark.skipif(not SAVED.exists(), reason="outputs/eval_lora_fp16_lr2e-4_e2_val.json is not on this machine")


def saved_responder(tweak=None):
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    val = load_items(ROOT / ECFG["paths"]["splits"]["val"])
    by_user = {it["prompt"][1]["content"]: next(s["output"] for s in saved["items"] if s["id"] == it["meta"]["id"]) for it in val}
    ids = {it["prompt"][1]["content"]: it["meta"]["id"] for it in val}

    def responder(messages):
        user = messages[1]["content"]
        out = by_user[user]
        return tweak(ids[user], out) if tweak else out
    return responder


def run_fake_parity(tmp_path, tweak=None, limit=None):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / SAVED.name).write_bytes(SAVED.read_bytes())
    # the fake backend replays a function of the saved outputs; the max_new_tokens limit is 256 words for the fake
    http = TestClient(appmod.create_app(FakeBackend("sft_0.5b", responder=saved_responder(tweak)), CFG5, CFG1))
    cfg5 = copy.deepcopy(CFG5)
    return P.run_parity(sc.ServiceClient(http), "fake", "sft_0.5b", cfg5, CFG1, ECFG, limit=limit, out_dir=out_dir), out_dir


@needs_outputs
def test_parity_ok_when_the_service_gives_the_saved_outputs(tmp_path):
    r, out_dir = run_fake_parity(tmp_path)
    assert r["items"]["n_items"] == 53 and r["items"]["identical"] == 53 and r["items"]["identical_percent"] == 100.0 and r["items"]["different"] == []
    assert r["ok"] is True and r["verdict"] == "OK" and r["reasons"] == []
    assert r["deltas"]["token_f1"] == pytest.approx(0.0, abs=1e-12) and r["deltas"]["strict_correct_refusals"] == 0 and r["deltas"]["false_refusals"] == 0
    assert r["saved"]["token_f1"] == pytest.approx(r["service"]["token_f1"], abs=1e-12)
    assert {k: v for k, v in r["saved"].items() if k != "token_f1"} == {k: v for k, v in r["service"].items() if k != "token_f1"}
    assert r["saved"]["n_refusal"] == 11 and r["saved"]["n_answerable"] == 42
    assert r["run_name"] == "svc_fake_lora_fp16_lr2e-4_e2" and (out_dir / "eval_svc_fake_lora_fp16_lr2e-4_e2_val.json").exists()
    assert "STOP" not in P.format_result(r) and "identical outputs 53 of 53 (100.0%)" in P.format_result(r)


@needs_outputs
def test_parity_service_file_is_in_the_evaluator_format_and_the_report_code_reads_it(tmp_path):
    r, out_dir = run_fake_parity(tmp_path)
    d = json.loads((out_dir / r["service_file"]).read_text(encoding="utf-8"))
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    assert set(saved) <= set(d) | {"model_revision"} and d["split"] == "val" and d["data_sha256"] == saved["data_sha256"]
    assert d["summary"]["n_items"] == 53 and [i["id"] for i in d["items"]] == [i["id"] for i in saved["items"]]
    assert d["summary"]["answerable"]["token_f1"]["mean"] == pytest.approx(saved["summary"]["answerable"]["token_f1"]["mean"], abs=1e-12)
    evals, skipped = report.load_evals(out_dir, ECFG)
    names = [e["run_name"] for e in evals]
    assert "svc_fake_lora_fp16_lr2e-4_e2" in names and report.is_hidden({"run_name": "svc_vllm_q15_lora_fp16_e3"}, ECFG)


@needs_outputs
def test_parity_stops_when_two_refusals_are_lost(tmp_path):
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    refused_ids = [i["id"] for i in saved["items"] if i["kind"] == "refusal" and i["refusal_strict"]][:2]
    r, _ = run_fake_parity(tmp_path, tweak=lambda id_, out: "An invented answer." if id_ in refused_ids else out)
    assert r["ok"] is False and r["verdict"] == "STOP" and r["deltas"]["strict_correct_refusals"] == -2
    assert any("strict correct refusals moved by -2" in x for x in r["reasons"]) and r["items"]["identical"] == 51
    assert {d["id"] for d in r["items"]["different"]} == set(refused_ids) and "STOP because" in P.format_result(r)


@needs_outputs
def test_parity_one_refusal_item_is_allowed_but_reported(tmp_path):
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    one = next(i["id"] for i in saved["items"] if i["kind"] == "refusal" and i["refusal_strict"])
    r, _ = run_fake_parity(tmp_path, tweak=lambda id_, out: "An invented answer." if id_ == one else out)
    assert r["ok"] is True and r["deltas"]["strict_correct_refusals"] == -1 and r["items"]["identical"] == 52 and len(r["items"]["different"]) == 1


@needs_outputs
def test_parity_stops_on_a_false_refusal_pair_and_on_f1(tmp_path):
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    ans = [i["id"] for i in saved["items"] if i["kind"] == "answerable" and not i["refusal_strict"]]
    r, _ = run_fake_parity(tmp_path, tweak=lambda id_, out: REF if id_ in ans[:2] else out)
    assert r["verdict"] == "STOP" and r["deltas"]["false_refusals"] == 2
    r2, _ = run_fake_parity(tmp_path / "f1", tweak=lambda id_, out: "x" if id_ in ans[:6] else out)
    assert r2["verdict"] == "STOP" and r2["deltas"]["token_f1"] < -0.03 and any("token F1 moved" in x for x in r2["reasons"])


@needs_outputs
def test_parity_limit_run_compares_the_same_first_items_only(tmp_path):
    r, _ = run_fake_parity(tmp_path, limit=5)
    assert r["limit"] == 5 and r["items"]["n_items"] == 5 and r["items"]["identical"] == 5 and r["ok"] is True
    summ = P.summary_of({"fake/sft_0.5b": r}, CFG5["parity"])
    assert summ["all_ok"] is False                                                                      # a partial run never opens the gate


def test_parity_verdict_boundaries():
    rule = CFG5["parity"]
    base = {"token_f1": 0.60, "strict_correct_refusals": 8, "false_refusals": 3, "n_items": 53}
    ok = lambda **kw: P.parity_verdict(base, {**base, **kw}, rule)["ok"]                                # noqa: E731
    assert ok() and ok(strict_correct_refusals=7) and ok(strict_correct_refusals=9) and ok(false_refusals=2) and ok(false_refusals=4)
    assert not ok(strict_correct_refusals=6) and not ok(strict_correct_refusals=10) and not ok(false_refusals=1) and not ok(false_refusals=5)
    assert ok(token_f1=0.63) and ok(token_f1=0.57) and not ok(token_f1=0.6301) and not ok(token_f1=0.5699)   # exactly 0.03 is inside
    assert not ok(n_items=52)
    assert rule["max_refusal_item_change"] == 1 and rule["max_f1_change"] == 0.03


def test_compare_items_and_metrics_of():
    saved = {"items": [{"id": "a", "output": "x", "token_f1": 1.0, "refusal_strict": False},
                       {"id": "b", "output": "y", "token_f1": 0.0, "refusal_strict": True}]}
    svc = {"items": [{"id": "a", "output": " x ", "kind": "answerable", "token_f1": 1.0, "refusal_strict": False},
                     {"id": "b", "output": "z", "kind": "refusal", "token_f1": 0.0, "refusal_strict": False},
                     {"id": "c", "output": "w", "kind": "answerable", "token_f1": 0.5, "refusal_strict": False}]}
    c = P.compare_items(saved, svc)
    assert c["identical"] == 1 and c["n_items"] == 3 and [d["id"] for d in c["different"]] == ["b", "c"] and c["missing_in_saved"] == ["c"]


def test_parity_summary_gate_structure():
    good = {"ok": True, "verdict": "OK", "reasons": [], "limit": None, "items": {"identical": 53, "n_items": 53},
            "deltas": {"token_f1": 0.0, "strict_correct_refusals": 0, "false_refusals": 0}}
    bad = {**good, "ok": False, "verdict": "STOP", "reasons": ["x"]}
    s = P.summary_of({"vllm/sft_0.5b": good, "vllm/sft_1.5b": bad}, CFG5["parity"])
    assert s["all_ok"] is False and s["results"]["vllm/sft_1.5b"]["verdict"] == "STOP" and P.summary_of({"a": good}, CFG5["parity"])["all_ok"] is True
    assert B.parity_gate(s, "vllm", "sft_0.5b")[0] is True
    ok, why = B.parity_gate(s, "vllm", "sft_1.5b")
    assert not ok and "STOP" in why
    assert not B.parity_gate(s, "vllm", "dpo2_0.5b")[0] and not B.parity_gate(None, "vllm", "sft_0.5b")[0]
    part = P.summary_of({"vllm/sft_0.5b": {**good, "limit": 5}}, CFG5["parity"])
    assert not B.parity_gate(part, "vllm", "sft_0.5b")[0] and "partial" in B.parity_gate(part, "vllm", "sft_0.5b")[1]
    assert B.parity_gate(None, "hf", "sft_0.5b")[0] is True                                             # hf is the evaluator's own code


def test_parity_service_backend_checks_the_prompt_it_sends():
    val = load_items(ROOT / ECFG["paths"]["splits"]["val"])[:2]
    http = TestClient(appmod.create_app(FakeBackend("sft_0.5b"), CFG5, CFG1))
    sb = P.ServiceBackend(sc.ServiceClient(http), CFG5["models"]["sft_0.5b"], CFG1)
    out = sb.generate([v["prompt"] for v in val])
    assert [o["text"] for o in out] == ["fake answer"] * 2 and sb.completion_nll([], []) == [] and len(sb.server_latency_ms) == 2
    altered = [{"role": "system", "content": "another system message"}, val[0]["prompt"][1]]
    with pytest.raises(ValueError, match="not what the service builds"):
        sb.generate([altered])


# ---- benchmark functions -----------------------------------------------------------------------------------------------------------------------------
def test_percentile_linear_interpolation():
    assert B.percentile([5.0], 95) == 5.0 and B.percentile([1, 2, 3, 4, 5], 50) == 3 and B.percentile([1, 2, 3, 4], 50) == 2.5
    assert B.percentile(list(range(1, 101)), 95) == pytest.approx(95.05) and B.percentile([3, 1, 2], 0) == 1 and B.percentile([3, 1, 2], 100) == 3
    with pytest.raises(ValueError):
        B.percentile([], 50)


def test_summarize_run_on_fake_timings():
    recs = [{"latency_s": 0.1 * (i + 1), "n_new_tokens": 10 + i, "error": None} for i in range(10)] + [{"latency_s": 9.0, "n_new_tokens": 0, "error": "X"}]
    s = B.summarize_run(recs, wall_s=2.0)
    assert s["n_requests"] == 11 and s["n_errors"] == 1 and s["output_tokens"] == sum(10 + i for i in range(10))
    assert s["latency_p50_ms"] == pytest.approx(550.0) and s["latency_p95_ms"] == pytest.approx(955.0) and s["latency_mean_ms"] == pytest.approx(550.0)
    assert s["tokens_per_s"] == pytest.approx(s["output_tokens"] / 2.0) and s["requests_per_s"] == pytest.approx(10 / 2.0)    # errors excluded
    empty = B.summarize_run([{"latency_s": 1.0, "n_new_tokens": 0, "error": "X"}], wall_s=1.0)
    assert empty["latency_p50_ms"] is None and empty["tokens_per_s"] == 0.0


def test_aggregate_repeats_min_median_max():
    runs = [{"latency_p50_ms": 100.0, "latency_p95_ms": 200.0, "tokens_per_s": 50.0, "requests_per_s": 2.0},
            {"latency_p50_ms": 120.0, "latency_p95_ms": 260.0, "tokens_per_s": 40.0, "requests_per_s": 1.8},
            {"latency_p50_ms": 110.0, "latency_p95_ms": 230.0, "tokens_per_s": 45.0, "requests_per_s": 1.9}]
    a = B.aggregate_repeats(runs)
    assert a["latency_p50_ms"] == {"min": 100.0, "median": 110.0, "max": 120.0, "n": 3} and a["tokens_per_s"]["median"] == 45.0
    assert B.aggregate_repeats([{"latency_p50_ms": None}])["latency_p50_ms"]["median"] is None


def test_too_close_to_separate_wording():
    a, b, c = {"min": 10, "median": 12, "max": 14}, {"min": 13, "median": 15, "max": 17}, {"min": 30, "median": 32, "max": 34}
    assert B.ranges_overlap(a, b) and not B.ranges_overlap(a, c) and B.ranges_overlap(a, {"min": 14, "median": 14, "max": 14})
    assert "too close to separate" in B.compare_note("vLLM", a, "hf", b, higher_is_better=False)
    n = B.compare_note("vLLM", a, "hf", c, higher_is_better=False, unit=" ms")
    assert "vLLM is clearly better" in n and "no overlap" in n
    assert "hf is clearly better" in B.compare_note("vLLM", a, "hf", c, higher_is_better=True)
    assert B.ranges_overlap({"min": None, "max": None, "median": None}, a)


def test_run_load_counts_errors_and_runs_requests_in_parallel():
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    def post(context, question):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        if question == "boom":
            raise RuntimeError("bad request")
        return {"n_new_tokens": 7, "latency_ms": 12.0}
    items = [("c", f"q{i}") for i in range(7)] + [("c", "boom")]
    records, wall = B.run_load(post, items, 4)
    assert len(records) == 8 and sum(1 for r in records if r["error"]) == 1 and "RuntimeError" in next(r["error"] for r in records if r["error"])
    assert state["max"] == 4 and wall < 8 * 0.05 and all(r["latency_s"] >= 0.05 for r in records)
    state["max"] = 0
    B.run_load(post, items[:3], 1)
    assert state["max"] == 1


def test_run_setting_warmup_levels_repeats_and_gpu():
    seen = []

    def post(context, question):
        seen.append(question)
        return {"n_new_tokens": 3, "latency_ms": 1.0}
    items = [("c", f"q{i}") for i in range(5)]
    mem = iter(range(1000, 100000, 100))
    out = B.run_setting(post, items, [1, 2], repeats=3, warmup=2, gpu_query=lambda: [{"name": "g", "total_mib": 1, "used_mib": next(mem)}], gpu_interval_s=0.005)
    assert seen[:2] == ["q0", "q1"] and len(seen) == 2 + 2 * 3 * 5                                      # warm-up first, excluded from the runs
    assert set(out) == {"1", "2"} and all(len(v["runs"]) == 3 and v["n_errors"] == 0 for v in out.values())
    assert all(r["n_requests"] == 5 and r["output_tokens"] == 15 for v in out.values() for r in v["runs"])
    assert out["2"]["stats"]["tokens_per_s"]["n"] == 3 and out["1"]["gpu"]["peak_used_mib"] is not None


def test_val_items_are_val_only():
    items = B.val_items(ECFG)
    val = load_items(ROOT / ECFG["paths"]["splits"]["val"])
    assert len(items) == 53 and items == [parse_example(r)[:2] for r in val] and B.val_items(ECFG, 6) == items[:6]
    for f in ("bench_serving.py", "parity_check.py", "phase5_report.py", "phase5_check.py"):
        src = (ROOT / "scripts" / f).read_text(encoding="utf-8")
        assert "sft_test" not in src and '"test"' not in src and "split test" not in src and 'splits"]["test' not in src, f
    assert CFG5["bench"]["split"] == "val" and CFG5["parity"]["split"] == "val"


def test_hf_available_is_false_on_the_laptop():
    ok, info = B.hf_available()
    assert ok is False and info


# ---- the scripts end to end with the fake backend (a real service process) ----------------------------------------------------------------------------
def patched_cfg(tmp_path):
    cfg = copy.deepcopy(CFG5)
    cfg["paths"] = {**cfg["paths"], "outputs_dir": str(tmp_path / "outputs"), "results_dir": str(tmp_path / "outputs" / "phase5"),
                    "parity_summary": str(tmp_path / "outputs" / "phase5" / "parity_summary.json"),
                    "results_md": str(tmp_path / "outputs" / "phase5_results.md"), "results_csv": str(tmp_path / "outputs" / "phase5_bench.csv"),
                    "mlruns_dir": str(tmp_path / "outputs" / "mlruns")}
    cfg["server"] = {**cfg["server"], "log_dir": str(tmp_path / "outputs" / "phase5"), "health_timeout_s": 90}
    cfg["bench"] = {**cfg["bench"], "request_timeout_s": 60}
    (tmp_path / "outputs" / "phase5").mkdir(parents=True)
    return cfg


def test_bench_main_with_the_fake_backend(tmp_path, monkeypatch, capsys):
    cfg = patched_cfg(tmp_path)
    monkeypatch.setattr(B, "load_phase5_config", lambda: cfg)
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(B, "nvidia_smi_query", lambda: None)
    monkeypatch.setattr(B, "val_items", lambda c, n=None: [(f"context {i}", f"question {i}?") for i in range(6)][: n or 6])
    port = free_port()
    assert B.main(["--backends", "fake", "--models", "sft_1.5b", "--concurrency", "1,3", "--repeats", "2", "--n-prompts", "4", "--port", str(port)]) == 0
    res = json.loads((tmp_path / "outputs" / "phase5" / "bench_fake_sft_1.5b.json").read_text(encoding="utf-8"))
    assert res["status"] == "ok" and res["backend"] == "fake" and res["split"] == "val" and res["partial_prompts"] is True and res["n_prompts"] == 4
    assert res["startup_wall_s"] > 0 and res["startup_service_s"] is not None and set(res["levels"]) == {"1", "3"}   # start-up apart from the requests
    assert "serving.app" in res["service_command"] and res["client_command"] == " ".join(sys.argv)
    for lv in res["levels"].values():
        assert len(lv["runs"]) == 2 and lv["n_errors"] == 0 and lv["stats"]["latency_p50_ms"]["n"] == 2 and lv["gpu"]["peak_used_mib"] is None
        assert all(r["n_requests"] == 4 for r in lv["runs"])
    assert "versions" in res and res["batching_config"]["window_ms"] == 5
    # a finished setting is skipped the second time
    capsys.readouterr()
    B.main(["--backends", "fake", "--models", "sft_1.5b", "--concurrency", "1", "--repeats", "1", "--n-prompts", "4", "--port", str(port)])
    assert "result exists, skipped" in capsys.readouterr().out


def test_bench_main_refuses_vllm_without_a_passed_parity_check(tmp_path, monkeypatch, capsys):
    cfg = patched_cfg(tmp_path)
    monkeypatch.setattr(B, "load_phase5_config", lambda: cfg)
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(B, "nvidia_smi_query", lambda: None)
    monkeypatch.setattr(B, "val_items", lambda c, n=None: [("context", "question?")] * (n or 2))
    assert B.main(["--backends", "vllm", "--models", "sft_0.5b", "--n-prompts", "2"]) == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "no parity check found" in out
    res = json.loads((tmp_path / "outputs" / "phase5" / "bench_vllm_sft_0.5b.json").read_text(encoding="utf-8"))
    assert res["status"] == "skipped" and "parity" in res["reason"] and "levels" not in res


def test_bench_main_records_hf_as_skipped_when_it_cannot_run(tmp_path, monkeypatch):
    cfg = patched_cfg(tmp_path)
    monkeypatch.setattr(B, "load_phase5_config", lambda: cfg)
    monkeypatch.setattr(B, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(B, "nvidia_smi_query", lambda: None)
    monkeypatch.setattr(B, "hf_available", lambda: (False, "ModuleNotFoundError: No module named 'peft'"))
    monkeypatch.setattr(B, "val_items", lambda c, n=None: [("context", "question?")] * (n or 2))
    B.main(["--backends", "hf", "--models", "sft_1.5b", "--n-prompts", "2"])
    res = json.loads((tmp_path / "outputs" / "phase5" / "bench_hf_sft_1.5b.json").read_text(encoding="utf-8"))
    assert res["status"] == "skipped" and "peft" in res["reason"] and "cannot run in this environment" in res["reason"]


@needs_outputs
def test_parity_main_with_the_fake_backend_stops_and_writes_the_summary(tmp_path, monkeypatch, capsys):
    cfg = patched_cfg(tmp_path)
    (tmp_path / "outputs" / SAVED.name).write_bytes(SAVED.read_bytes())
    monkeypatch.setattr(P, "load_phase5_config", lambda: cfg)
    monkeypatch.setattr(P, "REPO_ROOT", tmp_path)
    code = P.main(["--backends", "fake", "--models", "sft_0.5b", "--limit", "3", "--port", str(free_port())])
    out = capsys.readouterr().out
    assert code == 2 and "STOP:" in out and "Do NOT run the benchmark" in out                            # the fake outputs differ from the saved ones
    summ = json.loads((tmp_path / "outputs" / "phase5" / "parity_summary.json").read_text(encoding="utf-8"))
    assert summ["all_ok"] is False and summ["results"]["fake/sft_0.5b"]["limit"] == 3
    r = json.loads((tmp_path / "outputs" / "phase5" / "parity_fake_sft_0.5b.json").read_text(encoding="utf-8"))
    assert r["startup_wall_s"] > 0 and r["versions"]["fastapi"] and "serving.app" in r["service_command"] and r["verdict"] == "STOP"


# ---- check script --------------------------------------------------------------------------------------------------------------------------------------
def test_phase5_check_reports_missing_files(tmp_path):
    problems = C5.check_inputs(CFG5, ECFG, root=tmp_path)
    assert len(problems) >= 2 * 3 + 1 and any("sft_0.5b" in p and "adapter_model.safetensors" in p for p in problems)
    assert not any("sft_1.5b" in p and "adapter_model" in p for p in problems), "a Hugging Face adapter has no local files to check"
    assert any("eval_q15_lora_fp16_e3_val.json" in p for p in problems) and any("sft_val.jsonl" in p for p in problems)


def test_version_tuple():
    assert C5.version_tuple("0.10.0") == (0, 10, 0) and C5.version_tuple("0.16") == (0, 16) and C5.version_tuple("1.2.3") == (1, 2, 3)
    assert C5.version_tuple("0.16.0.dev1") == (0, 16, 0) and C5.version_tuple("x") == ()


def test_package_problems_rouge_score_and_torchao():
    ok = {"vllm": "0.32.0", "httpx": "0.28.1", "rouge_score": "0.1.2", "torchao": None}
    assert C5.package_problems(ok) == []
    assert C5.package_problems({**ok, "torchao": "0.16.0"}) == [] and C5.package_problems({**ok, "torchao": "0.17.1"}) == []
    old = C5.package_problems({**ok, "torchao": "0.10.0"})
    assert len(old) == 1 and "torchao 0.10.0" in old[0] and "pip uninstall -y torchao" in old[0]
    miss = C5.package_problems({**ok, "rouge_score": None})
    assert len(miss) == 1 and "rouge_score" in miss[0] and "pip install rouge_score" in miss[0]
    assert len(C5.package_problems({"vllm": None, "httpx": None, "rouge_score": None, "torchao": "0.10.0"})) == 4
    assert {"rouge_score", "torchao"} <= set(C5.PACKAGES)


def test_phase5_check_main_prints_the_new_problems(monkeypatch, capsys):
    monkeypatch.setattr(C5, "check_inputs", lambda *a, **k: [])
    monkeypatch.setattr(C5, "nvidia_smi_query", lambda: None)
    monkeypatch.setattr(C5, "hf_available", lambda: (True, "x"))
    vers = {p: "1.0" for p in C5.PACKAGES}
    monkeypatch.setattr(C5, "package_versions", lambda: {**vers, "rouge_score": None, "torchao": "0.10.0"})
    assert C5.main([]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: rouge_score is missing" in out and "PROBLEM: torchao 0.10.0" in out and "NOT READY (2 problems)" in out
    monkeypatch.setattr(C5, "package_versions", lambda: {**vers, "torchao": None})
    assert C5.main([]) == 0 and "READY" in capsys.readouterr().out


def test_phase5_check_on_the_real_files_if_present():
    if not (ROOT / "outputs" / "adapters_dpo2" / "dpo2_mix_a1_lr1e-5" / "e1" / "adapter_model.safetensors").exists():
        pytest.skip("adapters are not on this machine")
    assert C5.check_inputs(CFG5, ECFG) == []
    assert set(C5.package_versions()) >= {"vllm", "torch", "transformers", "peft", "httpx", "fastapi"}


# ---- report and MLflow -----------------------------------------------------------------------------------------------------------------------------------
def fake_run(backend, model, med_p50, spread=2.0, status="ok"):
    def st(med, hi_lo):
        return {"min": med - hi_lo, "median": med, "max": med + hi_lo, "n": 3}
    runs = [{"n_requests": 53, "n_errors": 0, "wall_s": 10.0, "output_tokens": 500, "latency_p50_ms": med_p50 + d, "latency_p95_ms": 2 * med_p50 + d,
             "tokens_per_s": 50.0 + d, "requests_per_s": 5.0, "latency_mean_ms": med_p50} for d in (-spread, 0.0, spread)]
    lv = {"concurrency": 4, "runs": runs, "n_errors": 0, "gpu": {"peak_used_mib": 9000, "min_used_mib": 8000, "n_samples": 10},
          "stats": {"latency_p50_ms": st(med_p50, spread), "latency_p95_ms": st(2 * med_p50, spread), "tokens_per_s": st(50.0, spread),
                    "requests_per_s": st(5.0, 0.1)}}
    if status != "ok":
        return {"backend": backend, "model": model, "status": status, "reason": "hf cannot run: no peft"}
    return {"backend": backend, "model": model, "status": "ok", "split": "val", "n_prompts": 53, "partial_prompts": False, "concurrency_levels": [4],
            "repeats": 3, "warmup_requests": 3, "max_new_tokens": 256, "gpu": [{"name": "Tesla T4", "total_mib": 15360, "used_mib": 1}],
            "client_command": "python scripts/bench_serving.py", "batching_config": CFG5["batching"], "startup_wall_s": 95.0, "startup_service_s": 80.0,
            "versions": {"python": "3.13", "vllm": "0.32.0", "torch": "2.x"}, "gpu_used_after_load_mib": 12000, "service_command": f"PHASE5_BACKEND={backend} python -m serving.app",
            "batching_stats": {"n_batches": 9, "mean_batch_size": 3.0}, "levels": {"4": lv}}


def fake_parity(ok=True):
    saved = {"token_f1": 0.615, "strict_correct_refusals": 8, "false_refusals": 3, "n_items": 53, "n_refusal": 11, "n_answerable": 42}
    svc = {**saved, "strict_correct_refusals": 8 if ok else 5}
    return {"backend": "vllm", "model": "sft_0.5b", "adapter": "a", "saved_file": "s.json", "service_file": "v.json", "limit": None,
            "saved": saved, "service": svc, "ok": ok, "verdict": "OK" if ok else "STOP", "reasons": [] if ok else ["x"],
            "deltas": {"token_f1": 0.0, "strict_correct_refusals": svc["strict_correct_refusals"] - 8, "false_refusals": 0},
            "rule": {"max_refusal_item_change": 1, "max_f1_change": 0.03}, "versions": {"vllm": "0.32.0"},
            "items": {"n_items": 53, "identical": 52, "identical_percent": 100 * 52 / 53, "missing_in_saved": [],
                      "different": [{"id": "tr2-0001", "kind": "answerable", "saved_output": "a", "service_output": "b", "saved_f1": 0.5, "service_f1": 0.4,
                                     "saved_refused": False, "service_refused": False}]}}


def test_report_tables_come_from_the_saved_results_only():
    results = {"parity": {("vllm", "sft_0.5b"): fake_parity()},
               "bench": {("vllm", "sft_0.5b"): fake_run("vllm", "sft_0.5b", 100.0), ("hf", "sft_0.5b"): fake_run("hf", "sft_0.5b", 400.0),
                         ("vllm", "sft_1.5b"): fake_run("vllm", "sft_1.5b", 300.0), ("hf", "sft_1.5b"): fake_run("hf", "sft_1.5b", 301.0),
                         ("hf", "dpo2_0.5b"): fake_run("hf", "dpo2_0.5b", 1, status="skipped")}}
    md = R.build_markdown(results, {"ok": True, "versions_before": {"vllm": "0.32.0"}}, CFG5)
    assert "## Parity" in md and "52/53 (98.1%)" in md and "**OK**" in md and "no differing output" not in md and "tr2-0001" in md
    assert "## Start-up time (separate from the request numbers)" in md and "| vllm | sft_0.5b | 95 | 80 | 12000 |" in md
    assert "Benchmark: 0.5B SFT" in md and "100 (98 to 102)" in md and "| vllm | 4 | 100 (98 to 102) |" in md
    assert "vLLM is clearly better" in md                                                                  # 100 vs 400 ms: no overlap
    assert "too close to separate" in md                                                                   # 300 vs 301 ms: overlap
    assert "## Settings that were not measured" in md and "hf / dpo2_0.5b: hf cannot run: no peft" in md
    assert "Tesla T4" in md and "continuous batching" in md and "reservation" in md and "test split not used" in md
    assert md.count("not measured") >= 1
    assert "0.5B DPO v2" in md and "1.5B SFT" in md


def test_report_stop_verdict_is_shown_and_empty_results_are_labelled():
    md = R.build_markdown({"parity": {("vllm", "sft_0.5b"): fake_parity(ok=False)}, "bench": {}}, None, CFG5)
    assert "**STOP**" in md and "no benchmark result file found" in md and "not measured" in md
    rows = R.csv_rows({("vllm", "sft_0.5b"): fake_run("vllm", "sft_0.5b", 100.0), ("hf", "x"): fake_run("hf", "x", 1, status="skipped")})
    assert len(rows) == 3 and rows[0]["backend"] == "vllm" and rows[0]["concurrency"] == 4 and rows[0]["repeat"] == 1 and rows[0]["gpu_peak_mib"] == 9000


def test_report_main_reads_files_and_writes_markdown_and_csv(tmp_path, monkeypatch):
    cfg = patched_cfg(tmp_path)
    res = tmp_path / "outputs" / "phase5"
    for (b, m), d in {("vllm", "sft_0.5b"): fake_run("vllm", "sft_0.5b", 100.0)}.items():
        (res / f"bench_{b}_{m}.json").write_text(json.dumps(d), encoding="utf-8")
    (res / "parity_vllm_sft_0.5b.json").write_text(json.dumps(fake_parity()), encoding="utf-8")
    (res / "parity_summary.json").write_text("{}", encoding="utf-8")                                       # must not be read as a result
    monkeypatch.setattr(R, "load_phase5_config", lambda: cfg)
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    assert R.main([]) == 0
    md = (tmp_path / "outputs" / "phase5_results.md").read_text(encoding="utf-8")
    assert "Phase 5: serving results" in md and "100 (98 to 102)" in md
    assert (tmp_path / "outputs" / "phase5_bench.csv").read_text(encoding="utf-8").splitlines()[0].startswith("backend,model,concurrency,repeat")


class FakeMlflow:
    def __init__(self, fail=()):
        self.calls, self.fail, self.runs = [], fail, []

    def module(self):
        m = types.ModuleType("mlflow")
        outer = self

        def make(name):
            def fn(*a, **k):
                outer.calls.append((name, a, k))
                if name in outer.fail:
                    raise RuntimeError(f"{name} broke")
            return fn
        for n in ("set_tracking_uri", "set_experiment", "log_params", "set_tags", "log_metrics", "end_run"):
            setattr(m, n, make(n))

        def start_run(*a, **k):
            outer.calls.append(("start_run", a, k))
            if "start_run" in outer.fail:
                raise RuntimeError("start_run broke")
            outer.runs.append(k.get("run_name"))
            return types.SimpleNamespace(info=types.SimpleNamespace(run_id="rid"))
        m.start_run = start_run
        return m


def test_mlflow_logging_one_run_per_setting_and_parity_experiment_name(tmp_path, monkeypatch):
    fm = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fm.module())
    cfg = patched_cfg(tmp_path)
    results = {"parity": {("vllm", "sft_0.5b"): fake_parity()},
               "bench": {("vllm", "sft_0.5b"): fake_run("vllm", "sft_0.5b", 100.0), ("hf", "sft_0.5b"): fake_run("hf", "sft_0.5b", 1, status="skipped")}}
    n = R.log_to_mlflow(cfg, results)
    assert n == 2 and fm.runs == ["parity_vllm_sft_0.5b", "bench_vllm_sft_0.5b_c4"]                       # skipped settings are not logged
    exp = [c for c in fm.calls if c[0] == "set_experiment"]
    assert exp and all(c[1] == ("phase5_serving",) for c in exp)
    params = [c for c in fm.calls if c[0] == "log_params"]
    assert any(p[1][0].get("kind") == "benchmark" and p[1][0].get("concurrency") == "4" and p[1][0].get("split") == "val" for p in params)
    metrics = {}
    for c in fm.calls:
        if c[0] == "log_metrics":
            metrics.update(c[1][0])
    assert {"latency_p50_ms_min", "latency_p50_ms_median", "latency_p50_ms_max", "tokens_per_s_median", "gpu_peak_mib", "startup_wall_s",
            "parity_ok", "identical_outputs", "delta_token_f1"} <= set(metrics)
    steps = [c[2].get("step") for c in fm.calls if c[0] == "log_metrics" and "step" in c[2]]
    assert steps == [1, 2, 3]                                                                              # one step per repeat


def test_mlflow_errors_never_stop_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow(fail=("start_run",)).module())
    cfg = patched_cfg(tmp_path)
    results = {"parity": {}, "bench": {("vllm", "sft_0.5b"): fake_run("vllm", "sft_0.5b", 100.0)}}
    assert R.log_to_mlflow(cfg, results) == 1
    assert "WARNING: MLflow start failed" in capsys.readouterr().out
    monkeypatch.delitem(sys.modules, "mlflow")
    monkeypatch.setitem(sys.modules, "mlflow", None)                                                      # mlflow not installed
    assert R.log_to_mlflow(cfg, results) == 1


# ---- collect sets, bundle, notebook ------------------------------------------------------------------------------------------------------------------
def test_phase5_collect_sets(tmp_path):
    out = tmp_path / "outputs"
    rels = ("phase5_results.md", "phase5_bench.csv", "phase5_smoke.json", "phase5/bench_vllm_sft_0.5b.json", "phase5/parity_vllm_sft_0.5b.json",
            "phase5/server_bench_vllm_sft_0.5b.log", "eval_svc_vllm_lora_fp16_lr2e-4_e2_val.json", "refusal_detection_review_svc_vllm_x_val.csv",
            "mlruns/0/meta.yaml", "phase4b_selection.json", "eval_dpo2_x_val.json", "adapters/lora_fp16_lr2e-4/e2/adapter_config.json", "phase3_final.md")
    for rel in rels:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text("x", encoding="utf-8")
    got = {p.relative_to(out).as_posix() for p in collect.phase5_result_files(out, ECFG)}
    assert got == set(rels[:9])
    assert not any(x in got for x in ("phase4b_selection.json", "eval_dpo2_x_val.json", "phase3_final.md")) and not any(x.startswith("adapters/") for x in got)
    assert not any("svc" in p.name or "phase5" in p.name for p in collect.phase3_result_files(out, ECFG))
    assert not any("svc" in p.name or "phase5" in p.name for p in collect.phase4b_result_files(out, ECFG))


def test_phase5_start_zip_lists_the_three_adapters_and_their_val_files(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "REPO_ROOT", tmp_path)
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for m in CFG5["models"].values():
        if not m.get("adapter"):
            (out / f"eval_{m['parity_eval']}_val.json").write_text("{}", encoding="utf-8")
            continue
        d = tmp_path / m["adapter"]
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        (d / "adapter_model.safetensors").write_bytes(b"w")
        (out / f"eval_{m['parity_eval']}_val.json").write_text("{}", encoding="utf-8")
    files = collect.phase5_start_files(out, CFG5)
    names = sorted(p.relative_to(out).as_posix() for p in files)
    assert len(names) == 7 and "test-adapters/b/adapter_model.safetensors" in names and "eval_q15_lora_fp16_e3_val.json" in names
    assert not any("sft_1.5b" in n for n in names)
    (out / "eval_q15_lora_fp16_e3_val.json").unlink()
    with pytest.raises(SystemExit, match="missing files"):
        collect.phase5_start_files(out, CFG5)


def test_bundle_has_all_phase5_files():
    b = ECFG["bundle"]["files"]
    for f in ("scripts/vllm_smoke.py", "scripts/phase5_check.py", "scripts/parity_check.py", "scripts/bench_serving.py", "scripts/phase5_report.py"):
        assert f in b, f
    z = ROOT / ECFG["paths"]["colab_bundle"]
    if z.exists():
        names = zipfile.ZipFile(z).namelist()
        for n in ("serving/batching.py", "serving/client.py", "serving/backends.py", "serving/app.py", "scripts/parity_check.py", "scripts/bench_serving.py",
                  "scripts/phase5_report.py", "scripts/phase5_check.py", "configs/phase5.yaml", "data/instruction/sft_val.jsonl"):
            assert n in names, n
        assert not any(x.startswith("outputs/") for x in names)


def test_config_fixes_the_stage_2_plan():
    assert CFG5["batching"] == {"window_ms": 5, "max_batch": {"vllm": 16, "hf": 8}}
    assert REAL5["parity"]["models"] == ["sft_1.5b"] and CFG5["parity"]["backends"] == ["vllm"]
    assert CFG5["parity"]["max_refusal_item_change"] == 1 and CFG5["parity"]["max_f1_change"] == 0.03 and CFG5["parity"]["concurrency"] == 1
    b = CFG5["bench"]
    assert b["concurrency"] == [1, 4, 16] and b["repeats"] == 3 and b["backends"] == ["vllm", "hf"] and b["max_new_tokens"] == 256
    assert set(REAL5["bench"]["models"]) == {"sft_1.5b"} and CFG5["mlflow"]["experiment_name"] == "phase5_serving"
    assert CFG5["backend"]["env_port"] == "PHASE5_PORT"
    text = (ROOT / "configs" / "phase5.yaml").read_text(encoding="utf-8")
    assert "P5-0 (2026-10-21)" in text and "vLLM 0.32.0" in text


def _cells():
    nb = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    return nb["cells"]


def _p5_code():
    code = ["".join(c["source"]) for c in _cells() if c["cell_type"] == "code"]
    out = {}
    for c in code:
        for line in c.splitlines()[:2]:
            if line.startswith("# P5-"):
                out[line.split(".")[0].replace("# P5-", "")] = c
    return code, out


def test_notebook_stage_2_cells():
    code, p5 = _p5_code()
    assert sorted(p5) == [str(i) for i in range(7)]                                                       # P5-0 .. P5-6
    for k, c in p5.items():
        assert c.splitlines()[0] == "%cd /content/climate_slm_project", k                                 # the cwd mistake cannot come back
        assert len(c.splitlines()) <= 15 and "test" not in c.lower() and "requirements-gpu.txt" not in c, k
        assert ("pip uninstall" in c) == (k == "1"), k                                                    # torchao is removed in P5-1 only
    assert "phase5_start.zip" in p5["1"] and "scripts/phase5_check.py" in p5["1"]
    assert "pip install -q rouge_score" in p5["1"] and "pip uninstall -y torchao" in p5["1"]
    assert p5["1"].index("rouge_score") < p5["1"].index("phase5_check.py") and p5["1"].index("torchao") < p5["1"].index("phase5_check.py")
    p51_md = [("".join(c["source"])) for c in _cells() if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("**P5-1.**")]
    assert len(p51_md) == 1 and "rouge_score" in p51_md[0] and "torchao" in p51_md[0] and "training cells are not touched" in p51_md[0]
    for k in ("2", "3", "4"):                                                                             # the cells that need both come after P5-1
        assert code.index(p5[k]) > code.index(p5["1"])
    for c in code:                                                                                        # training cells keep their own torchao line
        if c.lstrip().startswith(("# P3-1.", "# P4-1.", "# P4b-1.")):
            assert "pip uninstall -y -q torchao" in c and "rouge_score" not in c
    assert "scripts/parity_check.py" in p5["2"] and "--backends" not in p5["2"]
    assert "scripts/bench_serving.py --backends vllm" in p5["3"] and "scripts/bench_serving.py --backends hf" in p5["4"]
    assert "scripts/phase5_report.py" in p5["5"] and "phase5_results.md" in p5["5"]
    assert "phase5-results" in p5["6"] and "phase5_results.zip" in p5["6"]
    order = [code.index(p5[str(i)]) for i in range(7)]
    assert order == sorted(order)
    assert [c for c in code if "bench_serving" in c] == [p5["3"], p5["4"]] and [c for c in code if "parity_check" in c] == [p5["2"]]
    # the older sections are untouched
    for prefix in ("# P3-", "# P4-", "# P4b-"):
        assert len([c for c in code if c.lstrip().startswith(prefix)]) == 11
    md = " ".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "markdown" and ("P5-2" in "".join(c["source"]) or "stage 2" in "".join(c["source"])))
    assert "STOP" in md and "val split only" in md and "NOT run yet" in md and "Do NOT run set-up cell 3" in md


def test_stage_2_modules_import_no_gpu_library():
    for f in list((ROOT / "serving").glob("*.py")) + [ROOT / "scripts" / n for n in ("bench_serving.py", "parity_check.py", "phase5_report.py", "phase5_check.py")]:
        head = f.read_text(encoding="utf-8").split("\ndef ")[0].split("\nclass ")[0]
        for lib in ("torch", "transformers", "peft", "vllm", "mlflow"):
            assert f"\nimport {lib}" not in head and f"\nfrom {lib}" not in head, (f.name, lib)
    src = (ROOT / "serving" / "backends.py").read_text(encoding="utf-8")
    assert src.count("from vllm") == 3 and "import vllm" not in src.split("class VLLMServingBackend")[0]       # inside load() and _run_batch() only


def test_wait_for_gpu_release():
    assert sc.wait_for_gpu_release(None) == (True, None)
    seq = iter([14000, 9000, 1500, 1500])
    q = lambda: [{"name": "g", "total_mib": 15360, "used_mib": next(seq)}]                              # noqa: E731
    assert sc.wait_for_gpu_release(1000, slack_mib=600, timeout_s=5, poll_s=0.01, query=q) == (True, 1500)
    assert sc.wait_for_gpu_release(1000, slack_mib=600, timeout_s=0.05, poll_s=0.01, query=lambda: [{"name": "g", "total_mib": 1, "used_mib": 9000}]) == (False, 9000)
    assert sc.wait_for_gpu_release(1000, query=lambda: None) == (True, None)                           # no nvidia-smi: nothing to wait for


def test_server_stop_ends_the_process_group_on_linux_only():
    src = (ROOT / "serving" / "client.py").read_text(encoding="utf-8")
    assert "start_new_session=(os.name == \"posix\")" in src and "os.killpg" in src                    # Colab path: written, not run on the laptop


def test_a_real_backend_without_an_adapter_says_so_and_never_falls_back_to_the_fake_one():
    cfg = {**CFG5, "models": {**CFG5["models"], "broken": {"label": "x", "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct"}}}
    for name in ("hf", "vllm"):
        with pytest.raises(backends.BackendError, match="no adapter configured: set adapter_id"):
            make_backend(name, "broken", cfg, ECFG)
    assert isinstance(make_backend("fake", "broken", cfg), FakeBackend)                       # the fake one only when asked, for tests


def test_adapter_revision_is_passed_to_the_download(monkeypatch):
    downloads = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda repo_id, revision=None: downloads.append((repo_id, revision)) or "/cache/x"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    cfg = copy.deepcopy(CFG5)
    cfg["models"]["sft_1.5b"]["adapter_revision"] = "abc123"
    b = make_backend("vllm", "sft_1.5b", cfg)
    assert b.adapter == "GHOSTBHAIYA/climate-slm-1.5b-sft" and b.adapter_is_hub_id and b.adapter_revision == "abc123"
    assert b._local_adapter() == "/cache/x" and downloads == [("GHOSTBHAIYA/climate-slm-1.5b-sft", "abc123")]
    assert make_backend("vllm", "sft_0.5b", CFG5).adapter_is_hub_id is False                 # a local test adapter is not downloaded


def test_the_repository_has_no_adapter_weights_and_no_local_adapter_path_in_configs():
    banned = ("outputs/adapters", "adapters_dpo2", "q15_lora_fp16/e")
    for f in (ROOT / "configs").glob("phase5.yaml"):
        text = f.read_text(encoding="utf-8")
        assert not any(b in text for b in banned), f.name
    assert not list(ROOT.rglob("adapter_model.safetensors")) or all(".venv" in str(p) for p in ROOT.rglob("adapter_model.safetensors"))
