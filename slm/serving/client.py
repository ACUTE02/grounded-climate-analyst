"""Client side helpers for the parity check and the benchmark: talk to the service, start and stop a service process, read the GPU
memory with nvidia-smi. No torch / vllm / GPU library is imported (the client must not touch the GPU itself).

A service process is started as `python -m serving.app` with PHASE5_BACKEND / PHASE5_MODEL / PHASE5_PORT in its environment, its
output goes to a log file (the service never logs context text), and it is stopped after the runs, so each (backend, model) setting
has its own clean process and its own startup time."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from climate_slm.config import REPO_ROOT


class ServiceError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class ServiceClient:
    """Thin wrapper around an httpx-like client (httpx.Client with a base_url, or a Starlette TestClient in the tests)."""

    def __init__(self, http, timeout: float = 600.0):
        self.http, self.timeout = http, timeout

    def health(self) -> dict:
        r = self.http.get("/health", timeout=self.timeout)
        if r.status_code != 200:
            raise ServiceError(r.status_code, r.text[:200])
        return r.json()

    def answer(self, context: str, question: str, max_new_tokens: int | None = None, model: str | None = None) -> dict:
        body = {"context": context, "question": question}
        if max_new_tokens is not None:
            body["max_new_tokens"] = max_new_tokens
        if model is not None:
            body["model"] = model
        r = self.http.post("/answer", json=body, timeout=self.timeout)
        if r.status_code != 200:
            detail = r.json().get("detail", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            raise ServiceError(r.status_code, str(detail)[:300])
        return r.json()


# ---- GPU memory --------------------------------------------------------------------------------------------------------------
def parse_nvidia_smi(csv_text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits`."""
    out = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            out.append({"name": parts[0], "total_mib": int(float(parts[1])), "used_mib": int(float(parts[2]))})
    return out


def nvidia_smi_query() -> list[dict] | None:
    """GPU name, total and used memory in MiB, or None if nvidia-smi is not available."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(r.stdout) if r.returncode == 0 else None


class GpuSampler:
    """Samples the used GPU memory (whole GPU, MiB) in a thread while the runs go on; stop() returns the peak.
    The whole GPU is measured because the service is another process: this includes the CUDA context and, for vLLM, the memory pool
    it reserves on start-up (gpu_memory_utilization), so the vLLM number is a reservation, not what the model needs."""

    def __init__(self, interval_s: float = 0.5, query=nvidia_smi_query):
        self.interval_s, self.query = interval_s, query
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            rows = self.query()
            if rows:
                self.samples.append(rows[0]["used_mib"])
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuSampler":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        s = self.samples
        return {"peak_used_mib": max(s) if s else None, "min_used_mib": min(s) if s else None, "n_samples": len(s)}


def wait_for_gpu_release(baseline_mib: int | None, slack_mib: int = 600, timeout_s: float = 90.0, poll_s: float = 2.0,
                         query=nvidia_smi_query) -> tuple[bool, int | None]:
    """After a service was stopped: wait until the used GPU memory is back near what it was before the service started.
    Returns (released, used MiB at the end). Without nvidia-smi or a baseline there is nothing to wait for."""
    if baseline_mib is None:
        return True, None
    deadline = time.perf_counter() + timeout_s
    used = None
    while True:
        rows = query()
        if not rows:
            return True, None
        used = rows[0]["used_mib"]
        if used <= baseline_mib + slack_mib:
            return True, used
        if time.perf_counter() >= deadline:
            return False, used
        time.sleep(poll_s)


# ---- a service process ----------------------------------------------------------------------------------------------------
class ServerProcess:
    def __init__(self, backend: str, model_key: str, cfg5: dict, port: int, log_path: Path, python: str | None = None,
                 extra_env: dict | None = None):
        self.backend, self.model_key, self.port, self.log_path = backend, model_key, port, Path(log_path)
        self.cfg5 = cfg5
        self.cmd = [python or sys.executable, "-m", "serving.app"]
        b = cfg5["backend"]
        self.env = {**os.environ, b["env_backend"]: backend, b["env_model"]: model_key, b["env_port"]: str(port),
                    "PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1", **(extra_env or {})}
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg5['service']['host']}:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.started_at = time.perf_counter()
        # on Linux (Colab) the service gets its own process group, so stop() can end vLLM's engine sub-processes with it
        # and their GPU memory is really free before the next setting starts
        self.proc = subprocess.Popen(self.cmd, cwd=str(REPO_ROOT), env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=(os.name == "posix"))

    def log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_healthy(self, make_client, timeout_s: float, poll_s: float) -> tuple[dict, float]:
        """(health dict, seconds from process start until the model is loaded). `make_client()` returns a ServiceClient."""
        deadline = time.perf_counter() + timeout_s
        last = None
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise ServiceError(0, f"the service process exited with code {self.proc.returncode}; log tail:\n{self.log_tail()}")
            try:
                h = make_client().health()
                if h.get("model_loaded"):
                    return h, time.perf_counter() - self.started_at
                last = h
            except Exception as err:                         # not up yet
                last = repr(err)[:100]
            time.sleep(poll_s)
        raise ServiceError(0, f"the service was not healthy after {timeout_s:.0f} s (last: {last}); log tail:\n{self.log_tail()}")

    def stop(self, timeout_s: float = 30.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._signal(signal.SIGTERM if os.name == "posix" else None)
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL if os.name == "posix" else "kill")
                self.proc.wait(timeout=10)
        if os.name == "posix" and self.proc is not None:                  # children that outlived the parent
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self._log is not None:
            self._log.close()

    def _signal(self, sig) -> None:
        """Terminate (or kill) the service; on Linux the whole process group. Not exercised on the laptop (Windows)."""
        if os.name == "posix" and isinstance(sig, signal.Signals):
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        elif sig == "kill":
            self.proc.kill()
        else:
            self.proc.terminate()

    def command_text(self) -> str:
        b = self.cfg5["backend"]
        return (f"{b['env_backend']}={self.backend} {b['env_model']}={self.model_key} {b['env_port']}={self.port} "
                + " ".join(self.cmd))
