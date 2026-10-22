"""Micro-batching: requests that arrive together are collected into one batch and handed to the engine once.

Both real backends (vLLM's offline LLM class and the evaluator's HFBackend) are not safe to call from many request threads at the
same time and are most efficient with a batch, so one worker thread owns the engine. `submit` returns a Future; the worker waits for
the first request, then up to `window_ms` for more (at most `max_batch`), calls `run_batch(payloads)` once and sets every Future.
A request that arrives while a batch is running waits for the next batch (this is not continuous batching: say so when reporting).
After close() no request waits for ever: a request queued before close still runs; anything left in the queue when the worker stops
fails with RuntimeError, and submit() after close raises at once.
No torch or vllm here: the function that talks to the engine is passed in, so this is tested on the laptop with plain functions."""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable


class MicroBatcher:
    def __init__(self, run_batch: Callable[[list[Any]], list[Any]], max_batch: int, window_ms: float):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.run_batch, self.max_batch, self.window_s = run_batch, max_batch, window_ms / 1000.0
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.batch_sizes: list[int] = []           # for the report: how big were the batches really

    def submit(self, payload: Any) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("the batcher is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="micro-batcher", daemon=True)
                self._thread.start()
            fut: Future = Future()
            self._q.put((payload, fut))             # under the lock: nothing can be queued after close()'s stop signal
        return fut

    def _collect(self) -> list[tuple[Any, Future]] | None:
        first = self._q.get()
        if first is None:                           # close signal
            return None
        batch = [first]
        deadline = time.monotonic() + self.window_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            try:
                item = self._q.get(timeout=remaining) if remaining > 0 else self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._q.put(None)                   # finish this batch, then stop
                break
            batch.append(item)
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            if batch is None:
                self._fail_pending()
                return
            self.batch_sizes.append(len(batch))
            try:
                results = self.run_batch([p for p, _ in batch])
                if len(results) != len(batch):
                    raise RuntimeError(f"run_batch returned {len(results)} results for {len(batch)} requests")
            except BaseException as err:            # noqa: BLE001  (every waiting request must hear about it)
                for _, fut in batch:
                    fut.set_exception(err)
                continue
            for (_, fut), res in zip(batch, results):
                if isinstance(res, BaseException):
                    fut.set_exception(res)
                else:
                    fut.set_result(res)

    def _fail_pending(self) -> None:
        """Fail every request still in the queue once the worker stops, so no caller waits for ever."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item is not None and not item[1].done():
                item[1].set_exception(RuntimeError("the batcher was closed before this request ran"))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._thread is not None
            if started:
                self._q.put(None)                   # inside the lock, after every request already queued
        if started:
            self._thread.join(timeout=5)
            if not self._thread.is_alive():         # still running a long batch: the worker fails the rest itself when it stops
                self._fail_pending()

    def stats(self) -> dict:
        sizes = self.batch_sizes
        return {"n_batches": len(sizes), "mean_batch_size": (sum(sizes) / len(sizes)) if sizes else None,
                "max_batch_size": max(sizes) if sizes else None}
