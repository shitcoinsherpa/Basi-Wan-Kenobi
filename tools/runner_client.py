"""Persistent-worker client for the BASIWAN runner --serve mode.

Owns the subprocess lifecycle, a background reader thread that parses
[BASIWAN-EVENT] JSON lines, and a per-request Queue dispatch table keyed by
the request id. Consumers iterate `generate(req_id, args)` to receive
events as they arrive; the iterator terminates on a {"event":"result"} or
{"event":"error","fatal":true}.

Threading model:
  - One BasiwanRunner instance owns one subprocess + one reader thread.
  - The reader thread tags each event with its `id` field (or routes
    unsolicited events like "ready" to a special inbox).
  - generate() is itself thread-safe to call from a single Gradio worker
    thread — each request gets a fresh Queue dispatched by id.

Restart policy:
  - The runner emits {"event":"error","fatal":true} then exits(2) on
    CUDA-context corruption. The client surfaces RunnerDied and the next
    generate() call will respawn the worker (one-budget; second crash
    surfaces to caller without respawn).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable


class RunnerDied(RuntimeError):
    """Raised when the worker subprocess died unexpectedly."""


class BasiwanRunner:
    """Persistent worker handle. Spawns runner --serve once, dispatches N requests."""

    def __init__(self, *, python: str, runner_script: str,
                 cli_args: list[str], env: dict, cwd: str,
                 ready_timeout_s: int = 1800):
        # ready_timeout_s=1800 (was 900): a fresh install's FIRST start
        # re-packs both 14B experts from GGUF and writes ~19 GB of pack
        # cache — measured, which
        # would have killed a healthy worker mid-warm. Cached starts are
        # minutes, not seconds, only on the first ever run.
        self._python = python
        self._runner = runner_script
        self._cli_args = list(cli_args)
        self._env = dict(env)
        self._cwd = cwd
        self._ready_timeout_s = ready_timeout_s

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        # id → Queue of events. Unsolicited events (ready, pong) land in
        # self._global_inbox.
        self._mailboxes: dict[str, queue.Queue] = {}
        self._mailboxes_lock = threading.Lock()
        self._global_inbox: queue.Queue = queue.Queue()
        # Single-request serialization: phase events from text2video.py don't
        # carry an `id`, but we know they belong to the currently-active
        # request because the worker processes one at a time. Track it.
        self._active_id: str | None = None
        self._died = threading.Event()
        # Restart budget — start() can be called twice (initial + 1 respawn).
        self._respawned = False
        # Bytes-mode lock so concurrent generate() calls serialize on stdin.
        self._stdin_lock = threading.Lock()
        # Last 200 human-readable runner lines for debug surface on failure.
        self._human_tail: list[str] = []
        self._human_tail_max = 200

    # --- lifecycle ---

    def start(self, progress_cb=None) -> dict:
        """Spawn the worker and block until {"event":"ready"} is received.

        Returns the ready event dict, e.g. {"resident_gb": 1.91, ...}.

        progress_cb, if given, is called roughly once per second with
        (elapsed_s, last_human_line) so a UI can show live load progress
        instead of freezing for the multi-minute cold start.

        Raises RunnerDied if the worker exits before ready, or if ready
        doesn't arrive within ready_timeout_s.
        """
        cmd = [self._python, "-u", self._runner, *self._cli_args, "--serve"]
        # JSON-only output makes the parser's life easier — the human-readable
        # lines still flow but the IPC client never sees them via stdin pipe
        # because they go through the runner's own stdout (we tee them).
        # We KEEP both for now: the reader thread filters [BASIWAN-EVENT].
        self._proc = subprocess.Popen(
            cmd, env=self._env, cwd=self._cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="basiwan-runner-reader", daemon=True)
        self._reader_thread.start()

        t_start = time.time()
        deadline = t_start + self._ready_timeout_s
        while time.time() < deadline:
            try:
                ev = self._global_inbox.get(timeout=1.0)
            except queue.Empty:
                if self._proc.poll() is not None:
                    self._died.set()
                    raise RunnerDied(
                        f"worker exited before ready (code={self._proc.returncode})")
                if progress_cb is not None:
                    tail = self._human_tail[-1] if self._human_tail else ""
                    try:
                        progress_cb(time.time() - t_start, tail)
                    except Exception:
                        pass  # UI callback must never kill the warm-up
                continue
            if ev.get("event") == "ready":
                return ev
        # Kill the worker before raising — without this, the timed-out
        # (but otherwise healthy) worker stays orphaned holding ~27 GB
        # host RAM, and the caller's retry spawns a SECOND one on top.
        # Observed live when first-ever cold start exceeded
        # the old 900s budget mid low-noise pack.
        try:
            self._proc.kill()
        except Exception:
            pass
        self._died.set()
        raise RunnerDied(f"worker did not emit ready in {self._ready_timeout_s}s")

    def shutdown(self, timeout: float = 30.0) -> int:
        """Send {"cmd":"shutdown"} and wait for clean exit. Returns exit code."""
        if self._proc is None or self._proc.poll() is not None:
            return -1
        try:
            self._send({"cmd": "shutdown"})
            self._proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            return self._proc.wait(timeout=5)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # --- request ---

    def generate(self, req_id: str | None, args: dict) -> Iterable[dict]:
        """Submit one generate request, yield events until terminal.

        Terminal events: {"event":"result"} or {"event":"error","fatal":...}.
        Caller iterates; the generator terminates on the terminal event.

        Raises RunnerDied if the worker dies mid-request.
        """
        if not self.is_alive():
            raise RunnerDied("worker is not alive")
        if req_id is None:
            req_id = str(uuid.uuid4())
        # Allocate mailbox before sending so we can't miss the first event.
        mb: queue.Queue = queue.Queue()
        with self._mailboxes_lock:
            self._mailboxes[req_id] = mb

        try:
            # Mark this request as the active one BEFORE sending. The reader
            # routes id-less events (e.g. text2video.py phase markers) to the
            # active mailbox so the iterator sees them in order.
            self._active_id = req_id
            self._send({"cmd": "generate", "id": req_id, "args": args})
            while True:
                try:
                    ev = mb.get(timeout=1.0)
                except queue.Empty:
                    if not self.is_alive():
                        raise RunnerDied(
                            f"worker died during request id={req_id} "
                            f"(code={self._proc.returncode if self._proc else None})")
                    continue
                yield ev
                kind = ev.get("event")
                if kind == "result":
                    return
                if kind == "error":
                    if ev.get("fatal"):
                        # Worker will exit(2); surface as dead so caller can respawn
                        self._died.set()
                    return
        finally:
            self._active_id = None
            with self._mailboxes_lock:
                self._mailboxes.pop(req_id, None)

    def ping(self) -> dict:
        """Send a ping; return the pong event. Useful for liveness checks."""
        ping_id = str(uuid.uuid4())
        mb: queue.Queue = queue.Queue()
        with self._mailboxes_lock:
            self._mailboxes[ping_id] = mb
        try:
            self._send({"cmd": "ping", "id": ping_id})
            try:
                return mb.get(timeout=5.0)
            except queue.Empty:
                raise RunnerDied("worker did not respond to ping in 5s")
        finally:
            with self._mailboxes_lock:
                self._mailboxes.pop(ping_id, None)

    def set_lora(self, lora_dir, strength: float = 1.0,
                 timeout: float = 60.0, runtime_scalable: bool = False) -> dict:
        """Hot-swap the user-LoRA combo on the live worker. lora_dir
        is a directory holding {low,high}_noise_model.safetensors, or None to
        clear all LoRA. Returns the worker's lora_set event (cleared/low/high/
        wall_s). Modeled on ping(); 60 s timeout covers a load_file + attach
        (measured ~0.3 s, but a cold OS cache on the combo file can stall).

        runtime_scalable=True marks the combo as USER-ONLY (no Lightning
        entangled), so request-time lora_strength scales it exactly and the
        worker honors the slider instead of pinning to 1.0. Leave False for a
        Lightning+user combo (strength must be baked at build there)."""
        set_id = str(uuid.uuid4())
        mb: queue.Queue = queue.Queue()
        with self._mailboxes_lock:
            self._mailboxes[set_id] = mb
        try:
            self._send({"cmd": "set_lora", "id": set_id,
                        "args": {"dir": (str(lora_dir) if lora_dir else None),
                                 "strength": float(strength),
                                 "runtime_scalable": bool(runtime_scalable)}})
            try:
                ev = mb.get(timeout=timeout)
            except queue.Empty:
                raise RunnerDied(f"worker did not ack set_lora in {timeout}s")
            if ev.get("event") == "error":
                raise RunnerDied(f"set_lora failed: {ev.get('msg')}")
            return ev
        finally:
            with self._mailboxes_lock:
                self._mailboxes.pop(set_id, None)

    # --- internals ---

    def _send(self, msg: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RunnerDied("worker stdin closed")
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        with self._stdin_lock:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def _read_loop(self) -> None:
        """Background thread: parse each stdout line, route JSON events.

        Routing policy:
          - Line is [BASIWAN-EVENT] {...json}:
              - If event has `id`: route to that mailbox.
              - Else if a request is active (`self._active_id` set): route to
                that mailbox. This captures the text2video.py phase markers
                which don't carry an id field but ARE generated within the
                active request's lifetime (worker is single-threaded).
              - Else: drop into global_inbox.
          - Otherwise (human-readable): retain in a rolling tail buffer for
            debug surface on failure.
        """
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            if "[BASIWAN-EVENT]" not in line:
                # Retain human-readable lines for debug surface.
                self._human_tail.append(line)
                if len(self._human_tail) > self._human_tail_max:
                    del self._human_tail[: len(self._human_tail) - self._human_tail_max]
                continue
            try:
                payload = line.split("[BASIWAN-EVENT] ", 1)[1]
                ev = json.loads(payload)
            except Exception:
                continue
            ev_id = ev.get("id")
            target_id = ev_id if ev_id is not None else self._active_id
            if target_id is not None:
                with self._mailboxes_lock:
                    mb = self._mailboxes.get(target_id)
                if mb is not None:
                    mb.put(ev)
                    continue
            self._global_inbox.put(ev)
        # Stdout EOF: worker exited.
        self._died.set()
        # Wake any waiting generate() iterators.
        with self._mailboxes_lock:
            for mb in self._mailboxes.values():
                mb.put({"_basiwan_event": True, "event": "_EOF"})

    def get_human_tail(self, n: int = 50) -> list[str]:
        """Return the last n human-readable runner stdout lines (for debug)."""
        return self._human_tail[-n:] if n > 0 else list(self._human_tail)
