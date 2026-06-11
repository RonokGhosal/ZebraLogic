"""HTTP client to a local Ollama server — the model under test (Qwen3-14B Q4).

Talks to Ollama's native /api/chat, NOT the OpenAI-compatible /v1 endpoint:
the /v1 endpoint cannot set num_ctx per request and silently truncates the
context at the server default (~4k), which corrupts long thinking traces.

Sampling follows the Qwen3 model card for thinking mode (temperature 0.6,
top_p 0.95, top_k 20, min_p 0) — the card explicitly forbids greedy decoding
(repetition/degradation), so determinism comes from a fixed seed instead.

Ollama returns the <think> trace in message.thinking, separate from
message.content, so scorers (and referee retry prompts) see only the answer.
Every call records token counts and a `truncated` flag (done_reason !=
"stop" means the trace hit num_predict and the answer is untrustworthy).
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


class OllamaModel:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:14b",
        max_tokens: int = 32768,
        num_ctx: int = 40960,
        temperature: float = 0.6,
        seed: int = 1234,
        timeout: float = 3600.0,
        workers: int = 1,
        progress=None,
    ):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):  # tolerate old-style URLs
            base = base[:-3]
        self.url = base + "/api/chat"
        self.model = model
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.workers = workers
        self.client = httpx.Client(timeout=timeout)
        self.progress = progress
        self.calls: list[dict] = []  # one meta dict per completed call
        self.last_meta: list[dict] = []  # metas aligned with the last generate_many batch

    def _call(self, prompt: str) -> tuple[str, dict]:
        t0 = time.time()
        try:
            r = self.client.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": True,
                    "keep_alive": "60m",
                    "options": {
                        "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "seed": self.seed,
                    },
                },
            )
            r.raise_for_status()
            d = r.json()
            text = (d["message"].get("content") or "").strip()
            meta = {
                "prompt_tokens": d.get("prompt_eval_count", 0),
                "output_tokens": d.get("eval_count", 0),
                "truncated": d.get("done_reason") != "stop",
            }
            # belt-and-braces: a full window means the prompt or trace got clipped
            if meta["prompt_tokens"] + meta["output_tokens"] >= self.num_ctx:
                meta["truncated"] = True
        except Exception as e:  # keep the run alive; a failed call scores as wrong
            meta = {"prompt_tokens": 0, "output_tokens": 0, "truncated": False, "error": str(e)}
            meta["seconds"] = round(time.time() - t0, 1)
            return f"[ERROR {e}]", meta
        meta["seconds"] = round(time.time() - t0, 1)
        return text, meta

    def generate_with_meta(self, prompt: str, label: str = "") -> tuple[str, dict]:
        text, meta = self._call(prompt)
        self.calls.append(meta)
        if self.progress:
            self.progress.call_done(label or "call", meta, meta["seconds"])
        return text, meta

    def generate(self, prompt: str, label: str = "") -> str:
        return self.generate_with_meta(prompt, label)[0]

    def generate_many(self, prompts: list[str], label: str = "") -> list[str]:
        results: list[str] = [""] * len(prompts)
        metas: list[dict] = [{}] * len(prompts)
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.generate_with_meta, p, label): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i], metas[i] = fut.result()
                done += 1
                print(f"\r    {label} {done}/{len(prompts)} calls", end="", flush=True)
        sys.stdout.write("\n")
        self.last_meta = metas
        return results
