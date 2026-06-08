"""HTTP client to a vLLM OpenAI-compatible server — the GPU-side model under test.

vLLM serves Qwen at http://localhost:8000/v1. This sends chat requests and, for
Qwen3's reasoning mode, strips the <think>...</think> block so the scorers see
the final answer. Requests are issued concurrently so vLLM can batch them.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import httpx


class VLLMModel:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-14B",
        max_tokens: int = 6000,
        temperature: float = 0.0,
        timeout: float = 900.0,
        workers: int = 16,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.workers = workers
        self.client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str) -> str:
        try:
            r = self.client.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:  # keep the run alive; a failed call scores as wrong
            return f"[ERROR {e}]"
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    def generate_many(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self.generate, prompts))
