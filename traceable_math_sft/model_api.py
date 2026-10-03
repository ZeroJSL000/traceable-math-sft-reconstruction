"""Optional OpenAI-compatible local API client for dual sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen


class Sampler(Protocol):
    def sample(self, prompt: str, temperature: float) -> str: ...


def build_prompt(question: str, source_unit: str) -> str:
    return (
        "Answer the single math question using only the source excerpt. "
        "Do not invent facts missing from the excerpt. Give concise reasoning and "
        "end with one final answer in \\boxed{...}.\n\n"
        f"Question:\n{question}\n\nSource excerpt:\n{source_unit}"
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenAICompatibleSampler:
    endpoint: str
    model: str
    timeout_seconds: float = 60.0
    max_tokens: int = 384

    def sample(self, prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": "You are a careful math tutor."},
                {"role": "user", "content": prompt},
            ],
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.load(response)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("API response is missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise ValueError("API response content must be a string")
        return content
