"""Minimal OpenAI-compatible chat client with tool calling.

Written against the wire format instead of a vendor SDK, because the same
``/chat/completions`` contract is served by OpenAI, DeepSeek, Ollama, vLLM
and LM Studio. Swapping a cloud model for a local one is then a base URL,
which matters when the logs being analysed shouldn't leave the building.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Protocol

import aiohttp

PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b"),
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ChatModel(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class OpenAICompatible:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.1, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        # Low temperature: diagnosis rewards consistency. Two runs on the same
        # symptoms should probe the same things in the same order.
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def from_preset(cls, provider: str, model: str | None = None, base_url: str | None = None):
        url, env, default_model = PRESETS[provider]
        key = os.environ.get(env) if env else None
        if env and not key:
            raise RuntimeError(f"set {env} to use the {provider} provider")
        return cls(base_url or url, model or default_model, key)

    async def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as s:
            for attempt in range(4):
                async with s.post(f"{self.base_url}/chat/completions", json=body, headers=headers) as resp:
                    if resp.status in (429, 500, 502, 503) and attempt < 3:
                        await asyncio.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                        continue
                    data = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise RuntimeError(f"model API error {resp.status}: {data}")
                    return parse_reply(data)
        raise RuntimeError("model API kept failing")


def parse_reply(data: dict) -> Reply:
    msg = data["choices"][0]["message"]
    calls = []
    for raw in msg.get("tool_calls") or []:
        fn = raw["function"]
        text = fn.get("arguments") or "{}"
        try:
            args, err = json.loads(text), None
            if not isinstance(args, dict):
                args, err = {}, "arguments must be a JSON object"
        except json.JSONDecodeError as exc:
            # Small local models sometimes emit trailing commas or prose. Keep
            # the call and report the error back so the model can retry,
            # rather than crashing the run.
            args, err = {}, f"invalid JSON arguments: {exc.msg}"
        calls.append(ToolCall(raw["id"], fn["name"], args, text, err))
    return Reply(msg.get("content") or "", calls, data.get("usage", {}))
