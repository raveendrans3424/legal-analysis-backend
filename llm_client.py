"""
llm_client.py
One place that talks to the AI model. Tries providers in order and moves to
the next one if a provider is out of quota / down / slow.

.env settings (all optional except the keys you want to use):
    LLM_PROVIDERS=groq,gemini          order to try. Add ollama as last resort:
                                       groq,gemini,ollama   (or just: ollama  = fully local/private)
    GROQ_API_KEY=...
    GROQ_MODEL=openai/gpt-oss-120b
    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-3.5-flash-lite
    OLLAMA_BASE_URL / OLLAMA_MODEL     (already in config)

PRIVACY: with groq/gemini the case text is sent to that company's servers.
Use LLM_PROVIDERS=ollama to keep everything on your own machine.
Prompts / case text are never printed to the log by this file.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60
DEFAULT_COOLDOWN = 60          # seconds to skip a provider after a 429 with no hint
MAX_COOLDOWN = 3600            # never skip a provider for more than 1 hour

# provider -> unix time until which we skip it (it said "too many requests")
_blocked_until: dict[str, float] = {}


class LLMUnavailable(Exception):
    """Every configured provider failed or is out of quota."""


class _ProviderError(Exception):
    def __init__(self, msg, cooldown=0):
        super().__init__(msg)
        self.cooldown = cooldown


def _cooldown_from(resp) -> float:
    try:
        return min(float(resp.headers.get("retry-after", DEFAULT_COOLDOWN)), MAX_COOLDOWN)
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN


def _check(resp, name):
    if resp.status_code == 429:
        raise _ProviderError(f"{name}: rate/quota limit reached", _cooldown_from(resp))
    if resp.status_code in (401, 403):
        # bad / disabled key: don't hammer it on every request
        raise _ProviderError(f"{name}: key rejected (HTTP {resp.status_code}) - check the key in .env", MAX_COOLDOWN)
    if resp.status_code >= 400:
        raise _ProviderError(f"{name}: HTTP {resp.status_code}", 30 if resp.status_code >= 500 else 0)


# ---------------- providers ----------------

def _groq(prompt: str, max_tokens: int) -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise _ProviderError("groq: GROQ_API_KEY not set", MAX_COOLDOWN)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if "gpt-oss" in model:
        # reasoning model: "thinking" tokens count against the limit, so allow extra room
        body["max_completion_tokens"] = max_tokens + 1024
        body["reasoning_effort"] = "low"
    else:
        body["max_completion_tokens"] = max_tokens
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=body, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    _check(resp, "groq")
    return (resp.json()["choices"][0]["message"].get("content") or "").strip()


def _gemini(prompt: str, max_tokens: int) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise _ProviderError("gemini: GEMINI_API_KEY not set", MAX_COOLDOWN)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},      # header, not URL, so the key never lands in logs
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            # newer Gemini models "think" first and those tokens count too -> extra room
            "generationConfig": {"maxOutputTokens": max_tokens + 1024},
        },
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    )
    _check(resp, "gemini")
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def _ollama(prompt: str, max_tokens: int) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    resp = requests.post(
        f"{base}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "keep_alive": "30m", "options": {"num_predict": max_tokens}},
        timeout=(2, 120),
    )
    _check(resp, "ollama")
    return (resp.json().get("response") or "").strip()


_PROVIDERS = {"groq": _groq, "gemini": _gemini, "ollama": _ollama}


def _provider_order() -> list[str]:
    raw = os.getenv("LLM_PROVIDERS", "groq,gemini")
    return [p.strip().lower() for p in raw.split(",") if p.strip().lower() in _PROVIDERS]


def llm_call(prompt: str, max_tokens: int = 700) -> str:
    """
    Returns the model's text. Tries each provider in LLM_PROVIDERS order.
    Raises LLMUnavailable if all fail (callers already fall back gracefully).
    """
    order = _provider_order()
    if not order:
        raise LLMUnavailable("LLM_PROVIDERS has no valid provider (use groq, gemini, ollama)")

    problems = []
    for name in order:
        if time.time() < _blocked_until.get(name, 0):
            problems.append(f"{name}: skipped (cooling down)")
            continue
        try:
            text = _PROVIDERS[name](prompt, max_tokens)
            if not text:
                raise _ProviderError(f"{name}: empty answer")
            print(f"LLM: answered by {name} ({len(text)} chars)")   # no prompt/answer text logged
            return text
        except _ProviderError as e:
            problems.append(str(e))
            if e.cooldown:
                _blocked_until[name] = time.time() + e.cooldown
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            # network error, timeout, or unexpected reply shape. Type only: no URLs/keys/text.
            problems.append(f"{name}: {type(e).__name__}")
            _blocked_until[name] = time.time() + 15

    print("LLM: all providers failed -> " + " | ".join(problems))
    raise LLMUnavailable("; ".join(problems))
