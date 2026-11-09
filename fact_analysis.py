"""
fact_analysis.py
llm_call() now goes through llm_client (Groq -> Gemini -> optional Ollama).
Other files still do `from fact_analysis import llm_call`, so nothing else
needs to change. extract_facts is no longer used by main.py but is kept.
"""

import json

from llm_client import llm_call as _llm_call, LLMUnavailable  # noqa: F401

FACT_EXTRACTION_PROMPT = """
Extract the following from this legal case text as JSON with these exact keys:
- parties_involved
- key_actions
- intent_or_mens_rea_indicators
- circumstances
- outcome_or_harm

Respond with ONLY valid JSON, no extra commentary.

Case text:
{text}
"""


def llm_call(prompt: str) -> str:
    return _llm_call(prompt, max_tokens=700)


def extract_facts(text: str) -> str:
    prompt = FACT_EXTRACTION_PROMPT.format(text=text[:6000])
    try:
        raw = llm_call(prompt)
        json.loads(raw)
        return raw
    except Exception:
        return text[:2000]
