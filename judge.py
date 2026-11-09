"""
judge.py
An "AI Judge" persona: reacts critically to the user's arguments during a
mock hearing (pushes back, asks probing questions, doesn't just agree),
then delivers a structured verdict summarizing where the user's argument
was strong or weak.
"""

from llm_client import llm_call

JUDGE_SYSTEM_PROMPT = """You are a strict, no-nonsense judge presiding over a courtroom argument.
You are NOT a friendly assistant — you are skeptical, probing, and hold the
speaker to a high standard of evidence and logic, the way a real judge
challenges a lawyer's arguments in court.

Rules:
- Respond in 2-4 sentences, in character as the judge, addressing the user as "counsel" or "you".
- Push back on weak, vague, or unsupported claims. Ask pointed follow-up questions.
- Do not simply agree or praise; find holes in the reasoning if they exist.
- Stay respectful and professional, never insulting.
- Do not break character or mention that you are an AI.
"""

JUDGE_SUMMARY_PROMPT = """You are the judge concluding this hearing. Based on the full exchange
below, give a structured verdict with these exact headings:

1. Summary of the argument presented
2. Strengths in the argument
3. Weaknesses / gaps — specifically where the arguer failed to support
   claims, contradicted themselves, dodged questions, or lacked evidence
4. Overall verdict on how persuasive the argument was (rate: Weak / Moderate / Strong)

Be specific and refer to actual points made in the conversation below.
Do not invent claims that were not made.

Conversation:
{transcript}
"""


def _call_ollama(prompt: str, timeout=None, max_tokens: int = 500) -> str:
    """Name kept for compatibility; now uses llm_client (Groq -> Gemini -> optional Ollama)."""
    return llm_call(prompt, max_tokens=max_tokens)


def _build_transcript(history: list[dict]) -> str:
    return "\n".join(
        f"{'Counsel' if h.get('role') == 'user' else 'Judge'}: {h.get('text', '')}"
        for h in history
    )


def judge_reply(history: list[dict], user_message: str) -> str:
    """
    history: list of {"role": "user"|"judge", "text": "..."}
    Returns the judge's next in-character reply.
    """
    transcript = _build_transcript(history)
    prompt = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"Counsel: {user_message}\nJudge:"
    )
    try:
        reply = _call_ollama(prompt)
        return reply if reply else "The court requires a clearer statement, counsel. Please restate your argument."
    except Exception:
        return (
            "The court is unable to respond at this moment "
            "(the AI service is unreachable or out of quota — try again in a minute)."
        )


def judge_summary(history: list[dict]) -> str:
    """
    Produces the final structured verdict once the user ends the debate.
    """
    transcript = _build_transcript(history)
    prompt = JUDGE_SUMMARY_PROMPT.format(transcript=transcript)
    try:
        return _call_ollama(prompt, max_tokens=900)
    except Exception:
        return "The court could not prepare a verdict at this time (the AI service is unreachable or out of quota)."
