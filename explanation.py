"""
explanation.py
Generates explanations for ALL top-ranked provisions in a SINGLE LLM call
(instead of one call per provision) — this is the main speed fix.
"""

import re
from fact_analysis import llm_call

BATCH_PROMPT = """Case facts: {case_facts}

Below are {n} candidate legal provisions. For EACH one, in 1-2 sentences,
explain whether and why it applies to these case facts, referencing the
specific facts that match (or, if it does not clearly apply, say plainly
that it is only a loose match rather than forcing a justification).
Do not invent facts that are not present above.

Respond in EXACTLY this format, one block per provision, nothing else:
### 1
<explanation>
### 2
<explanation>
(continue for all {n})

Provisions:
{provisions_block}
"""


def generate_explanations_batch(case_facts: str, provisions: list[dict]) -> list[str]:
    if not provisions:
        return []

    provisions_block = "\n".join(
        f"{i+1}. {p.get('act_name','')} Section {p.get('section_number','')} - "
        f"{p.get('title','')}: {(p.get('full_text') or '')[:400]}"
        for i, p in enumerate(provisions)
    )
    prompt = BATCH_PROMPT.format(
        case_facts=case_facts[:2000],
        n=len(provisions),
        provisions_block=provisions_block,
    )

    fallback = [
        "Matched based on semantic similarity to the case facts."
        for _ in provisions
    ]

    try:
        raw = llm_call(prompt)
    except Exception:
        return fallback

    blocks = re.split(r"###\s*\d+\s*\n?", raw)
    blocks = [b.strip() for b in blocks if b.strip()]

    if len(blocks) != len(provisions):
        return fallback

    return blocks