"""
judgment_utils.py
Shared helpers for judgments: finding "Section 302 of the Indian Penal Code" style
citations inside judgment text, and converting old IPC sections to the matching
BNS sections (the BNS dataset rows list the IPC sections they replace).

This is text parsing of the judgments, not a keyword -> law lookup used to
choose answers for a user.
"""

import json
import re

# alias in judgment text  ->  normalised act-name prefix (see norm_act)
ACT_ALIASES = [
    (r"indian\s+penal\s+code|i\.?\s?p\.?\s?c\.?", "ipc"),
    (r"bharatiya\s+nyaya\s+sanhita|b\.?\s?n\.?\s?s\.?", "bharatiyanyayasanhita"),
    (r"negotiable\s+instruments?\s+act|n\.?\s?i\.?\s+act", "negotiableinstrumentsact"),
    (r"information\s+technology\s+act|i\.?\s?t\.?\s+act", "informationtechnologyact"),
    (r"motor\s+vehicles?\s+act|m\.?\s?v\.?\s+act", "motorvehiclesact"),
    (r"protection\s+of\s+children\s+from\s+sexual\s+offences\s+act|pocso\s+act", "protectionofchildrenfromsexualoffences"),
    (r"dowry\s+prohibition\s+act", "dowryprohibitionact"),
    (r"protection\s+of\s+women\s+from\s+domestic\s+violence\s+act|domestic\s+violence\s+act", "protectionofwomenfromdomesticviolenceact"),
]

_SEC = r"\d+[A-Za-z]{0,2}(?:\s*\([0-9a-zA-Z]{1,3}\))*"
_LIST = rf"{_SEC}(?:\s*(?:,|/|&|and|to|r/w|read\s+with)\s*(?:sections?\s+)?{_SEC})*"
_ACT = "|".join(f"(?:{a})" for a, _ in ACT_ALIASES)
CITE_RE = re.compile(
    rf"(?:sections?|secs?\.?|ss\.|s\.|u/s\.?)\s*(?P<secs>{_LIST})\s*(?:of\s+(?:the\s+)?|,\s*)?(?P<act>{_ACT})",
    re.I,
)
_ALIAS_RES = [(re.compile(a, re.I), key) for a, key in ACT_ALIASES]


def norm_act(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", (name or "").lower())
    return re.sub(r"[^a-z0-9]", "", name)


def _act_key(text: str) -> str | None:
    for rx, key in _ALIAS_RES:
        if rx.fullmatch(text.strip()):
            return key
    return None


def _split_sections(raw: str) -> list[str]:
    out = []
    for m in re.finditer(_SEC, raw):
        num = re.sub(r"\s*\(.*", "", m.group(0)).strip().upper()
        if num and num not in out:
            out.append(num)
    return out


def find_citations(text: str) -> set[tuple[str, str]]:
    """Returns {(act_key, section_number)}; act_key 'ipc' means an old IPC section."""
    found = set()
    for m in CITE_RE.finditer(text or ""):
        key = _act_key(m.group("act"))
        if not key:
            continue
        for sec in _split_sections(m.group("secs")):
            if len(sec) <= 6:
                found.add((key, sec))
    return found


def load_ipc_to_bns(jsonl_path: str) -> dict[str, list[str]]:
    """Reads nyaya_data/bns_2023.jsonl: each BNS row lists the IPC sections it replaces ('IPC 378')."""
    mapping: dict[str, list[str]] = {}
    try:
        with open(jsonl_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bns = str(row.get("section") or "").strip().upper()
                for rep in row.get("replaces") or []:
                    m = re.match(r"\s*IPC\s+(\d+[A-Za-z]{0,2})\b", str(rep), re.I)
                    if m and bns:
                        lst = mapping.setdefault(m.group(1).upper(), [])
                        if bns not in lst:
                            lst.append(bns)
    except OSError:
        pass
    return mapping


def resolve_citations(citations: set[tuple[str, str]], ipc_to_bns: dict) -> set[tuple[str, str]]:
    """Old IPC citations become BNS sections; everything else is kept as is."""
    out = set()
    for key, sec in citations:
        if key == "ipc":
            for b in ipc_to_bns.get(sec, []):
                out.add(("bharatiyanyayasanhita", b))
        else:
            out.add((key, sec))
    return out


KNOWN_ACT_KEYS = [key for _, key in ACT_ALIASES if key != "ipc"]


def act_key_for(act_name: str) -> str | None:
    """Maps a legal_provisions.act_name to the act_key used in judgment_sections."""
    n = norm_act(act_name)
    for key in KNOWN_ACT_KEYS:
        if n.startswith(key):
            return key
    return None
