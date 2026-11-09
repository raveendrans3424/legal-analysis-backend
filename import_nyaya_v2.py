"""
import_nyaya.py - loads REAL statute text into your legal_provisions table.

Source: public dataset NyayaLabs98/nyaya-statute-db on Hugging Face (statutory
text taken from India Code; licence: public domain). Each row there has the
act, section, title, full text and a source_url.

What this script does
  1. Downloads the act files you pick (cached in nyaya_data/, only once).
  2. Drops rows that are useless for case matching: omitted / repealed /
     empty / tiny text.
  3. Asks the LLM (Groq -> Gemini, via llm_client) to put each section in ONE of
     your CASE_CATEGORIES from config.py, or "none" (definitions, procedure,
     administration ...). No keyword rules. Answers are cached in
     nyaya_data/categories.json, so a re-run never repeats finished work and
     you can continue the next day if the free quota runs out.
  4. Writes nyaya_data/review.csv  (act, section, title, category) so you / your
     sir can check what the LLM decided.
  5. Inserts into MySQL. If the same act+section already exists (rows I wrote
     earlier from memory) its text/title/source are REPLACED by the official
     text; its id and category stay the same.

Usage (venv active, in the backend folder):
  python import_nyaya.py --limit 30                   small trial spread over all acts, check review.csv
  python import_nyaya.py                              default acts, everything
  python import_nyaya.py --acts bns_2023 it_act_2000  choose acts
  python import_nyaya.py --all                        every substantive act (takes days on free quota,
                                                      but it resumes where it stopped)

Files you downloaded by hand: put the .jsonl files inside the folder  nyaya_data\
(next to this script, create it if missing) with the same names, e.g. nyaya_data\bns_2023.jsonl.
Existing files are never downloaded again.
  python import_nyaya.py --no-db                      only classify + CSV, no DB writes

Afterwards ALWAYS rebuild the search index:  python build_faiss_index.py

PRIVACY: only law text goes to the LLM here - no case data.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests

from config import CASE_CATEGORIES

BASE_URL = "https://huggingface.co/datasets/NyayaLabs98/nyaya-statute-db/resolve/main/"
DATA_DIR = "nyaya_data"
CATEGORY_CACHE = os.path.join(DATA_DIR, "categories.json")
REVIEW_CSV = os.path.join(DATA_DIR, "review.csv")

# Acts that matter for criminal / family / cyber case matching.
# (more files exist in the dataset: bnss_2023, bsa_2023, jj_act_2015, ... see its page)
DEFAULT_ACTS = ["bns_2023", "it_act_2000", "mv_act_1988", "pocso_2012",
                "dv_act_2005", "dpa_1961", "ni_act_1881", "posh_2013"]

# More substantive acts in the same dataset (use --all to include them).
# Skipped on purpose: bnss_2023 / bsa_2023 / limitation_1963 / lsa_1987 (procedure & evidence, mostly "none"),
# law_mappings and procedures_kb (different file format).
MORE_ACTS = ["jj_act_2015", "cpa_2019", "contract_act_1872", "tpa_1882", "hma_1955", "sma_1954",
             "hsa_1956", "hmga_1956", "hama_1956", "gwa_1890", "isa_1925", "mwpsc_2007",
             "rti_2005", "wages_code_2019", "css_2020", "constitution_1950"]

MIN_TEXT_CHARS = 40
BATCH_SIZE = 15
PAUSE_BETWEEN_CALLS = 20         # seconds (Groq free tier: ~8,000 tokens per minute)
MAX_TEXT_STORED = 6000           # characters kept per section in the DB
SNIPPET_FOR_LLM = 350

OMITTED_RE = re.compile(r"^\W*\[?\s*(omitted|repealed|deleted)\b", re.I)


# ---------------- download / parse ----------------

def fetch_file(name: str) -> str:
    """Downloads <name>.jsonl once into nyaya_data/ and returns the local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"{BASE_URL}{name}.jsonl"
    print(f"downloading {url}")
    r = requests.get(url, timeout=(10, 120))
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def read_sections(path: str) -> list[dict]:
    """One dict per (act, section). Rows that are split into sub-parts are joined."""
    merged: dict[tuple, dict] = {}
    order = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            section = str(row.get("section") or "").strip()
            act = (row.get("act_name") or "").strip()
            text = (row.get("text") or "").strip()
            if not section or not act:
                continue
            key = (act, section)
            if key not in merged:
                merged[key] = {"act_name": act, "section_number": section,
                               "title": (row.get("title") or "").strip(),
                               "text": text, "source_url": row.get("source_url") or ""}
                order.append(key)
            elif text and text not in merged[key]["text"]:
                merged[key]["text"] += "\n" + text
    return [merged[k] for k in order]


def is_useful(sec: dict) -> bool:
    text, title = sec["text"], sec["title"]
    if len(text) < MIN_TEXT_CHARS:
        return False
    if OMITTED_RE.match(text) or OMITTED_RE.match(title):
        return False
    return True


# ---------------- LLM categorisation ----------------

PROMPT = """You are helping organise Indian statute sections for a legal case-matching tool.
For EACH numbered section below choose ONE category from this list, or "none":
{labels}

Use "none" for definitions, general principles, procedure, courts, administration,
rule-making powers, and anything that does not describe an offence or right that a
person could be accused of / harmed by.
Reply with ONLY a JSON object mapping the number to the label, e.g. {{"1": "theft", "2": "none"}}.
No explanation.

Sections:
{items}
"""


def _key(sec: dict) -> str:
    return f"{sec['act_name']}|{sec['section_number']}"


def classify_batch(batch: list[dict], llm_call) -> dict[str, str]:
    items = "\n".join(
        f"{i + 1}. [{s['act_name']} s.{s['section_number']}] {s['title']} :: "
        f"{s['text'][:SNIPPET_FOR_LLM].replace(chr(10), ' ')}"
        for i, s in enumerate(batch)
    )
    prompt = PROMPT.format(labels="\n".join(f"- {c}" for c in CASE_CATEGORIES), items=items)
    raw = llm_call(prompt, max_tokens=700)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON in LLM answer")
    data = json.loads(m.group(0))
    allowed = {c.lower(): c for c in CASE_CATEGORIES}
    out = {}
    for i, s in enumerate(batch):
        label = str(data.get(str(i + 1), "none")).strip().lower()
        out[_key(s)] = allowed.get(label, "none")
    return out


def load_cache() -> dict:
    if os.path.exists(CATEGORY_CACHE):
        with open(CATEGORY_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CATEGORY_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)


def categorise(sections: list[dict], llm_call, llm_error, sleep=time.sleep) -> dict:
    cache = load_cache()
    todo = [s for s in sections if _key(s) not in cache]
    print(f"categories: {len(sections) - len(todo)} cached, {len(todo)} to ask the LLM")
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        for attempt in range(4):
            try:
                cache.update(classify_batch(batch, llm_call))
                break
            except llm_error as e:
                print(f"  AI busy / out of quota ({str(e)[:80]}...) waiting 65s [{attempt + 1}/4]")
                sleep(65)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  unusable answer ({e}); retrying")
                sleep(3)
        else:
            save_cache(cache)
            print("Stopped: AI quota is finished for now. Progress is saved - run the same command later.")
            return cache
        save_cache(cache)
        print(f"  classified {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")
        sleep(PAUSE_BETWEEN_CALLS)
    return cache


# ---------------- database ----------------

def norm_act(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def write_to_db(rows: list[dict]) -> tuple[int, int]:
    from db import SessionLocal
    from models import LegalProvision
    db = SessionLocal()
    inserted = updated = 0
    try:
        existing = {}
        for p in db.query(LegalProvision).all():
            existing[(norm_act(p.act_name or ""), (p.section_number or "").strip().lower())] = p
        for r in rows:
            k = (norm_act(r["act_name"]), r["section_number"].strip().lower())
            p = existing.get(k)
            if p:
                p.title = r["title"][:500]
                p.full_text = r["text"][:MAX_TEXT_STORED]
                p.source_url = r["source_url"][:500]
                if not p.category:
                    p.category = r["category"]
                updated += 1
            else:
                p = LegalProvision(act_name=r["act_name"][:255], section_number=r["section_number"][:50],
                                   title=r["title"][:500], full_text=r["text"][:MAX_TEXT_STORED],
                                   category=r["category"], source_url=r["source_url"][:500])
                db.add(p)
                existing[k] = p
                inserted += 1
        db.commit()
    finally:
        db.close()
    return inserted, updated


def write_review(sections, cache) -> str:
    """Writes review.csv. If Excel has it open (Windows locks it) use a new file name instead."""
    base, ext = os.path.splitext(REVIEW_CSV)
    for path in (REVIEW_CSV, f"{base}_{int(time.time())}{ext}"):
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["act", "section", "title", "category"])
                for s in sections:
                    w.writerow([s["act_name"], s["section_number"], s["title"], cache.get(_key(s), "(not classified yet)")])
            if path != REVIEW_CSV:
                print("review.csv is open in another program (Excel?) - saved as a new file instead.")
            return path
        except PermissionError:
            continue
    raise PermissionError("Close review.csv in Excel and run again.")


# ---------------- main ----------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="+", default=None, help="file names without .jsonl")
    ap.add_argument("--all", action="store_true", help="default acts + all the other substantive acts")
    ap.add_argument("--limit", type=int, default=0, help="only first N useful sections (trial run)")
    ap.add_argument("--no-db", action="store_true", help="classify + write review.csv only")
    args = ap.parse_args(argv)
    acts = args.acts or (DEFAULT_ACTS + MORE_ACTS if args.all else DEFAULT_ACTS)

    from llm_client import llm_call, LLMUnavailable

    sections = []
    for act in acts:
        try:
            rows = read_sections(fetch_file(act))
        except requests.RequestException as e:
            print(f"SKIP {act}: download failed ({type(e).__name__}). Check the file name / internet.")
            continue
        useful = [s for s in rows if is_useful(s)]
        if not rows:
            print(f"  {act}: no sections found (different file layout?) - skipped")
        print(f"{act}: {len(rows)} sections, {len(useful)} useful (skipped {len(rows) - len(useful)} omitted/empty)")
        sections += useful
    if args.limit and len(sections) > args.limit:
        # trial run: spread the sample over ALL sections/acts, not just the first few
        step = len(sections) / args.limit
        sections = [sections[int(i * step)] for i in range(args.limit)]
    if not sections:
        print("Nothing to do.")
        return

    cache = categorise(sections, llm_call, LLMUnavailable)

    final = []
    for s in sections:
        cat = cache.get(_key(s))
        if cat and cat != "none":
            final.append({**s, "category": cat})
    os.makedirs(DATA_DIR, exist_ok=True)
    review_path = write_review(sections, cache)
    print(f"review file: {review_path}  ({len(final)} sections will be added, "
          f"{sum(1 for s in sections if cache.get(_key(s)) == 'none')} marked none)")

    if args.no_db:
        return
    inserted, updated = write_to_db(final)
    print(f"DB done: {inserted} new, {updated} existing rows replaced with official text.")
    counts = {}
    for s in final:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print("NEXT: python build_faiss_index.py")


if __name__ == "__main__":
    main()
