"""
ingest_judgments.py - one-time (and resumable) import of Supreme Court judgments.

Source: "Indian Supreme Court Judgments" open dataset (eCourts, AWS Open Data, CC-BY-4.0)
  https://indian-supreme-court-judgments.s3.amazonaws.com/

What it does (run on YOUR laptop, results go into the same MySQL the app uses):
  1. downloads the small per-year metadata files (parquet)
  2. picks judgments with an English PDF, spread evenly over the years (default: 2010-2025, 5000 in total)
  3. downloads each PDF only long enough to read it (PDFs are NOT kept), reads the text,
     keeps a short snippet and finds "Section X of the IPC / NI Act / IT Act ..." citations
     (old IPC sections are converted to the matching BNS section)
  4. saves metadata + snippet + citations in MySQL (tables judgments, judgment_sections)
  5. builds the search index  faiss_index/judgments.index  + judgments_id_map.json

Usage (venv active, backend folder):
  pip install pandas pyarrow
  python ingest_judgments.py --inspect                 CHECK FIRST: prints columns + tests a few PDF links
  python ingest_judgments.py --limit 50                small trial
  python ingest_judgments.py                           full pilot (5000), safe to stop and run again
  python ingest_judgments.py --rebuild-index-only      only rebuild the FAISS files

AFTER HOSTING: copy faiss_index/judgments.index and judgments_id_map.json to the server together
with the backend; the judgments tables must exist in the server's MySQL (run this script
against that database once, or export/import the tables).
"""

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com/"
DATA_DIR = "judgments_data"
SNIPPET_CHARS = 3000
MAX_PAGES = 25
MIN_TEXT_CHARS = 500           # fewer characters = scanned PDF, skipped (no OCR here)
HEADERS = {"User-Agent": "lexindia-research-ingest/1.0 (+student project)"}


# ---------------- helpers ----------------

def http_get(url: str, tries: int = 3, timeout=(10, 90)) -> bytes | None:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def meta_path(year: int) -> str:
    return os.path.join(DATA_DIR, f"meta_{year}.parquet")


def load_year(year: int):
    import pandas as pd
    path = meta_path(year)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = http_get(f"{BASE}metadata/parquet/year={year}/metadata.parquet")
        if data is None:
            return None
        with open(path, "wb") as f:
            f.write(data)
    return pd.read_parquet(path)


ENGLISH_WORDS = {"english", "en", "eng", "en-in", "en_in"}


def language_tokens(val) -> list[str]:
    """available_languages may be a list, a numpy array, or a string like "['English']" or 'en,hi'."""
    if val is None:
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    text = " ".join(map(str, val)) if isinstance(val, (list, tuple, set)) else str(val)
    return [t for t in re.split(r"[^a-z_-]+", text.lower()) if t and t not in ("nan", "none")]


def has_english(val) -> bool:
    """True if English is listed, or if no language information is given at all."""
    toks = language_tokens(val)
    return not toks or any(t in ENGLISH_WORDS for t in toks)


def clean(val) -> str:
    s = "" if val is None else str(val)
    return "" if s.lower() in ("nan", "none", "nat") else s.strip()


def list_pdf_keys(year: int) -> dict[str, str]:
    """
    Lists the English PDFs of one year from the public bucket and returns
    {name-without-extension: full key}, plus the same name without a language suffix
    (so 2019_9_593_605_EN.pdf can be found from the metadata path 2019_9_593_605).
    Cached in judgments_data/keys_<year>.json.
    """
    import html
    cache = os.path.join(DATA_DIR, f"keys_{year}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    prefix = f"data/pdf/year={year}/english/"
    keys, token = [], None
    for _ in range(200):                                    # safety limit
        url = f"{BASE}?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            from urllib.parse import quote
            url += "&continuation-token=" + quote(token, safe="")
        data = http_get(url)
        if data is None:
            break
        xml = data.decode("utf-8", "replace")
        keys += [html.unescape(k) for k in re.findall(r"<Key>(.*?)</Key>", xml)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if "<IsTruncated>true</IsTruncated>" in xml and m:
            token = html.unescape(m.group(1))
        else:
            break
    mapping: dict[str, str] = {}
    for k in keys:
        if not k.lower().endswith(".pdf"):
            continue
        stem = os.path.basename(k)[:-4]
        mapping[stem] = k
        mapping.setdefault(re.sub(r"[_-](EN|ENG|ENGLISH)$", "", stem, flags=re.I), k)
    if mapping:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
    return mapping


GUESS_SUFFIXES = ["", "_EN", "_ENG", "_english"]


def resolve_pdf_keys(row, year: int, listing: dict[str, str]) -> list[str]:
    """Candidate S3 keys for one metadata row (exactly one when the bucket listing is available)."""
    p = os.path.basename(clean(row.get("path")).replace("\\", "/"))
    p = re.sub(r"\.(pdf|json)$", "", p, flags=re.I)
    if not p:
        return []
    if listing:
        k = listing.get(p)
        return [k] if k else []
    return [f"data/pdf/year={year}/english/{p}{suf}.pdf" for suf in GUESS_SUFFIXES]


def extract_text(pdf_bytes: bytes) -> str:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts = []
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                break
            parts.append(page.get_text())
        return "\n".join(parts)
    finally:
        doc.close()


def make_snippet(text: str) -> str:
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"\bJ\s?U\s?D\s?G\s?M\s?E\s?N\s?T\b", flat[:6000])
    start = m.end() if m else 0
    return flat[start:start + SNIPPET_CHARS].strip()


# ---------------- inspect ----------------

def inspect(year: int):
    from collections import Counter
    df = load_year(year)
    if df is None:
        print(f"Could not download metadata for {year}. Check internet / try another year.")
        return
    print(f"year {year}: {len(df)} rows")
    print("columns:", list(df.columns))
    if "available_languages" in df.columns:
        vals = Counter(repr(v.tolist() if hasattr(v, "tolist") else v) for v in df["available_languages"])
        print("available_languages values (top 8):", vals.most_common(8))
    else:
        print("no available_languages column")
    listing = list_pdf_keys(year)
    print(f"bucket listing: {len(listing)} English PDF names found for {year}")
    if listing:
        print("example file keys:", list(listing.values())[:3])
    else:
        print("could not list the bucket (will try guessed names)")
    sample = df.head(3)
    ok = 0
    for _, row in sample.iterrows():
        cands = resolve_pdf_keys(row, year, listing)
        print("-" * 60)
        print("title   :", clean(row.get("title"))[:100])
        print("path    :", clean(row.get("path")))
        print("citation:", clean(row.get("citation")))
        print("pdf key candidates:", cands)
        for key in cands:
            try:
                r = requests.head(BASE + key, headers=HEADERS, timeout=20)
                print("  link check:", key.split("/")[-1], "->", r.status_code, "(200 = good)")
                if r.status_code == 200:
                    ok += 1
                    break
            except requests.RequestException as e:
                print("  link check failed:", type(e).__name__)
    print("=" * 60)
    print("RESULT:", "pdf links work" if ok else "PDF LINKS DID NOT WORK - paste this whole output to Claude")


# ---------------- main ingest ----------------

def process_one(row, year: int, ipc_to_bns: dict, listing: dict):
    """Worker thread: download + read one judgment. Returns a dict or a reason string."""
    from judgment_utils import find_citations, resolve_citations
    candidates = resolve_pdf_keys(row, year, listing)
    if not candidates:
        return "no_pdf_found"
    key, pdf = None, None
    for cand in candidates:
        pdf = http_get(BASE + cand)
        if pdf is not None:
            key = cand
            break
    if pdf is None:
        return "download_failed"
    try:
        text = extract_text(pdf)
    except Exception:
        return "unreadable_pdf"
    if len(text.strip()) < MIN_TEXT_CHARS:
        return "scanned_or_empty"
    cites = resolve_citations(find_citations(text), ipc_to_bns)
    return {
        "pdf_key": key,
        "title": clean(row.get("title"))[:500],
        "petitioner": clean(row.get("petitioner"))[:500],
        "respondent": clean(row.get("respondent"))[:500],
        "judge": clean(row.get("judge"))[:500],
        "court": (clean(row.get("court")) or "Supreme Court of India")[:150],
        "decision_date": clean(row.get("decision_date"))[:30],
        "citation": clean(row.get("citation"))[:255],
        "year": year,
        "snippet": make_snippet(text),
        "cites": sorted(cites),
    }


def ingest(args):
    import pandas as pd
    from db import SessionLocal, engine, Base
    import models
    from judgment_utils import load_ipc_to_bns
    from models import Judgment, JudgmentSection

    Base.metadata.create_all(bind=engine)          # creates the new judgments tables if missing
    ipc_to_bns = load_ipc_to_bns(args.bns_file)
    if not ipc_to_bns:
        print(f"WARNING: no IPC->BNS mapping found in {args.bns_file}; IPC citations will be ignored.")
    else:
        print(f"IPC->BNS mapping loaded for {len(ipc_to_bns)} IPC sections")

    years = list(range(args.from_year, args.to_year + 1))
    per_year = math.ceil(args.limit / len(years))
    db = SessionLocal()
    existing_keys = {k for (k,) in db.query(Judgment.pdf_key).all()}
    stats = {"added": 0}
    try:
        for year in years:
            have = db.query(Judgment).filter(Judgment.year == year).count()
            need = per_year - have
            if need <= 0:
                print(f"{year}: already has {have}, skipping")
                continue
            df = load_year(year)
            if df is None:
                print(f"{year}: metadata not available, skipping")
                continue
            # NOTE: no language filter on purpose. `available_languages` can list only the
            # extra regional translations; every judgment has an English PDF, and a missing
            # one simply fails the download and is skipped.
            if "title" in df.columns:
                df = df[df["title"].map(lambda t: bool(clean(t)))]
            df = df.sample(frac=1.0, random_state=7)                 # fixed shuffle = repeatable
            listing = list_pdf_keys(year)
            if not listing:
                print(f"{year}: could not list the bucket; will guess file names (slower)")
            rows = [r for _, r in df.iterrows()]
            rows = [r for r in rows if (resolve_pdf_keys(r, year, listing) or [""])[0] not in existing_keys]
            fails = {}
            got = 0
            chunk = max(args.workers * 3, 8)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i in range(0, len(rows), chunk):
                    if got >= need:
                        break
                    futures = [pool.submit(process_one, r, year, ipc_to_bns, listing) for r in rows[i:i + chunk]]
                    for fut in as_completed(futures):
                        res = fut.result()
                        if isinstance(res, str):
                            fails[res] = fails.get(res, 0) + 1
                            continue
                        if got >= need or res["pdf_key"] in existing_keys:
                            continue
                        cites = res.pop("cites")
                        j = Judgment(**res)
                        db.add(j)
                        db.flush()
                        for act_key, sec in cites:
                            db.add(JudgmentSection(judgment_id=j.id, act_key=act_key, section_number=sec))
                        existing_keys.add(res["pdf_key"])
                        got += 1
                    db.commit()
            stats["added"] += got
            print(f"{year}: +{got} judgments (target {need})  skipped: {fails or 'none'}")
        print(f"Added {stats['added']} judgments in this run.")
    finally:
        db.close()


def rebuild_index():
    import numpy as np
    import faiss
    from db import SessionLocal
    from models import Judgment
    import judgments as jd

    db = SessionLocal()
    try:
        rows = db.query(Judgment).order_by(Judgment.id).all()
        if not rows:
            print("No judgments in the database yet.")
            return
        texts = [f"{j.title}. {j.petitioner} vs {j.respondent}. {(j.snippet or '')[:1500]}" for j in rows]
        vecs = []
        for i in range(0, len(texts), 64):
            vecs.append(jd.embed_texts(texts[i:i + 64]))
            print(f"  embedded {min(i + 64, len(texts))}/{len(texts)}")
        mat = np.vstack(vecs).astype("float32")
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        os.makedirs(os.path.dirname(jd.INDEX_PATH) or ".", exist_ok=True)
        faiss.write_index(index, jd.INDEX_PATH)
        with open(jd.IDMAP_PATH, "w", encoding="utf-8") as f:
            json.dump([j.id for j in rows], f)
        print(f"Index saved: {jd.INDEX_PATH} ({len(rows)} judgments). Restart uvicorn.")
    finally:
        db.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="check the dataset layout and PDF links first")
    ap.add_argument("--inspect-year", type=int, default=2019)
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=5000, help="total judgments wanted (spread over the years)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bns-file", default=os.path.join("nyaya_data", "bns_2023.jsonl"))
    ap.add_argument("--rebuild-index-only", action="store_true")
    args = ap.parse_args(argv)

    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        print("Run first:  pip install pandas pyarrow")
        sys.exit(1)

    if args.inspect:
        inspect(args.inspect_year)
        return
    if not args.rebuild_index_only:
        ingest(args)
    rebuild_index()


if __name__ == "__main__":
    main()
