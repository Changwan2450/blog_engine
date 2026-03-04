# src/trends.py
"""
trends.py – Detect bursty terms (new/rapidly increasing words) from collected items,
store trend report, and update a topic pool for future runs.

Designed to be cheap + stdlib-only:
- Uses titles only (no page scraping) so it stays fast.
- Maintains a rolling history in data/term_history.jsonl (1 JSON per run).
- Produces data/trend_terms.json (written by run_once.py) via detect_trends().
- Updates data/topic_pool.txt via update_topic_pool() with debate-ready templates.

Works with collect.SourceItem.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from collect import SourceItem

# -----------------------------------------------------------------------------
# Tokenization / filters
# -----------------------------------------------------------------------------

# Keep tokens like: MCP, RAG, vLLM, LangGraph, tool-calling, GPT-4o, etc.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9][A-Za-z0-9\-\+_]{1,}")

# Basic stopwords + overly generic tokens.
# Note: we intentionally keep some AI terms out of STOP so trends can surface.
STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from",
    "is", "are", "be", "as", "at", "by", "it", "this", "that", "these", "those",
    "we", "you", "they", "i", "our", "your", "their", "vs",
    "new", "best", "how", "why", "what", "when", "where",
    "today", "yesterday", "tomorrow", "week", "month", "year",
}

# Source weighting: reddit tends to surface new jargon earlier.
SOURCE_WEIGHT = {
    "reddit": 1.50,
    "google_news": 1.05,
    "rss": 1.15,
}

# Terms that are too frequent & not useful can be demoted here later.
SOFT_DOWNWEIGHT = {
    "ai": 0.35,
    "llm": 0.55,
    "llms": 0.55,
    "agent": 0.70,
    "agents": 0.70,
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    toks = [m.group(0) for m in TOKEN_RE.finditer(text)]
    out: list[str] = []
    for t in toks:
        low = t.lower()
        if low in STOP:
            continue
        # drop pure numeric-ish tokens
        if low.isdigit():
            continue
        # drop very short
        if len(t) < 3:
            continue
        out.append(t)
    return out


def _term_weight(term: str) -> float:
    """Downweight overly generic tokens without fully removing them."""
    return float(SOFT_DOWNWEIGHT.get(term.lower(), 1.0))


# -----------------------------------------------------------------------------
# Snapshot + history
# -----------------------------------------------------------------------------

def snapshot_term_counts(items: Iterable[SourceItem]) -> dict:
    """
    Build a snapshot from items:
    {
      "counts": {term: count, ...},
      "by_source": {source: {term: count, ...}, ...}
    }
    """
    counts = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)

    for it in items:
        txt = it.title or ""
        for tok in _tokenize(txt):
            counts[tok] += 1
            by_source[it.source][tok] += 1

    return {
        "counts": dict(counts),
        "by_source": {k: dict(v) for k, v in by_source.items()},
    }


def append_history(history_path: str | Path, snapshot: dict) -> None:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def load_recent_history(history_path: str | Path, n: int = 14) -> list[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-n:] if len(lines) > n else lines
    out: list[dict] = []
    for ln in tail:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _baseline_average(history: list[dict]) -> dict[str, float]:
    """
    Compute baseline term avg count per run from history snapshots.
    Returns {term: avg_count}.
    """
    if not history:
        return {}
    acc = Counter()
    for h in history:
        acc.update(h.get("counts", {}))
    denom = max(1, len(history))
    return {k: (v / denom) for k, v in acc.items()}


# -----------------------------------------------------------------------------
# Trend scoring
# -----------------------------------------------------------------------------

def detect_trends(
    items: list[SourceItem],
    history_path: str | Path,
    top_k: int = 25,
    history_n: int = 14,
) -> dict:
    """
    Detect bursty terms:
      score = burst * source_weight * log(1+count) * term_weight
      burst = (count + 1) / (baseline_avg + 1)

    Also appends this run's snapshot to history_path.

    Returns dict suitable for JSON:
    {
      "n_items": int,
      "top_terms": [
        {"term": str, "count": int, "base_avg": float, "burst": float,
         "source_hint": str, "score": float},
        ...
      ]
    }
    """
    now = snapshot_term_counts(items)
    hist = load_recent_history(history_path, n=history_n)
    base_avg = _baseline_average(hist)

    now_counts = Counter(now.get("counts", {}))
    by_source = {s: Counter(d) for s, d in now.get("by_source", {}).items()}

    scored: list[dict] = []
    for term, c in now_counts.items():
        c = float(c)
        b = float(base_avg.get(term, 0.0))
        burst = (c + 1.0) / (b + 1.0)

        # choose source where the term appears most
        best_s, best_c = "", 0
        for s, cnts in by_source.items():
            sc = int(cnts.get(term, 0))
            if sc > best_c:
                best_c = sc
                best_s = s

        sw = SOURCE_WEIGHT.get(best_s, 1.0) if best_s else 1.0
        tw = _term_weight(term)

        # filter weak noise
        if c < 2 and burst < 2.2:
            continue

        score = burst * sw * math.log(1.0 + c) * tw

        scored.append({
            "term": term,
            "count": int(c),
            "base_avg": round(b, 3),
            "burst": round(burst, 3),
            "source_hint": best_s,
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    result = {
        "n_items": len(items),
        "top_terms": top,
    }

    # persist this run snapshot for next baseline
    append_history(history_path, now)
    return result


# -----------------------------------------------------------------------------
# Topic pool update
# -----------------------------------------------------------------------------

_DEFAULT_TEMPLATES = [
    # Debate / cost / ops style (fits your Lens system)
    "{TERM}가 뜨는 이유: 컨텍스트 비용/운영 관점에서 정리",
    "{TERM} 도입이 실패하는 3가지 이유 (실전 운영 기준)",
    "{TERM} vs 기존 방식: 뭐가 더 싸고 안정적인가",
    "{TERM}를 오케스트레이션에 붙이는 최소 설계 (MVP)",
    "스킬 기반 아키텍처에서 {TERM}는 어디에 들어가나",
    # Vibe/MD/MCP/skills angles
    "{TERM}가 바이브코딩 워크플로우를 바꾸는 지점",
    "{TERM}를 MD 기반 작업흐름에 녹이는 방법",
    "MCP 없이도 {TERM}로 충분한가? (툴/스킬 호출 관점)",
]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def update_topic_pool(
    topic_pool_path: str | Path,
    trend: dict,
    max_add: int = 12,
    templates: list[str] | None = None,
) -> list[str]:
    """
    Append new topic candidates derived from top trend terms.
    - Avoid duplicates.
    - Add up to max_add new lines.

    Returns list of newly added topics.
    """
    path = Path(topic_pool_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    for ln in _read_lines(path):
        s = ln.strip()
        if s and not s.startswith("#"):
            existing.add(s)

    tpls = templates if templates is not None else list(_DEFAULT_TEMPLATES)

    added: list[str] = []
    top_terms = trend.get("top_terms", [])

    # Go deeper than max_add so templates can still fill even if some are duplicates.
    for rec in top_terms[: max_add * 3]:
        term = (rec.get("term") or "").strip()
        if not term:
            continue
        # Skip ultra generic terms even if they appear
        if term.lower() in {"ai", "llm", "agent", "agents"}:
            continue

        for tpl in tpls:
            topic = tpl.replace("{TERM}", term)
            if topic in existing:
                continue
            existing.add(topic)
            added.append(topic)
            if len(added) >= max_add:
                break
        if len(added) >= max_add:
            break

    if added:
        with path.open("a", encoding="utf-8") as f:
            for t in added:
                f.write(t + "\n")

    return added