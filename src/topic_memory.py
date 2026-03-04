"""topic_memory.py - Persistent topic dedupe memory across pipeline runs."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


_MEMORY_VERSION = 1
_TOPICS_KEY = "topics"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_\+]*|[가-힣]{2,}")
_QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalise_quotes(text: str) -> str:
    return (text or "").translate(_QUOTE_TRANSLATION)


def _topic_tokens(text: str) -> set[str]:
    cleaned = _normalise_quotes((text or "").lower())
    return {m.group(0) for m in _TOKEN_RE.finditer(cleaned)}


def jaccard_similarity(a: str, b: str) -> float:
    ta = _topic_tokens(a)
    tb = _topic_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def _prune_recent(entries: list[dict], rolling_days: int = 14) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=rolling_days)
    kept: list[dict] = []
    for rec in entries:
        ts = _parse_iso_ts(str(rec.get("ts", "")))
        if ts is None or ts.tzinfo is None:
            continue
        if ts >= cutoff:
            kept.append({
                "ts": ts.isoformat(),
                "topic": str(rec.get("topic", "")).strip(),
                "topic_kr": str(rec.get("topic_kr", "")).strip(),
            })
    return kept


def load_topic_memory(path: str | Path, rolling_days: int = 14) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if isinstance(raw, dict):
        entries = raw.get(_TOPICS_KEY, [])
    elif isinstance(raw, list):
        entries = raw
    else:
        entries = []
    if not isinstance(entries, list):
        entries = []
    return _prune_recent(entries, rolling_days=rolling_days)


def save_topic_memory(path: str | Path, entries: list[dict], rolling_days: int = 14) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _MEMORY_VERSION,
        _TOPICS_KEY: _prune_recent(entries, rolling_days=rolling_days),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_topic_in_memory(
    topic: str,
    topic_kr: str,
    entries: list[dict],
    threshold: float = 0.7,
) -> bool:
    if not entries:
        return False

    candidates = [topic, topic_kr]
    for rec in entries:
        prev_topic = str(rec.get("topic", ""))
        prev_topic_kr = str(rec.get("topic_kr", ""))
        prev_candidates = [prev_topic, prev_topic_kr]

        for cur in candidates:
            if not cur:
                continue
            for prev in prev_candidates:
                if not prev:
                    continue
                if jaccard_similarity(cur, prev) >= threshold:
                    return True
    return False


def append_topic_memory(entries: list[dict], topic: str, topic_kr: str) -> list[dict]:
    updated = list(entries)
    updated.append({
        "ts": _utc_now_iso(),
        "topic": (topic or "").strip(),
        "topic_kr": (topic_kr or "").strip(),
    })
    return updated
