"""summarize.py – Filter, deduplicate, and extract summaries from collected items.

The summarize_text() function is a placeholder that returns the first few
sentences of the input. Replace it with an LLM API call later.

clean_topic() deterministically cleans raw feed titles:
  - removes bracket tags: [Discussion], (2026), {..}
  - strips emojis and excessive punctuation
  - collapses whitespace
  - trims to 90 chars max
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from collect import SourceItem


@dataclass
class TopicSummary:
    topic: str
    key_points: list[str]
    source_urls: list[str]
    score: float
    topic_kr: str = ""
    topic_angle: str = ""


# ---------------------------------------------------------------------------
# Topic cleaning
# ---------------------------------------------------------------------------

# Bracket tags: [Discussion], [Show HN], (2026), {meta}, etc.
_BRACKET_RE = re.compile(r"\[[^\]]{0,40}\]|\([^\)]{0,20}\)|\{[^\}]{0,20}\}")

# Emojis and misc unicode symbols (broad ranges covering most emoji blocks)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000025A0-\U000025FF"  # geometric shapes
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

# Excessive punctuation: 2+ consecutive punctuation chars (excluding hyphens)
_EXCESS_PUNCT_RE = re.compile(r"[!?.,:;…·•|/\\~`@#$%^&*+=<>]{2,}")

# Leading/trailing punctuation junk after cleaning
_EDGE_PUNCT_RE = re.compile(r"^[\s\-—:,;.!?·•|/]+|[\s\-—:,;.!?·•|/]+$")

# Whitespace collapse
_MULTI_SPACE_RE = re.compile(r"\s{2,}")

_MAX_TOPIC_LEN = 90
_MIN_TOPIC_LEN = 12


def clean_topic(raw: str) -> str:
    """Deterministically clean a raw feed title into a usable topic string.

    Steps:
      1. Remove bracket tags: [Discussion], (2026), {meta}
      2. Strip emojis and unicode symbols
      3. Collapse excessive punctuation
      4. Collapse whitespace, strip edge junk
      5. Trim to 90 chars (break at word boundary)
      6. Fallback to original trimmed if result too short (<12 chars)
    """
    if not raw:
        return raw

    text = raw

    # 1. Bracket tags
    text = _BRACKET_RE.sub(" ", text)

    # 2. Emojis
    text = _EMOJI_RE.sub(" ", text)

    # 3. Excessive punctuation
    text = _EXCESS_PUNCT_RE.sub(" ", text)

    # 4. Whitespace + edge cleanup
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _EDGE_PUNCT_RE.sub("", text)
    text = text.strip()

    # 5. Trim to max length (break at last space before limit)
    if len(text) > _MAX_TOPIC_LEN:
        cut = text[:_MAX_TOPIC_LEN]
        last_space = cut.rfind(" ")
        if last_space > _MAX_TOPIC_LEN // 2:
            text = cut[:last_space].rstrip(" ,;:-")
        else:
            text = cut.rstrip(" ,;:-")

    # 6. Fallback if too short
    if len(text) < _MIN_TOPIC_LEN:
        fallback = raw.strip()
        if len(fallback) > _MAX_TOPIC_LEN:
            fallback = fallback[:_MAX_TOPIC_LEN].rstrip()
        return fallback

    return text


# ---------------------------------------------------------------------------
# Topic rewriting helpers
# ---------------------------------------------------------------------------

_DANGLING_RE = re.compile(
    r"\s*[—\-–]\s*(and|or|but|with|for|to|the|a|an)\s*$",
    re.IGNORECASE,
)
_TRAILING_JUNK_RE = re.compile(r"[\s:,;.!?—\-–]+$")

_KR_TEMPLATES = {
    "Failure":     "{short}가 실패하는 이유",
    "Cost":        "{short}의 숨은 비용",
    "Operations":  "{short}를 운영에 올리면 터지는 것들",
    "SysDesign":   "{short} 시스템 설계 구조",
    "Measurement": "{short}를 측정하는 방법",
    "Alternative": "{short} 없이도 되는 방법",
}
_KR_DEFAULT = "{short}가 중요한 이유"


def _heuristic_topic_en(raw: str) -> str:
    t = _DANGLING_RE.sub("", raw)
    t = re.sub(r"\s+:\s*$", "", t)
    t = re.sub(r",\s*and\s*$", "", t, flags=re.IGNORECASE)
    t = _TRAILING_JUNK_RE.sub("", t).strip()
    if len(t) > 80:
        cut = t[:80]
        sp = cut.rfind(" ")
        t = cut[:sp].rstrip(" ,;:-") if sp > 40 else cut
    return t or raw.strip()[:80]


def _heuristic_topic_kr(en: str, lens_label: str = "") -> str:
    def _word_cut(s: str, max_len: int) -> str:
        """Cut s to max_len chars at a word boundary."""
        if len(s) <= max_len:
            return s
        cut = s[:max_len]
        sp = cut.rfind(" ")
        return cut[:sp].rstrip(" ,;:-") if sp > max_len // 3 else cut.rstrip(" ,;:-")

    tmpl = _KR_TEMPLATES.get(lens_label, _KR_DEFAULT)
    # Try progressively shorter cuts until the result fits 45 chars
    for limit in (28, 22, 16):
        short = _word_cut(en, limit)
        candidate = tmpl.format(short=short)
        if len(candidate) <= 45:
            return candidate
    return _word_cut(en, 42)


def rewrite_topic(raw: str, lens_label: str = "") -> dict:
    """Return {topic_en, topic_kr, topic_angle}. LLM if key present, else heuristic."""
    en = _heuristic_topic_en(raw)
    kr = _heuristic_topic_kr(en, lens_label)
    angle_prefix = (lens_label + " 관점에서 ") if lens_label else ""
    angle = angle_prefix + en + "의 실질적 영향"

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"topic_en": en, "topic_kr": kr, "topic_angle": angle}

    user_msg = (
        'Raw headline: "' + raw + '"\n'
        "Lens: " + (lens_label or "general") + "\n\n"
        "Return ONLY valid JSON with keys:\n"
        "  topic_en  - clean English headline, <=80 chars, no trailing junk\n"
        "  topic_kr  - Korean headline, <=45 chars\n"
        "  topic_angle - 1 sentence angle aligned with the lens\n"
        "No markdown, no extra keys."
    )
    try:
        payload = json.dumps({
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": 0.3,
            "max_tokens": 200,
        }).encode()
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        parsed = json.loads(data["choices"][0]["message"]["content"])
        return {
            "topic_en": str(parsed.get("topic_en", en))[:80],
            "topic_kr": str(parsed.get("topic_kr", kr))[:45],
            "topic_angle": str(parsed.get("topic_angle", angle)),
        }
    except Exception:
        return {"topic_en": en, "topic_kr": kr, "topic_angle": angle}


# ---------------------------------------------------------------------------
# Source-grounded key_points extraction
# ---------------------------------------------------------------------------

_CAP_SKIP = {
    "The", "This", "That", "With", "From", "Into", "Their", "They",
    "What", "When", "Where", "Which", "How", "Why", "Who", "Will",
    "Has", "Have", "Been", "Are", "Was", "Were", "And", "But", "For",
    "Its", "New", "Now", "Just", "Also", "More", "Most", "Some",
}


def _extract_key_points_from_item(item: SourceItem) -> list[str]:
    """Build source-grounded key_points from item.title and item.url."""
    points: list[str] = []
    title = item.title.strip()

    # Numbers + surrounding context from title
    for m in re.finditer(
        r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|x|X|\+|billion|million|thousand|B\b|M\b|K\b)?",
        title,
    ):
        start = max(0, m.start() - 30)
        end = min(len(title), m.end() + 40)
        snippet = re.sub(r"\s+", " ", title[start:end]).strip()
        if len(snippet) > 12:
            points.append(snippet)
        if len(points) >= 2:
            break

    # Capitalized product / company names from title
    cap_phrases = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", title)
    caps = [p for p in cap_phrases if p not in _CAP_SKIP and len(p) > 3]
    if caps:
        points.append("관련 기술/기업: " + ", ".join(caps[:4]))

    # Domain as publication signal
    try:
        domain = urlparse(item.url).netloc.replace("www.", "")
        if domain:
            points.append("출처: " + domain)
    except Exception:
        pass

    # URL path keywords
    try:
        path = urlparse(item.url).path
        parts = [
            p.replace("-", " ").replace("_", " ")
            for p in path.split("/")
            if len(p) > 5 and not p.isdigit() and p.replace("-", "").replace("_", "").isalpha()
        ]
        if parts:
            kw = " / ".join(parts[:3])
            points.append("URL 컨텍스트: " + kw[:60])
    except Exception:
        pass

    # Fallback: title itself
    if not points:
        points.append(title[:100])

    return points[:5]


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _canonical_url(raw: str) -> str:
    p = urlparse(raw)
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), "", "", ""))


def _normalise_title(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Placeholder summariser (swap with LLM later)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def summarize_text(text: str, max_sentences: int = 5) -> str:
    """Placeholder: return the first *max_sentences* sentences of *text*.

    Designed to be replaced by an LLM API call that accepts a text string
    and returns a concise summary string.
    """
    sentences = _SENTENCE_RE.split(text.strip())
    return " ".join(sentences[:max_sentences])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MAX_ITEMS_FOR_SUMMARIZE = 200  # soft limit to prevent pipeline explosion


def summarize_items(items: list[SourceItem], top_n: int = 10) -> List[TopicSummary]:
    """Filter to en/ko, deduplicate, rank by score, then summarise top N items."""

    # --- Filter language ---
    lang_ok = [it for it in items if it.language in {"en", "ko"}]

    # --- Soft limit: take top-scoring items first ---
    if len(lang_ok) > _MAX_ITEMS_FOR_SUMMARIZE:
        lang_ok = sorted(lang_ok, key=lambda x: x.score, reverse=True)[:_MAX_ITEMS_FOR_SUMMARIZE]

    # --- Deduplicate (canonical URL → normalised title) ---
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[SourceItem] = []

    for item in lang_ok:
        c_url = _canonical_url(item.url)
        if c_url in seen_urls:
            continue
        seen_urls.add(c_url)

        n_title = _normalise_title(item.title)
        if n_title in seen_titles:
            continue
        seen_titles.add(n_title)

        unique.append(item)

    # --- Rank and slice ---
    ranked = sorted(unique, key=lambda x: x.score, reverse=True)[:top_n]

    # --- Build summaries ---
    summaries: list[TopicSummary] = []
    for item in ranked:
        topic_raw = clean_topic(item.title)
        rewritten = rewrite_topic(topic_raw)
        topic = rewritten["topic_en"]
        key_points = _extract_key_points_from_item(item)

        summaries.append(TopicSummary(
            topic=topic,
            key_points=key_points,
            source_urls=[item.url],
            score=item.score,
            topic_kr=rewritten["topic_kr"],
            topic_angle=rewritten["topic_angle"],
        ))

    return summaries
