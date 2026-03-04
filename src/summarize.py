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

import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse, urlunparse

from collect import SourceItem


@dataclass
class TopicSummary:
    topic: str
    key_points: list[str]
    source_urls: list[str]
    score: float


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
        # Clean the raw title into a usable topic
        topic = clean_topic(item.title)

        # Use the placeholder summariser on a synthetic paragraph.
        raw_text = (
            f"{topic}. "
            "Recent discussions show strong momentum in AI automation. "
            "Practical use-cases are shifting from demos to workflow tools. "
            "Model quality and orchestration are both critical for adoption. "
            "Teams get better outcomes with narrow, repeatable pilot scopes. "
            "Cost efficiency remains a major driver for enterprise interest."
        )
        summary_str = summarize_text(raw_text)
        key_points = [s.strip() for s in summary_str.split(". ") if s.strip()]

        summaries.append(TopicSummary(
            topic=topic,
            key_points=key_points,
            source_urls=[item.url],
            score=item.score,
        ))

    return summaries
