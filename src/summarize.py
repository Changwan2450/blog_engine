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

from collect import SourceItem
from openai_client import call_openai


@dataclass
class TopicSummary:
    topic: str
    key_points: list[str]
    source_urls: list[str]
    score: float
    topic_en: str = ""
    topic_kr: str = ""
    topic_angle: str = ""
    key_claims: list[dict] = field(default_factory=list)
    source_domains: list[str] = field(default_factory=list)
    topic_primary: str = "DevTools"
    topic_secondary: str = ""

    def __post_init__(self) -> None:
        # Backward compatibility: topic aliases topic_en.
        if not self.topic_en:
            self.topic_en = self.topic
        if not self.topic:
            self.topic = self.topic_en


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

_KR_TEMPLATES_ASCII = {
    "Failure": "{short} 실패의 이유",
    "Cost": "{short} 비용의 함정",
    "Operations": "{short} 운영 이슈",
    "SysDesign": "{short} 시스템 설계 포인트",
    "Measurement": "{short} 측정 방법",
    "Alternative": "{short} 대안 전략",
}
_KR_DEFAULT_ASCII = "{short} 핵심 포인트"


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

    is_ascii_heavy = bool(re.fullmatch(r"[A-Za-z0-9\s\-+_:'&,./()]+", en or ""))
    if is_ascii_heavy:
        tmpl = _KR_TEMPLATES_ASCII.get(lens_label, _KR_DEFAULT_ASCII)
    else:
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
        raw = call_openai(
            stage="summarize",
            system="You are a concise editor. Return strict JSON only.",
            user=user_msg,
            max_tokens=200,
            temperature=0.3,
        )
        parsed = json.loads(raw)
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

_QUOTE_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{8,120})[\"'“”‘’]")
_BANNED_KEYPOINT_FRAGMENTS = (
    "관련 기술/기업:",
    "출처:",
    "URL 컨텍스트:",
)
_ACTION_VERBS = {
    "launch", "launched", "release", "released", "ship", "shipped",
    "open-source", "opensourced", "announce", "announced", "raise",
    "raised", "acquire", "acquired", "deploy", "deployed", "build", "built",
}

_PRIMARY_BUCKETS = [
    ("AI Infra", {"inference", "serving", "gpu", "latency", "pipeline", "infrastructure"}),
    ("Models", {"model", "llm", "gpt", "transformer", "reasoning"}),
    ("DevTools", {"sdk", "ide", "cli", "tool", "agent", "coding"}),
    ("Chips", {"chip", "npu", "gpu", "silicon", "semiconductor", "cuda"}),
    ("Security", {"security", "vulnerability", "cve", "auth", "zero trust"}),
    ("Web", {"browser", "web", "frontend", "javascript", "react"}),
    ("Product", {"feature", "rollout", "ux", "subscription", "pricing"}),
    ("OpenSource", {"github", "open source", "oss", "repository"}),
    ("Data", {"dataset", "etl", "warehouse", "analytics", "db"}),
    ("Business", {"revenue", "market", "funding", "enterprise", "acquisition"}),
]

_SECONDARY_ENTITIES = [
    "OpenAI", "NVIDIA", "Apple", "AWS", "Amazon", "Google", "Microsoft",
    "Kubernetes", "Rust", "Meta", "Anthropic", "GitHub", "Cloudflare", "Netflix",
]

_EVIDENCE_STOP_TOKENS = {"there", "new", "in", "town", "the", "a", "an", "of", "to", "for"}


def _sanitize_fragments(values: list[str]) -> list[str]:
    clean: list[str] = []
    for v in values:
        if not v:
            continue
        s = v.strip()
        if len(s) < 12:
            continue
        if re.fullmatch(r"[A-Za-z]{1,12}", s):
            continue
        token = re.sub(r"[^A-Za-z]", "", s).lower()
        if token in _EVIDENCE_STOP_TOKENS:
            continue
        if any(bad in v for bad in _BANNED_KEYPOINT_FRAGMENTS):
            continue
        clean.append(s)
    return clean


def _extract_source_domains(urls: list[str]) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()
    for u in urls:
        try:
            d = urlparse(u).netloc.lower().replace("www.", "")
        except Exception:
            d = ""
        if not d or d in seen:
            continue
        seen.add(d)
        domains.append(d)
    return domains


def _classify_topic(topic_en: str, item: SourceItem) -> tuple[str, str]:
    text = f"{topic_en} {item.title} {item.url}".lower()
    primary = "DevTools"
    best = 0
    for label, kws in _PRIMARY_BUCKETS:
        score = sum(1 for kw in kws if kw in text)
        if score > best:
            best = score
            primary = label

    secondary = ""
    for ent in _SECONDARY_ENTITIES:
        if ent.lower() in text:
            secondary = ent
            break
    return primary, secondary


def _confidence_for_claim(claim: str, evidence: list[str]) -> str:
    joined = " ".join([claim] + evidence)
    has_quote_or_num = bool(_QUOTE_RE.search(joined) or re.search(r"\b\d[\d,]*(?:\.\d+)?\b", joined))
    has_entity = bool(re.search(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2}\b", joined))
    has_action = any(v in joined.lower() for v in _ACTION_VERBS)

    if has_quote_or_num and has_entity:
        return "high"
    if has_entity or has_action:
        return "med"
    return "low"


def _build_key_claims(item: SourceItem) -> list[dict]:
    title = item.title.strip()
    reader_text = str(getattr(item, "reader_text", "") or "")
    if len(reader_text.strip()) >= 500:
        source_text = (title + " " + reader_text[:1600]).strip()
    else:
        source_text = title
    claims: list[dict] = []

    quoted: list[str] = []
    for m in _QUOTE_RE.finditer(source_text):
        q = m.group(1).strip()
        if len(q) >= 10:
            quoted.append('"' + q + '"')
        if len(quoted) >= 2:
            break

    numeric_windows: list[str] = []
    for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|x|X|\+|billion|million|thousand|B\b|M\b|K\b)?", source_text):
        st = max(0, m.start() - 28)
        ed = min(len(source_text), m.end() + 36)
        win = re.sub(r"\s+", " ", source_text[st:ed]).strip(" ,;:-")
        if len(win) >= 14:
            numeric_windows.append(win)
        if len(numeric_windows) >= 2:
            break

    entities = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2}\b", source_text)
    entities = [e for e in entities if e not in _CAP_SKIP]

    if quoted or numeric_windows or entities:
        ev = _sanitize_fragments((quoted + numeric_windows + entities)[:4])
        claim = clean_topic(title)
        claims.append({
            "claim": claim,
            "evidence": ev[:3] or [claim[:90]],
            "confidence": _confidence_for_claim(claim, ev[:3]),
        })

    if not claims:
        fallback = clean_topic(title) or title[:90]
        claims.append({
            "claim": fallback,
            "evidence": [fallback[:90]],
            "confidence": "low",
        })

    return claims[:3]


def _extract_key_points_from_item(item: SourceItem) -> list[str]:
    """Build source-grounded key_points from item.title and item.url."""
    points: list[str] = []
    title = item.title.strip()

    # Direct quoted claims from title
    for m in _QUOTE_RE.finditer(title):
        quote = m.group(1).strip()
        if len(quote) >= 10:
            points.append('"' + quote + '"')
        if len(points) >= 2:
            break

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
        points.append(", ".join(caps[:4]) + " mentioned in the report")

    # Fallback: title itself
    if not points:
        points.append(title[:100])

    # Final guard: remove metadata-labeled fragments if present.
    filtered_points = [
        p for p in points
        if p and not any(bad in p for bad in _BANNED_KEYPOINT_FRAGMENTS)
    ]
    if not filtered_points:
        filtered_points = [title[:100]] if title else []

    # Deduplicate while preserving order.
    unique_points: list[str] = []
    seen: set[str] = set()
    for p in filtered_points:
        norm = " ".join(p.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        unique_points.append(p)
    return unique_points[:5]


def _flatten_points_from_claims(key_claims: list[dict]) -> list[str]:
    points: list[str] = []
    for kc in key_claims:
        claim = str(kc.get("claim", "")).strip()
        ev = [str(x).strip() for x in kc.get("evidence", []) if str(x).strip()]
        if claim:
            points.append(claim)
        points.extend(ev[:2])
    return _sanitize_fragments(points)[:5]


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
        topic_en = rewritten["topic_en"]
        key_claims = _build_key_claims(item)
        key_points = _flatten_points_from_claims(key_claims)
        if not key_points:
            key_points = _extract_key_points_from_item(item)
            key_points = _sanitize_fragments(key_points)

        source_urls = [item.url]
        source_domains = _extract_source_domains(source_urls)
        topic_primary, topic_secondary = _classify_topic(topic_en, item)

        summaries.append(TopicSummary(
            topic=topic_en,
            topic_en=topic_en,
            key_points=key_points,
            key_claims=key_claims,
            source_urls=source_urls,
            source_domains=source_domains,
            score=item.score,
            topic_kr=rewritten["topic_kr"],
            topic_angle=rewritten["topic_angle"],
            topic_primary=topic_primary,
            topic_secondary=topic_secondary,
        ))

    return summaries
