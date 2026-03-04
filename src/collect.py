"""collect.py – Fetch articles from Google News RSS, Reddit JSON, and external RSS feeds.

Uses only urllib (standard library). Each source type has its own fetcher with
a 10-second timeout. Deduplication is done by canonical URL first, then by
normalised title.

Robustness:
  - Single retry on 429/5xx with 2-second sleep
  - Single retry on 301/302/307/308 redirects (follow Location header)
  - Retry on 406 with relaxed Accept header
  - Rich default headers for compatibility
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen


@dataclass
class SourceItem:
    title: str
    url: str
    source: str  # e.g. "google_news", "reddit", "rss"
    published_at: str = ""
    language: str = "en"
    score: float = 0.0


_TIMEOUT = 10  # seconds
_RETRY_SLEEP = 2  # seconds before retry on 429/5xx

_DEFAULT_HEADERS = {
    "User-Agent": "BlogEngine/1.0",
    "Accept": "application/rss+xml,application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}

_RELAXED_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

_REDIRECT_CODES = {301, 302, 307, 308}


# ---------------------------------------------------------------------------
# URL normalisation / canonical key
# ---------------------------------------------------------------------------

def _canonical_url(raw: str) -> str:
    """Strip query params, fragments, trailing slashes; lowercase scheme+host."""
    p = urlparse(raw)
    return urlunparse((
        p.scheme.lower(),
        p.netloc.lower(),
        p.path.rstrip("/"),
        "",  # params
        "",  # query
        "",  # fragment
    ))


def _normalise_title(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# HTTP fetch helper with redirect following, 406 retry, and 429/5xx retry
# ---------------------------------------------------------------------------

def _read_url(url: str) -> bytes:
    """Fetch URL content with robust retry logic.

    - 3xx: follows Location header up to 3 hops
    - 406: retry once with relaxed Accept header
    - 429/5xx: retry once after 2-second sleep
    """
    headers = dict(_DEFAULT_HEADERS)

    def _fetch(u: str, hdrs: dict) -> bytes:
        with urlopen(Request(u, headers=hdrs), timeout=_TIMEOUT) as resp:
            return resp.read()

    try:
        return _fetch(url, headers)
    except HTTPError as exc:
        # Redirect: follow Location up to 3 hops
        if exc.code in _REDIRECT_CODES:
            current = url
            hop_exc = exc
            for _hop in range(3):
                loc = hop_exc.headers.get("Location", "")
                if not loc:
                    break
                target = urljoin(current, loc)
                print("[collect] WARN  %d redirect %s → %s" % (hop_exc.code, current, target),
                      file=sys.stderr)
                current = target
                try:
                    return _fetch(current, headers)
                except HTTPError as next_exc:
                    if next_exc.code in _REDIRECT_CODES:
                        hop_exc = next_exc
                        continue
                    raise next_exc
                except (URLError, OSError):
                    raise
            raise hop_exc

        # 406 Not Acceptable: retry with relaxed Accept
        if exc.code == 406:
            relaxed = dict(headers)
            relaxed["Accept"] = _RELAXED_ACCEPT
            try:
                return _fetch(url, relaxed)
            except (HTTPError, URLError, OSError):
                raise exc

        # 429 / 5xx: retry once after sleep
        if exc.code == 429 or exc.code >= 500:
            print("[collect] WARN  %d from %s, retrying in %ds …" % (exc.code, url, _RETRY_SLEEP),
                  file=sys.stderr)
            time.sleep(_RETRY_SLEEP)
            try:
                return _fetch(url, headers)
            except (HTTPError, URLError, OSError):
                raise exc

        raise


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

def _fetch_google_news_rss(url: str) -> List[SourceItem]:
    items: List[SourceItem] = []
    try:
        data = _read_url(url)
        root = ET.fromstring(data)
        for item_el in root.iter("item"):
            title = (item_el.findtext("title") or "").strip()
            link = (item_el.findtext("link") or "").strip()
            pub = (item_el.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            items.append(SourceItem(
                title=title,
                url=link,
                source="google_news",
                published_at=pub,
                language="en",
                score=80.0,
            ))
    except (URLError, ET.ParseError, OSError) as exc:
        print(f"[collect] WARN  google_news fetch failed: {url} – {exc}", file=sys.stderr)
    return items


# ---------------------------------------------------------------------------
# Reddit JSON (legacy – kept for any remaining .json URLs)
# ---------------------------------------------------------------------------

def _fetch_reddit_json(url: str) -> List[SourceItem]:
    items: List[SourceItem] = []
    try:
        raw = _read_url(url)
        payload = json.loads(raw)
        children = payload.get("data", {}).get("children", [])
        for child in children:
            d = child.get("data", {})
            title = d.get("title", "").strip()
            permalink = d.get("permalink", "")
            link = f"https://www.reddit.com{permalink}" if permalink else d.get("url", "")
            created = d.get("created_utc", "")
            ups = d.get("ups", 0)
            if not title:
                continue
            items.append(SourceItem(
                title=title,
                url=link,
                source="reddit",
                published_at=str(created),
                language="en",
                score=min(100.0, float(ups) * 0.1),
            ))
    except (URLError, json.JSONDecodeError, OSError) as exc:
        print(f"[collect] WARN  reddit fetch failed: {url} – {exc}", file=sys.stderr)
    return items


# ---------------------------------------------------------------------------
# Reddit RSS/Atom (avoids 403 that the JSON API returns to bots)
# ---------------------------------------------------------------------------

def _fetch_reddit_rss(url: str) -> Tuple[List[SourceItem], bool]:
    """Fetch a Reddit subreddit feed via the .rss (Atom) endpoint."""
    items: List[SourceItem] = []
    try:
        data = _read_url(url)
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            pub = (entry.findtext("atom:published", namespaces=ns)
                   or entry.findtext("atom:updated", namespaces=ns) or "")
            if not title or not link:
                continue
            items.append(SourceItem(
                title=title,
                url=link,
                source="reddit",
                published_at=pub,
                language="en",
                score=60.0,
            ))
    except HTTPError as exc:
        if exc.code == 403:
            # Reddit often blocks bot-like RSS requests. Skip silently.
            return [], True
        print(f"[collect] WARN  reddit rss fetch failed: {url} – {exc}", file=sys.stderr)
    except (URLError, ET.ParseError, OSError) as exc:
        print(f"[collect] WARN  reddit rss fetch failed: {url} – {exc}", file=sys.stderr)
    return items, False


# ---------------------------------------------------------------------------
# Generic external RSS / Atom
# ---------------------------------------------------------------------------

def _fetch_external_rss(url: str) -> List[SourceItem]:
    items: List[SourceItem] = []
    try:
        data = _read_url(url)
        root = ET.fromstring(data)
        # RSS 2.0
        for item_el in root.iter("item"):
            title = (item_el.findtext("title") or "").strip()
            link = (item_el.findtext("link") or "").strip()
            pub = (item_el.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            items.append(SourceItem(
                title=title,
                url=link,
                source="rss",
                published_at=pub,
                language="en",
                score=60.0,
            ))
        # Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            pub = (entry.findtext("atom:published", namespaces=ns)
                   or entry.findtext("atom:updated", namespaces=ns) or "")
            if not title or not link:
                continue
            items.append(SourceItem(
                title=title,
                url=link,
                source="rss",
                published_at=pub,
                language="en",
                score=60.0,
            ))
    except (URLError, ET.ParseError, OSError) as exc:
        print(f"[collect] WARN  rss fetch failed: {url} – {exc}", file=sys.stderr)
    return items


# ---------------------------------------------------------------------------
# Source list loader
# ---------------------------------------------------------------------------

def load_source_targets(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_RE_REDDIT = re.compile(r"reddit\.com", re.IGNORECASE)
_RE_GOOGLE_NEWS = re.compile(r"news\.google\.com", re.IGNORECASE)


def _dispatch_url(url: str) -> List[SourceItem]:
    if _RE_GOOGLE_NEWS.search(url):
        return _fetch_google_news_rss(url)
    if _RE_REDDIT.search(url):
        # Use RSS endpoint to avoid 403 that the JSON API returns to bots
        if ".rss" in url:
            items, _blocked = _fetch_reddit_rss(url)
            return items
        return _fetch_reddit_json(url)
    return _fetch_external_rss(url)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FALLBACK_URLS = [
    "https://news.google.com/rss/search?q=AI+agents",
    "https://www.reddit.com/r/MachineLearning/top/.rss?t=day",
]


def collect_sources(rss_list_path: str | Path) -> list[SourceItem]:
    """Collect, deduplicate, and return source items from all configured feeds."""
    targets = load_source_targets(rss_list_path)
    if not targets:
        print("[collect] WARN  No sources in file, using fallback URLs", file=sys.stderr)
        targets = list(_FALLBACK_URLS)

    raw_items: list[SourceItem] = []
    reddit_rss_total = 0
    reddit_rss_blocked = 0
    for url in targets:
        if _RE_REDDIT.search(url) and ".rss" in url:
            reddit_rss_total += 1
            items, blocked = _fetch_reddit_rss(url)
            raw_items.extend(items)
            if blocked:
                reddit_rss_blocked += 1
            continue
        raw_items.extend(_dispatch_url(url))

    # Deduplicate: canonical URL first, then normalised title
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[SourceItem] = []
    for item in raw_items:
        canon = _canonical_url(item.url)
        if canon in seen_urls:
            continue
        seen_urls.add(canon)

        norm = _normalise_title(item.title)
        if norm in seen_titles:
            continue
        seen_titles.add(norm)

        deduped.append(item)

    print(f"[collect] Collected {len(raw_items)} raw -> {len(deduped)} unique items", file=sys.stderr)
    print(
        f"[collect] INFO reddit rss blocked (403): {reddit_rss_blocked}/{reddit_rss_total} sources",
        file=sys.stderr,
    )
    return deduped
