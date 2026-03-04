"""render.py – Convert a Draft + TopicSummary into Naver-style blog markdown.

Follows rules in docs/naver_style.md:
  - Title: 26–34 characters
  - Intro: 3–4 short sentences
  - H2 sections: 핵심 요약 / 왜 중요한가 / 바로 적용하기
  - Ending: checklist (오늘 핵심 정리)
  - Metadata header for traceability
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from summarize import TopicSummary
from write import Draft, LENSES


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


# ---------------------------------------------------------------------------
# Title helper
# ---------------------------------------------------------------------------

def _fit_title(raw_title: str, min_len: int = 26, max_len: int = 34) -> str:
    """Best-effort fit to Naver's 26–34 character guideline."""
    if len(raw_title) > max_len:
        return raw_title[: max_len - 1] + "…"
    return raw_title


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_markdown(draft: Draft, summary: TopicSummary, timestamp: str = "") -> str:
    """Produce the final blog markdown string."""
    ts = timestamp or make_timestamp()
    lens_label = LENSES.get(draft.lens, {}).get("label", draft.lens)
    title = _fit_title(draft.title)

    lines = [
        f"# {title}",
        "",
        f"> **arm**: {draft.arm} | **lens**: {draft.lens} ({lens_label}) | **date**: {ts}",
        "",
        draft.body,
        "",
        "---",
        "",
        "## Sources",
    ]
    lines.extend([f"- {url}" for url in summary.source_urls])
    lines.extend([
        "",
        "## 오늘 핵심 정리",
        "",
        "- AI 자동화는 반복 작업에 가장 강하다",
        "- 작은 실험부터 시작하는 것이 중요하다",
        "- 데이터 기반으로 개선해야 한다",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_markdown(markdown: str, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
