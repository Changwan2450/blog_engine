"""render.py – Convert a Draft + TopicSummary into final markdown."""
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

def _fit_title(raw_title: str, min_len: int = 26, max_len: int = 45) -> str:
    """Best-effort fit to title length guideline (raised to 45 for Korean headlines)."""
    if len(raw_title) > max_len:
        # Break at last space before limit to avoid mid-word cuts
        cut = raw_title[:max_len - 1]
        sp = cut.rfind(" ")
        return (cut[:sp] if sp > max_len // 3 else cut) + "…"
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
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_markdown(markdown: str, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
