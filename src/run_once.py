"""run_once.py – Single-run orchestrator for the blog engine pipeline.

Usage:
    python3 src/run_once.py --slot morning
    python3 src/run_once.py --slot evening --seed 42

Pipeline: collect → (optional: trends/topic update) → summarize → write (3 drafts)
          → quality gate (+ optional regen) → select → render → save → log
Logs each run to data/runs.jsonl and updates bandit counts.

IMPORTANT: The very first thing this script does is fix sys.path so that
Python's stdlib ``select`` module is not shadowed by our ``select.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- sys.path fix ----
# Python adds the script's directory (src/) to sys.path[0].
# This shadows the stdlib ``select`` module with our ``select.py``.
# We temporarily remove it, force-import the stdlib modules that need
# the real ``select``, then put it back.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR in sys.path:
    sys.path.remove(_SRC_DIR)
import select as _stdlib_select          # noqa: F401 – cache stdlib select
import selectors as _stdlib_selectors    # noqa: F401 – cache stdlib selectors
import urllib.request as _urllib_req     # noqa: F401 – triggers full stdlib chain
sys.path.insert(0, _SRC_DIR)
# ---- end fix ----

from env_loader import get_model, load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

import argparse
import importlib.util
import json
import random
import re
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from collect import collect_sources, load_source_targets
from summarize import summarize_items
from write import LENSES, draft_passes_quality, generate_drafts, rewrite_with_feedback
from research import extract_patterns, load_patterns
from render import make_timestamp, render_markdown, save_markdown
from learn import log_run
from critic import critique_draft
from hook import generate_hook
from topic_memory import (
    append_topic_memory,
    is_topic_in_memory,
    load_topic_memory,
    save_topic_memory,
)


def _load_select_module():
    """Load our local select.py without conflicting with stdlib select."""
    mod_path = Path(_SRC_DIR) / "select.py"
    spec = importlib.util.spec_from_file_location("blog_select", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sel = _load_select_module()
load_bandit_state = _sel.load_bandit_state
save_bandit_state = _sel.save_bandit_state
choose_best_draft = _sel.choose_best_draft
score_draft = getattr(_sel, "score_draft", None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one blog-engine pipeline: collect → summarize → write → select → render → save"
    )
    parser.add_argument(
        "--slot",
        required=True,
        choices=["morning", "evening"],
        help="Time slot label (morning / evening)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers: sources merge + topic selection
# ---------------------------------------------------------------------------

def _merge_source_files(data_dir: Path) -> Path:
    """
    Merge three files into one temp list for collect_sources():
      - data/google_news_rss.txt
      - data/reddit_sources.txt
      - data/sources_rss.txt

    collect_sources() currently accepts a single path; it dedupes internally.
    """
    paths = [
        data_dir / "google_news_rss.txt",
        data_dir / "reddit_sources.txt",
        data_dir / "sources_rss.txt",
    ]
    merged: list[str] = []
    for p in paths:
        merged.extend(load_source_targets(p))

    # Write merged list to a deterministic temp file under data/
    tmp_path = data_dir / "_sources_all.txt"
    tmp_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    return tmp_path


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+_]{2,}")

def _keywords(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}

def _pick_summary_by_topic(summaries, topic_hint: str):
    """
    Choose the summary whose (topic/title/url) best overlaps with topic_hint.
    Falls back to summaries[0] if nothing matches.
    """
    if not summaries:
        raise RuntimeError("No summaries to pick from.")

    if not topic_hint:
        return summaries[0]

    hint = _keywords(topic_hint)
    if not hint:
        return summaries[0]

    best = summaries[0]
    best_score = -1
    for s in summaries:
        hay = " ".join([getattr(s, "topic", "") or "", getattr(s, "title", "") or "", getattr(s, "url", "") or ""])
        kws = _keywords(hay)
        score = len(hint & kws)
        if score > best_score:
            best_score = score
            best = s

    return best if best_score > 0 else summaries[0]


def _load_topic_pool(topic_pool_path: Path) -> list[str]:
    if not topic_pool_path.exists():
        return []
    lines = topic_pool_path.read_text(encoding="utf-8").splitlines()
    # allow comments, blanks
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# Quality gate with 1 extra regeneration pass
# ---------------------------------------------------------------------------

def _quality_gate_with_regen(
    summary,
    topic_hint: str,
    initial_drafts: list,
) -> list:
    """Apply quality gate. If fewer than 2 pass, generate 2 more (max 5 total).

    Returns all drafts that passed the gate (or all if none pass, as fallback).
    """
    passed = [d for d in initial_drafts if draft_passes_quality(d)]

    if len(passed) >= 2:
        print(f"[quality] {len(passed)}/{len(initial_drafts)} drafts passed quality gate",
              file=sys.stderr)
        return passed

    # Regeneration pass: generate 2 more drafts
    print(f"[quality] Only {len(passed)}/{len(initial_drafts)} passed; "
          f"regenerating 2 more drafts …", file=sys.stderr)
    extra = generate_drafts(summary, count=2, topic_hint=topic_hint)
    extra_passed = [d for d in extra if draft_passes_quality(d)]
    all_passed = passed + extra_passed

    if all_passed:
        print(f"[quality] After regen: {len(all_passed)} drafts passed", file=sys.stderr)
        return all_passed

    # Fallback: return all drafts if nothing passes (avoid empty pipeline)
    print("[quality] WARN  No drafts passed quality gate; using all as fallback",
          file=sys.stderr)
    return initial_drafts + extra


# ---------------------------------------------------------------------------
# Trend term extraction for run logging
# ---------------------------------------------------------------------------

def _load_top_trend_terms(data_dir: Path, max_terms: int = 5) -> list[str]:
    """Load top terms from trend_terms.json for run logging."""
    path = data_dir / "trend_terms.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            rec.get("term", "")
            for rec in data.get("top_terms", [])[:max_terms]
            if rec.get("term")
        ]
    except (json.JSONDecodeError, OSError):
        return []


def _load_recent_runs(runs_path: Path, n: int = 60) -> list[dict]:
    if not runs_path.exists():
        return []
    rows = runs_path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for ln in rows[-n:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Optional: trends integration (guarded import)
# ---------------------------------------------------------------------------

def _maybe_run_trends(project_root: Path, items) -> str:
    """
    If src/trends.py exists and imports cleanly:
      - detect trends from collected items
      - write data/trend_terms.json
      - update data/topic_pool.txt
    Returns: a topic hint string (randomly sampled from topic_pool if added/exists)
    """
    data_dir = project_root / "data"
    trends_mod_path = Path(_SRC_DIR) / "trends.py"
    if not trends_mod_path.exists():
        return ""

    try:
        spec = importlib.util.spec_from_file_location("blog_trends", trends_mod_path)
        if spec is None or spec.loader is None:
            return ""
        trends = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trends)

        history_path = data_dir / "term_history.jsonl"
        trend = trends.detect_trends(items, history_path=history_path, top_k=25, history_n=14)

        (data_dir / "trend_terms.json").write_text(
            json.dumps(trend, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        topic_pool_path = data_dir / "topic_pool.txt"
        added = trends.update_topic_pool(topic_pool_path, trend, max_add=10)

        # pick a topic hint:
        pool = _load_topic_pool(topic_pool_path)
        if pool:
            return random.choice(pool)
        return added[0] if added else ""
    except Exception as exc:
        print(f"[trends] WARN  trends disabled due to error: {exc}", file=sys.stderr)
        return ""


def _fetch_text(url: str, timeout: int = 10) -> str:
    req = Request(url, headers={
        "User-Agent": "BlogEngine/1.0",
        "Accept": "text/markdown,text/plain,text/html;q=0.9,*/*;q=0.8",
        "Connection": "close",
    })
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _fetch_reader_markdown(original_url: str, timeout: int = 10) -> str:
    src = (original_url or "").strip()
    if not src:
        return ""

    defuddle_url = "https://defuddle.md/" + src
    try:
        text = _fetch_text(defuddle_url, timeout=timeout)
        if len(text.strip()) >= 500:
            return text
    except (HTTPError, URLError, OSError, ValueError):
        pass

    parsed = urlparse(src)
    no_scheme = src.split("://", 1)[1] if parsed.scheme in {"http", "https"} and "://" in src else src
    jina_url = "https://r.jina.ai/http://" + no_scheme
    try:
        text = _fetch_text(jina_url, timeout=timeout)
        if len(text.strip()) >= 500:
            return text
    except (HTTPError, URLError, OSError, ValueError):
        return ""
    return ""


def _attach_reader_text(items, max_items: int = 20) -> None:
    if not items:
        return
    ranked = sorted(items, key=lambda x: getattr(x, "score", 0.0), reverse=True)[:max_items]
    for it in ranked:
        text = _fetch_reader_markdown(getattr(it, "url", ""), timeout=10)
        if text:
            setattr(it, "reader_text", text)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_once(project_root: Path, slot: str, seed: int | None = None) -> tuple[Path, list[Path]]:
    """Execute the full pipeline and return (final_path, candidate_paths)."""
    data_dir = project_root / "data"
    out_dir = project_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Seed for determinism
    if seed is not None:
        random.seed(seed)

    # --- 1. Collect ---
    print("[pipeline] 1/9  Collecting sources …", file=sys.stderr)
    merged_sources = _merge_source_files(data_dir)
    items = collect_sources(merged_sources)
    if not items:
        print("[pipeline] WARN  No items collected; using fallback", file=sys.stderr)

    # --- 2. Research (extract writing patterns) ---
    print("[pipeline] 2/9  Extracting research patterns …", file=sys.stderr)
    try:
        research_patterns = extract_patterns(items, data_dir / "research_patterns.json")
    except Exception as exc:
        print(f"[research] WARN  Pattern extraction failed ({exc}); skipping", file=sys.stderr)
        research_patterns = load_patterns(data_dir / "research_patterns.json")

    # --- 3. Trends/Topic update (optional) ---
    print("[pipeline] 3/9  Detecting trends (optional) …", file=sys.stderr)
    topic_hint = _maybe_run_trends(project_root, items)

    # --- 4. Summarize ---
    print("[pipeline] 4/9  Summarizing …", file=sys.stderr)
    _attach_reader_text(items, max_items=20)
    summaries = summarize_items(items, top_n=10)
    if not summaries:
        raise RuntimeError("No summaries generated. Check source inputs.")

    # Remove topics that are too similar to recent runs (rolling 14 days).
    topic_memory_path = data_dir / "topic_memory.json"
    topic_memory = load_topic_memory(topic_memory_path, rolling_days=14)
    fresh_summaries = [
        s for s in summaries
        if not is_topic_in_memory(
            getattr(s, "topic", "") or "",
            getattr(s, "topic_kr", "") or "",
            topic_memory,
            threshold=0.7,
        )
    ]
    if fresh_summaries:
        summaries_for_pick = fresh_summaries
    else:
        print("[topic] WARN  all candidates overlap recent topics; allowing best available",
              file=sys.stderr)
        summaries_for_pick = summaries

    # pick summary using topic hint (if any), otherwise the top summary
    target_summary = _pick_summary_by_topic(summaries_for_pick, topic_hint)
    subject_for_hook = (
        (getattr(target_summary, "topic_kr", "") or "").strip()
        or (getattr(target_summary, "topic_secondary", "") or "").strip()
        or (getattr(target_summary, "topic_en", "") or "").strip()
        or (getattr(target_summary, "topic_primary", "") or "").strip()
    )
    hook_line = generate_hook(subject_for_hook)

    # --- 5. Write (5 drafts) with topic_hint + research ---
    print("[pipeline] 5/9  Generating 5 drafts …", file=sys.stderr)
    drafts = generate_drafts(target_summary, count=5, topic_hint=topic_hint,
                             research_patterns=research_patterns,
                             hook_line=hook_line)

    # --- 5.5 Critic + one rewrite pass ---
    print("[pipeline] 5.5/9  Critic + rewrite …", file=sys.stderr)
    for d in drafts:
        lens_label = LENSES.get(getattr(d, "lens", ""), {}).get("label", getattr(d, "lens", ""))
        critique = critique_draft(
            getattr(d, "body", "") or "",
            getattr(target_summary, "topic_kr", "") or "",
            lens_label,
        )
        setattr(d, "critique", critique)

    rewrite_idx = 0
    if callable(score_draft):
        try:
            tmp_state = load_bandit_state(data_dir / "bandit_state.json")
            tmp_runs = _load_recent_runs(data_dir / "runs.jsonl", n=60)
            scored = []
            for i, d in enumerate(drafts):
                scored_result = score_draft(
                    d,
                    tmp_state,
                    topic_hint=topic_hint,
                    summary=target_summary,
                    topic_memory=topic_memory,
                    recent_runs=tmp_runs,
                )
                if isinstance(scored_result, tuple):
                    total = float(scored_result[0])
                elif isinstance(scored_result, (int, float)):
                    total = float(scored_result)
                else:
                    total = 0.0
                scored.append((total, i))
            rewrite_idx = max(scored, key=lambda x: x[0])[1]
        except Exception:
            rewrite_idx = 0

    if drafts:
        target = drafts[rewrite_idx]
        lens_label = LENSES.get(getattr(target, "lens", ""), {}).get("label", getattr(target, "lens", ""))
        rewritten = rewrite_with_feedback(
            getattr(target, "body", "") or "",
            getattr(target_summary, "topic_kr", "") or "",
            lens_label,
            getattr(target, "critique", {}) or {},
        )
        if rewritten and rewritten.strip():
            target.body = rewritten

    # --- 6. Quality gate (+ optional regen) ---
    print("[pipeline] 6/9  Quality gate …", file=sys.stderr)
    drafts = _quality_gate_with_regen(target_summary, topic_hint, drafts)

    # --- 7. Select (with expanded bandit scoring) ---
    print("[pipeline] 7/9  Selecting best draft …", file=sys.stderr)
    bandit_path = data_dir / "bandit_state.json"
    state = load_bandit_state(bandit_path)
    recent_runs = _load_recent_runs(data_dir / "runs.jsonl", n=60)
    selected = choose_best_draft(
        drafts,
        state,
        topic_hint=topic_hint,
        summary=target_summary,
        topic_memory=topic_memory,
        recent_runs=recent_runs,
    )

    # Update bandit counts for the selected arm
    state.setdefault("counts", {})
    state["counts"][selected.arm] = state["counts"].get(selected.arm, 0) + 1
    save_bandit_state(state, bandit_path)

    # --- 8. Render ---
    print("[pipeline] 8/9  Rendering markdown …", file=sys.stderr)
    stamp = make_timestamp()

    candidate_dir = out_dir / f"{stamp}_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths: list[Path] = []
    for idx, draft in enumerate(drafts, start=1):
        md = render_markdown(draft, target_summary, timestamp=stamp)
        path = save_markdown(md, candidate_dir / f"{idx}_{draft.arm}_{draft.lens}.md")
        candidate_paths.append(path)

    final_md = render_markdown(selected, target_summary, timestamp=stamp)
    final_path = save_markdown(final_md, out_dir / f"{stamp}_final.md")

    # --- 9. Log (include topic_hint and trend terms) ---
    print("[pipeline] 9/9  Logging run …", file=sys.stderr)
    trend_terms = _load_top_trend_terms(data_dir, max_terms=5)
    log_run(
        data_dir / "runs.jsonl",
        slot=slot,
        arm=selected.arm,
        lens=selected.lens,
        topic=getattr(target_summary, "topic", "") or "",
        topic_hint=topic_hint,
        trend_terms=trend_terms,
        output_file=str(final_path),
        extra={
            "hook_id": getattr(selected, "hook_id", ""),
            "hook_cat": getattr(selected, "hook_cat", ""),
            "topic_primary": getattr(target_summary, "topic_primary", "DevTools"),
            "topic_secondary": getattr(target_summary, "topic_secondary", ""),
        },
    )

    topic_memory = append_topic_memory(
        topic_memory,
        getattr(target_summary, "topic", "") or "",
        getattr(target_summary, "topic_kr", "") or "",
    )
    save_topic_memory(topic_memory_path, topic_memory, rolling_days=14)

    return final_path, candidate_paths


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    print("[model routing]", file=sys.stderr)
    print(f"summarize={get_model('summarize')}", file=sys.stderr)
    print(f"write={get_model('write')}", file=sys.stderr)
    print(f"critic={get_model('critic')}", file=sys.stderr)
    try:
        final_file, candidate_files = run_once(project_root, slot=args.slot, seed=args.seed)
    except Exception as exc:
        print(f"[pipeline] ERROR  {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅  Final post: {final_file}")
    print("📝  Candidates:")
    for p in candidate_files:
        print(f"   - {p}")


if __name__ == "__main__":
    main()
