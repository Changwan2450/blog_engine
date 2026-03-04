"""learn.py – Run logging, reward calculation, and multi-key bandit weight updates.

Expanded bandit keys:
  - arm_lens:   "{arm}|{lens}"         (e.g. "Q-T|L3")
  - term:       "term:{term}"          (e.g. "term:MCP")
  - combo:      "{arm}|{lens}|term:{t}" (e.g. "Q-T|L3|term:MCP")

Each key tracks: {"impressions": int, "total_reward": float, "last_updated": str}

Reward formula:
  Phase 1 (new blog, low traffic — views < 20):
    reward = 1.0 if published (human selected this draft)
    reward is ignored (0.0) if views < 20 and not explicitly published
  Phase 2 (when views >= 20):
    reward = log1p(views) * 0.25 + likes * 0.05 + comments * 0.1

Global decay (DECAY = 0.985):
  Applied to ALL state["keys"][*]["total_reward"] BEFORE adding new reward.
  This provides recency bias – old winners fade over ~50 updates.

Term attribution filter:
  Only attribute to terms from trend_terms.json that pass a quality filter:
    - ALL-CAPS (MCP, RAG, GPU)
    - Contains digit, hyphen, or underscore (GPT-4, vLLM, tool-calling)
  Rejects TitleCase single-word noise like "Face", "Apple", etc.
  Max 2 terms per update.

Domain term preference:
  When attributing terms, domain-related terms are sorted first.
  This keeps the learning loop anchored to the blog's niche identity.

Weight update uses EMA:
  new_weight = (1 - α) * old_weight + α * reward
  α = 0.3
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_select_module():
    """Load our local select.py without conflicting with stdlib select."""
    mod_path = Path(__file__).with_name("select.py")
    spec = importlib.util.spec_from_file_location("blog_select", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sel = _load_select_module()
load_bandit_state = _sel.load_bandit_state
save_bandit_state = _sel.save_bandit_state

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+_]{2,}")

# ---------------------------------------------------------------------------
# Domain terms (must stay in sync with write.py / select.py DOMAIN_TERMS)
# ---------------------------------------------------------------------------

DOMAIN_TERMS: set[str] = {
    "vibe coding", "agentic", "agentic coding", "workflow", "md", "markdown",
    "mcp", "model context protocol", "skills", "tool calling", "tool-calling",
    "orchestration", "multi-agent", "multi agent", "local ai", "on-device",
    "npu", "ane", "apple neural engine", "mlx", "prompt", "context",
    "context window", "compression", "registry",
}

# Single-word domain tokens for fast matching against candidate terms
_DOMAIN_TOKENS: set[str] = set()
for _dt in DOMAIN_TERMS:
    for _tok in _dt.split():
        if len(_tok) >= 2:
            _DOMAIN_TOKENS.add(_tok.lower())


def _is_domain_term(term: str) -> bool:
    """Return True if term overlaps with any domain vocabulary."""
    return term.lower() in _DOMAIN_TOKENS


# ---------------------------------------------------------------------------
# Project-root-relative path normalisation
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel_path(raw: str) -> str:
    """Convert an absolute or relative path to a project-root-relative string.

    Used for robust matching of output_file between runs.jsonl records and
    learn.py CLI input.
    """
    try:
        p = Path(raw).resolve()
        return str(p.relative_to(_PROJECT_ROOT))
    except (ValueError, OSError):
        # Already relative or outside project → return as-is
        return raw


# ---------------------------------------------------------------------------
# Run logging  →  data/runs.jsonl
# ---------------------------------------------------------------------------

def log_run(
    runs_path: str | Path,
    *,
    slot: str,
    arm: str,
    lens: str,
    output_file: str,
    topic: str = "",
    topic_hint: str = "",
    trend_terms: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Append a JSON-lines record for this pipeline run."""
    path = Path(runs_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "slot": slot,
        "arm": arm,
        "lens": lens,
        "topic": topic,
        "topic_hint": topic_hint,
        "trend_terms": trend_terms or [],
        "output_file": output_file,
        "output_file_rel": _rel_path(output_file),
    }
    if extra:
        record.update(extra)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Reward calculation (Option B – log-normalised)
# ---------------------------------------------------------------------------

_MIN_VIEWS_THRESHOLD = 20  # Below this, views are unreliable (new blog phase)
_PUBLISH_BONUS = 1.0       # Phase 1: reward for human selection/publish


def compute_reward(
    views: int = 0,
    likes: int = 0,
    comments: int = 0,
    published: bool = False,
) -> float:
    """Compute reward using phase-based strategy for new blogs.

    Phase 1 (views < 20, new blog):
      If published=True: return PUBLISH_BONUS (human selected this draft)
      Otherwise: return 0.0 (insufficient signal to learn from)

    Phase 2 (views >= 20):
      reward = log1p(views) * 0.25 + likes * 0.05 + comments * 0.1
      If published: add PUBLISH_BONUS
    """
    # Phase 1: low traffic — rely on human selection signal only
    if views < _MIN_VIEWS_THRESHOLD:
        return _PUBLISH_BONUS if published else 0.0

    # Phase 2: engagement-based reward
    reward = math.log1p(views) * 0.25 + likes * 0.05 + comments * 0.1
    if published:
        reward += _PUBLISH_BONUS
    return reward


# ---------------------------------------------------------------------------
# Run record lookup (robust path matching)
# ---------------------------------------------------------------------------

def _find_run_record(runs_path: str | Path, output_file: str) -> dict | None:
    """Find the most recent run record matching output_file.

    Matches against both exact string and project-root-relative path.
    """
    path = Path(runs_path)
    if not path.exists():
        return None

    needle_rel = _rel_path(output_file)

    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Skip metric_update records
        if rec.get("slot") == "metric_update":
            continue
        rec_file = rec.get("output_file", "")
        rec_rel = rec.get("output_file_rel", "") or _rel_path(rec_file)
        if rec_file == output_file or rec_rel == needle_rel:
            return rec
    return None


# ---------------------------------------------------------------------------
# Term attribution – quality filter + domain preference
# ---------------------------------------------------------------------------

# Term quality patterns:
#   - ALL-CAPS: MCP, RAG, GPU, NVIDIA
#   - Contains digit/hyphen/underscore: GPT-4, vLLM, tool-calling, GPT-4o
_TERM_QUALITY_RE = re.compile(
    r"^[A-Z][A-Z0-9\-\+_]{1,}$"    # all-caps (len≥2)
    r"|"
    r"[\d\-_]"                       # contains digit, hyphen, or underscore
)


def _is_quality_term(term: str) -> bool:
    """Return True if term passes quality filter (not generic TitleCase noise)."""
    if len(term) < 2:
        return False
    return bool(_TERM_QUALITY_RE.search(term))


def _extract_attributed_terms(
    topic: str,
    topic_hint: str,
    trend_terms_path: str | Path,
    max_terms: int = 2,
) -> list[str]:
    """Return up to *max_terms* quality trend terms for reward attribution.

    Only attributes to terms that:
    1. Exist in trend_terms.json top_terms
    2. Pass quality filter (all-caps or contains digit/hyphen/underscore)

    Domain preference: domain-related terms are sorted first so that
    if we have both domain and non-domain quality terms, learnings
    stay anchored to the blog's niche.
    """
    path = Path(trend_terms_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    candidates = [
        (rec.get("term") or "").strip()
        for rec in data.get("top_terms", [])
    ]

    # Filter: must pass quality check
    quality = [t for t in candidates if t and _is_quality_term(t)]

    # Sort: domain terms first, then non-domain (stable sort preserves rank)
    quality.sort(key=lambda t: (0 if _is_domain_term(t) else 1))

    # Prefer terms that overlap with the topic/topic_hint text
    topic_lower = f"{topic} {topic_hint}".lower()
    matched = [t for t in quality if t.lower() in topic_lower]
    if matched:
        return matched[:max_terms]

    # Fallback: top quality terms regardless of topic overlap
    return quality[:max_terms]


# ---------------------------------------------------------------------------
# Global decay
# ---------------------------------------------------------------------------

_DECAY = 0.985  # applied to all total_reward each learn invocation


def _apply_global_decay(state: dict) -> None:
    """Decay all existing total_reward values to provide recency bias.

    After ~50 updates with no new reward, a key's total_reward halves.
    Formula: total_reward *= DECAY
    Also decay legacy weights dict.
    """
    for entry in state.get("keys", {}).values():
        entry["total_reward"] = round(entry.get("total_reward", 0.0) * _DECAY, 6)

    for k in list(state.get("weights", {}).keys()):
        old_w = state["weights"][k]
        # Decay toward 1.0 (the default/neutral weight)
        state["weights"][k] = round(1.0 + (old_w - 1.0) * _DECAY, 4)


# ---------------------------------------------------------------------------
# Multi-key bandit update
# ---------------------------------------------------------------------------

_ALPHA = 0.3  # learning rate


def _ensure_key(state: dict, key: str) -> dict:
    """Ensure a key exists in state['keys'] with default structure."""
    state.setdefault("keys", {})
    if key not in state["keys"]:
        state["keys"][key] = {
            "impressions": 0,
            "total_reward": 0.0,
            "last_updated": "",
        }
    return state["keys"][key]


def _update_key(state: dict, key: str, reward: float) -> None:
    """Update a single expanded bandit key with EMA."""
    entry = _ensure_key(state, key)
    entry["impressions"] += 1
    entry["total_reward"] = round(entry["total_reward"] + reward, 6)
    entry["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Also update legacy weights dict for backward compat
    old_w = state.get("weights", {}).get(key, 1.0)
    new_w = (1 - _ALPHA) * old_w + _ALPHA * reward
    state.setdefault("weights", {})[key] = round(new_w, 4)


def update_bandit_expanded(
    bandit_path: str | Path,
    arm: str,
    lens: str,
    terms: list[str],
    reward: float,
    hook_id: str = "",
    hook_cat: str = "",
) -> None:
    """Update bandit state for all expanded keys.

    1. Apply global decay to all existing keys
    2. Update arm, arm|lens, term:X, arm|lens|term:X keys
    3. Update hook:<id>, hookcat:<cat>, hook:<id>|arm:<arm>, hook:<id>|lens:<lens>
    """
    state = load_bandit_state(bandit_path)

    # Global decay BEFORE adding new reward
    _apply_global_decay(state)

    # Key: arm (legacy)
    _update_key(state, arm, reward)
    state.setdefault("counts", {})
    state["counts"][arm] = state["counts"].get(arm, 0) + 1

    # Key: arm|lens
    arm_lens_key = f"{arm}|{lens}"
    _update_key(state, arm_lens_key, reward)

    # Key: term:X  (only quality terms)
    for t in terms:
        term_key = f"term:{t}"
        _update_key(state, term_key, reward)

    # Key: arm|lens|term:X
    for t in terms:
        combo_key = f"{arm}|{lens}|term:{t}"
        _update_key(state, combo_key, reward)

    # Hook keys (skip gracefully if missing)
    if hook_id:
        _update_key(state, hook_id, reward)
        _update_key(state, f"{hook_id}|arm:{arm}", reward)
        _update_key(state, f"{hook_id}|lens:{lens}", reward)
    if hook_cat:
        _update_key(state, f"hookcat:{hook_cat}", reward)

    save_bandit_state(state, bandit_path)


# ---------------------------------------------------------------------------
# Metrics CSV logging
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "timestamp", "output_file", "views", "likes", "comments",
    "reward", "arm", "lens", "term1", "term2", "hook_id", "hook_cat",
]


def _append_metrics_csv(
    csv_path: str | Path,
    output_file: str,
    views: int,
    likes: int,
    comments: int,
    reward: float,
    arm: str,
    lens: str,
    terms: list[str],
    hook_id: str = "",
    hook_cat: str = "",
) -> None:
    """Append a row to data/metrics.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    term1 = terms[0] if len(terms) > 0 else ""
    term2 = terms[1] if len(terms) > 1 else ""

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CSV_HEADER)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            output_file,
            views,
            likes,
            comments,
            round(reward, 4),
            arm,
            lens,
            term1,
            term2,
            hook_id,
            hook_cat,
        ])


# ---------------------------------------------------------------------------
# Public: record_metrics (full pipeline)
# ---------------------------------------------------------------------------

def record_metrics(
    data_dir: str | Path,
    output_file: str,
    views: int,
    likes: int,
    comments: int,
    published: bool = False,
) -> float:
    """Full metric recording pipeline.

    1. Lookup run record by output_file in runs.jsonl
    2. Compute reward (log-normalised)
    3. Extract attributed terms (quality-filtered, domain-preferred)
    4. Update expanded bandit state (with global decay)
    5. Append to metrics.csv
    6. Log to runs.jsonl

    Returns the computed reward.
    """
    data = Path(data_dir)
    runs_path = data / "runs.jsonl"
    bandit_path = data / "bandit_state.json"
    trend_path = data / "trend_terms.json"
    csv_path = data / "metrics.csv"

    # 1. Lookup
    rec = _find_run_record(runs_path, output_file)
    if rec is None:
        print(f"[learn] ERROR  No run record found for: {output_file}", file=sys.stderr)
        print(f"[learn]        Relative: {_rel_path(output_file)}", file=sys.stderr)
        print("[learn]        Tip: provide the exact path from pipeline output", file=sys.stderr)
        arm = "Q-T"
        lens = "L1"
        topic = ""
        topic_hint = ""
        hook_id = ""
        hook_cat = ""
    else:
        arm = rec.get("arm", "Q-T")
        lens = rec.get("lens", "L1")
        topic = rec.get("topic", "")
        topic_hint = rec.get("topic_hint", "")
        hook_id = rec.get("hook_id", "")
        hook_cat = rec.get("hook_cat", "")

    # 2. Reward (phase-based)
    reward = compute_reward(views, likes, comments, published=published)

    # 3. Terms (quality-filtered, domain-preferred, max 2)
    terms = _extract_attributed_terms(topic, topic_hint, trend_path, max_terms=2)

    # 4. Update bandit (with global decay + hook keys)
    update_bandit_expanded(bandit_path, arm, lens, terms, reward,
                           hook_id=hook_id, hook_cat=hook_cat)

    # 5. CSV
    _append_metrics_csv(csv_path, output_file, views, likes, comments, reward,
                        arm, lens, terms, hook_id=hook_id, hook_cat=hook_cat)

    # 6. Log
    log_run(
        runs_path,
        slot="metric_update",
        arm=arm,
        lens=lens,
        output_file=output_file,
        topic=topic,
        topic_hint=topic_hint,
        trend_terms=terms,
        extra={"views": views, "likes": likes, "comments": comments,
               "reward": reward, "hook_id": hook_id, "hook_cat": hook_cat},
    )

    return reward


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Record blog post metrics and update bandit state"
    )
    parser.add_argument("--output_file", "--output-file", required=True,
                        help="Path to the final .md file (must match runs.jsonl)")
    parser.add_argument("--views", type=int, default=0)
    parser.add_argument("--likes", type=int, default=0)
    parser.add_argument("--comments", type=int, default=0)
    parser.add_argument("--published", action="store_true", default=False,
                        help="Flag: this draft was selected and published")
    parser.add_argument("--data-dir", default=None, help="Path to data/ directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"

    reward = record_metrics(
        data_dir=data_dir,
        output_file=args.output_file,
        views=args.views,
        likes=args.likes,
        comments=args.comments,
        published=args.published,
    )
    print(f"✅  Recorded: output={args.output_file}  reward={reward:.4f}")
