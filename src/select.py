"""select.py – Epsilon-greedy bandit for draft selection with expanded keys.

Loads/saves bandit state from data/bandit_state.json.
Epsilon = 0.15 → 15 % random exploration.

Scoring (bounded, additive):
  Score(draft) = clamp(ev(arm))
               + clamp(ev(arm|lens))
               + clamp(ev(term:t1)) + clamp(ev(term:t2))
               + clamp(ev(arm|lens|term:t1)) + clamp(ev(arm|lens|term:t2))
               + domain_bonus   (0.15 if title/body contains a domain term)

  ev(key) = total_reward / max(1, impressions)   from state["keys"]
  clamp range: [-5, +5] per component

NOTE: No top-level imports from write.py to avoid stdlib select collision.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path


# Arm list duplicated here to avoid top-level import of write.py.
# Must stay in sync with write.ARMS.
ARMS = ["Q-T", "Q-C", "Q-K", "S-T", "S-C", "S-K", "R-T", "R-C", "R-K"]
LENS_IDS = ["L1", "L2", "L3", "L4", "L5", "L6"]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+_]{2,}")

# ---------------------------------------------------------------------------
# Domain terms (duplicated from write.py to avoid circular import)
# ---------------------------------------------------------------------------

DOMAIN_TERMS: set[str] = {
    "vibe coding", "agentic", "agentic coding", "workflow", "md", "markdown",
    "mcp", "model context protocol", "skills", "tool calling", "tool-calling",
    "orchestration", "multi-agent", "multi agent", "local ai", "on-device",
    "npu", "ane", "apple neural engine", "mlx", "prompt", "context",
    "context window", "compression", "registry",
}

_DOMAIN_BONUS = 0.15  # small bounded bonus for on-niche drafts


def _has_domain_term(text: str) -> bool:
    """Check if text contains at least one DOMAIN_TERMS entry."""
    low = text.lower()
    return any(dt in low for dt in DOMAIN_TERMS)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def default_state() -> dict:
    return {
        "weights": {arm: 1.0 for arm in ARMS},
        "counts":  {arm: 0 for arm in ARMS},
        "keys": {},  # expanded key tracking
    }


def load_bandit_state(file_path: str | Path) -> dict:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return default_state()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_state()

    state = default_state()
    state["weights"].update(data.get("weights", {}))
    state["counts"].update(data.get("counts", {}))
    state["keys"] = data.get("keys", {})
    return state


def save_bandit_state(state: dict, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_EPSILON = 0.15
_EV_DEFAULT = 0.0   # expected value for unseen keys
_EV_MIN = -5.0      # clamp lower bound
_EV_MAX = 5.0       # clamp upper bound


def _key_ev(state: dict, key: str) -> float:
    """Expected value for a key: total_reward / max(1, impressions).

    Returns 0.0 for unseen keys (neutral, doesn't bias toward default).
    Clamped to [_EV_MIN, _EV_MAX] to prevent runaway scores.
    """
    entry = state.get("keys", {}).get(key)
    if entry is None or entry.get("impressions", 0) == 0:
        return _EV_DEFAULT
    raw = entry["total_reward"] / max(1, entry["impressions"])
    return max(_EV_MIN, min(_EV_MAX, raw))


def _extract_topic_tokens(topic_hint: str) -> list[str]:
    """Extract unique tokens from topic_hint for term-key lookup."""
    if not topic_hint:
        return []
    seen = set()
    tokens = []
    for m in _WORD_RE.finditer(topic_hint):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens[:5]


def score_draft(draft, state: dict, topic_hint: str = "") -> float:
    """Compute a bounded composite score for a draft.

    Components (all clamped to [-5, +5]):
      1. ev(arm)                  – arm-only expected value
      2. ev(arm|lens)             – arm+lens combo expected value
      3. ev(term:t1) + ev(term:t2) – term expected values (up to 2)
      4. ev(arm|lens|term:t1) + ev(arm|lens|term:t2) – full combo (up to 2)
      5. domain_bonus             – +0.15 if title/body contains domain term
      6. ev(hook:<hook_id>)       – hook template expected value
      7. ev(hookcat:<hook_cat>)   – hook category expected value
      8. ev(hook:<id>|arm:<arm>)  – hook×arm combo
      9. ev(hook:<id>|lens:<lens>) – hook×lens combo

    Parameters
    ----------
    draft : write.Draft  (has .arm, .lens, .title, .body, .hook_id, .hook_cat)
    state : bandit state dict
    topic_hint : optional topic hint string for term matching
    """
    arm = getattr(draft, "arm", "")
    lens = getattr(draft, "lens", "")

    # 1. arm-level
    arm_ev = _key_ev(state, arm)

    # 2. arm|lens
    arm_lens_key = f"{arm}|{lens}"
    arm_lens_ev = _key_ev(state, arm_lens_key)

    score = arm_ev + arm_lens_ev

    # 3+4. term and combo keys
    tokens = _extract_topic_tokens(topic_hint)
    for tok in tokens[:2]:
        term_key = f"term:{tok}"
        score += _key_ev(state, term_key)

        combo_key = f"{arm}|{lens}|term:{tok}"
        score += _key_ev(state, combo_key)

    # 5. Domain term bonus
    title = getattr(draft, "title", "")
    body = getattr(draft, "body", "")
    if _has_domain_term(f"{title} {body}"):
        score += _DOMAIN_BONUS

    # 6-9. Hook-based scoring
    hook_id = getattr(draft, "hook_id", "")
    hook_cat = getattr(draft, "hook_cat", "")
    if hook_id:
        score += _key_ev(state, hook_id)
        score += _key_ev(state, f"{hook_id}|arm:{arm}")
        score += _key_ev(state, f"{hook_id}|lens:{lens}")
    if hook_cat:
        score += _key_ev(state, f"hookcat:{hook_cat}")

    return score


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def choose_best_draft(drafts, state: dict, topic_hint: str = ""):
    """Epsilon-greedy selection using expanded bandit scoring.

    With probability ε, pick a random draft (explore).
    Otherwise, pick the draft with the highest composite score (exploit).
    Deterministic when random is seeded.

    Parameters
    ----------
    drafts : list[write.Draft]
    state : dict  – bandit state with 'weights' and 'keys'
    topic_hint : str – optional topic hint for term-based scoring
    """
    if not drafts:
        raise ValueError("No drafts to select from")

    if random.random() < _EPSILON:
        return random.choice(drafts)

    return max(drafts, key=lambda d: score_draft(d, state, topic_hint))
