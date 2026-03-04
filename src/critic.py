"""critic.py - Draft critic for insight/voice improvements."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from env_loader import get_model


_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_CRITIC_TIMEOUT = 35

_CRITIC_SYSTEM_PROMPT = """You are a ruthless editor for Korean tech insight posts.

TASK:
Given a draft, return strict JSON only.

YOU CHECK (HARD):
1) Opener punchiness (no boring intro).
2) Any generic reusable lines (flag them).
3) Micro-story completeness: team size + constraint + failure mode + fix + numeric outcome.
4) At least 3 concrete operational details exist:
   (cost cap / retry rule / timeout / SLO / rollout / logs / alerts / canary / queue)
5) 2–3 quotable hot-take lines exist.
6) Banned phrase presence must be flagged and rewritten:
   - 요즘, 최근 몇 년, 신호는 이미, 무엇이 달라졌고, 이 주제, 정리, 소개, 동향
   - 관련 기술/기업:, 출처:, URL 컨텍스트:
   - source headline, appears in the source headline, 소스 텍스트에, 단서가 반복된다
7) List generic sentences that could fit any topic.

OUTPUT JSON SCHEMA (STRICT):
{
  "punchline": "string (one sharp line opener suggestion)",
  "whats_boring": ["string", "... up to 3"],
  "whats_strong": ["string", "... up to 3"],
  "rewrite_instructions": ["string", "... 3 to 7 items"],
  "risk_flags": ["string", "... up to 3"]
}

RULES:
- No markdown.
- No extra keys.
- rewrite_instructions must be concrete actions, not vague advice.

Now critique the draft and output JSON only."""


def _fallback_critique() -> dict:
    return {
        "punchline": "핵심 주장은 선명하게, 문장은 더 짧게 다듬는다.",
        "whats_boring": ["도입 문장이 평이함"],
        "whats_strong": ["실행 관점이 있음"],
        "rewrite_instructions": [
            "첫 두 문장을 대비 구조로 시작",
            "메커니즘을 비용/워크플로우 관점으로 명확화",
            "중복 문장을 제거하고 한 문단 한 주장 유지",
        ],
        "risk_flags": [],
    }


def _normalise_list(value: Any, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        s = str(v).strip()
        if s:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def _call_critic_llm(user_prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = json.dumps({
        "model": get_model("critic"),
        "messages": [
            {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }).encode("utf-8")

    req = Request(
        _OPENAI_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(req, timeout=_CRITIC_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def critique_draft(draft_text: str, topic_kr: str, lens_label: str) -> dict:
    """Return compact JSON critique for a draft."""
    if not draft_text.strip():
        return _fallback_critique()

    prompt = (
        "Topic: " + (topic_kr or "") + "\n"
        "Lens: " + (lens_label or "") + "\n\n"
        "Draft:\n" + draft_text[:6000]
    )
    try:
        raw = _call_critic_llm(prompt)
        parsed = json.loads(raw)
        out = {
            "punchline": str(parsed.get("punchline", "")).strip(),
            "whats_boring": _normalise_list(parsed.get("whats_boring"), 3),
            "whats_strong": _normalise_list(parsed.get("whats_strong"), 3),
            "rewrite_instructions": _normalise_list(parsed.get("rewrite_instructions"), 7),
            "risk_flags": _normalise_list(parsed.get("risk_flags"), 3),
        }
        if not out["rewrite_instructions"]:
            out["rewrite_instructions"] = _fallback_critique()["rewrite_instructions"]
        if not out["punchline"]:
            out["punchline"] = _fallback_critique()["punchline"]
        return out
    except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError, TypeError, ValueError):
        return _fallback_critique()
