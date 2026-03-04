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
            {
                "role": "system",
                "content": (
                    "You are a concise technology writing critic. "
                    "Return strict JSON only."
                ),
            },
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
        "Analyze this tech draft for insight and voice quality.\n"
        "Topic: " + (topic_kr or "") + "\n"
        "Lens: " + (lens_label or "") + "\n\n"
        "Return ONLY JSON with keys: punchline, whats_boring, whats_strong, "
        "rewrite_instructions, risk_flags.\n"
        "- whats_boring: 1-3 bullets\n"
        "- whats_strong: 1-3 bullets\n"
        "- rewrite_instructions: 3-7 bullets\n"
        "- risk_flags: 0-3 bullets about hallucination/too generic/too rigid\n"
        "No markdown.\n\n"
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
