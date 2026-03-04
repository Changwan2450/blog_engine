"""write.py – Generate blog-first content drafts using (Arm × Lens).

Arms  = Hook (Q/S/R) × Structure (T/C/K)  →  9 arms
Lenses = L1–L6 from docs/arms.md

Each draft produces:
  1) Korean BLOG ARTICLE (800–1200 words, analytical prose)
  2) English X THREAD (7–9 tweets, copy/paste ready)

LLM integration:
  - Calls OpenAI-compatible API via urllib (stdlib only, no external deps)
  - Env var: OPENAI_API_KEY
  - Model: gpt-4o-mini (default, override with OPENAI_MODEL)
  - Falls back to template text if API key absent or call fails

Blog article structure:
  ## BLOG ARTICLE (KR)
  Hook (2 lines)
  ---
  ## TL;DR
  ## Why it matters
  ## 핵심 주장 (Thesis)
  ## 변화의 핵심
  ## 실전 체크리스트
  ## 실패 사례와 한계
  ## 결론

X thread structure:
  ## X THREAD (EN copy/paste)
  1/ strong hook
  2/-8/ content tweets
  9/ CTA
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from env_loader import get_model
from summarize import TopicSummary, _heuristic_topic_kr


# ---------------------------------------------------------------------------
# Arms: Hook × Structure
# ---------------------------------------------------------------------------

ARMS = ["Q-T", "Q-C", "Q-K", "S-T", "S-C", "S-K", "R-T", "R-C", "R-K"]

# ---------------------------------------------------------------------------
# Lenses (L1–L6)
# ---------------------------------------------------------------------------

LENSES = {
    "L1": {"label": "Cost",        "question": "얼마나 비용을 줄일 수 있는가?"},
    "L2": {"label": "Failure",     "question": "사람들이 실패하는 이유는 무엇인가?"},
    "L3": {"label": "Operations",  "question": "실제 서비스 환경에서 어떤 문제가 생기는가?"},
    "L4": {"label": "Alternative", "question": "같은 결과를 더 간단하게 만드는 방법은?"},
    "L5": {"label": "Measurement", "question": "성능을 어떻게 측정할 것인가?"},
    "L6": {"label": "SysDesign",   "question": "시스템을 어떻게 나눠야 하는가?"},
}

LENS_IDS = list(LENSES.keys())


# ---------------------------------------------------------------------------
# Domain terms – shared niche vocabulary for the blog identity
# ---------------------------------------------------------------------------

DOMAIN_TERMS: set[str] = {
    "vibe coding", "agentic", "agentic coding", "workflow", "md", "markdown",
    "mcp", "model context protocol", "skills", "tool calling", "tool-calling",
    "orchestration", "multi-agent", "multi agent", "local ai", "on-device",
    "npu", "ane", "apple neural engine", "mlx", "prompt", "context",
    "context window", "compression", "registry",
}

_DOMAIN_CLAUSES = [
    "— 오케스트레이션/스킬 관점",
    "— 컨텍스트 비용 관점",
    "— 에이전틱 워크플로우 관점",
    "— 로컬 AI/온디바이스 관점",
    "— MCP/도구 호출 관점",
]

_DOMAIN_ANGLES = [
    "AI 오케스트레이션 파이프라인의 비용 구조",
    "컨텍스트 윈도우 압축과 토큰 비용 절감",
    "MCP 기반 도구 호출과 스킬 등록 패턴",
    "에이전틱 워크플로우 자동화의 실전 한계",
    "로컬 AI 추론과 온디바이스 배포 전략",
    "멀티 에이전트 시스템의 설계 트레이드오프",
]


def _has_domain_term(text: str) -> bool:
    low = text.lower()
    return any(dt in low for dt in DOMAIN_TERMS)


def _pick_domain_clause(seed_text: str) -> str:
    idx = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % len(_DOMAIN_CLAUSES)
    return _DOMAIN_CLAUSES[idx]


def _pick_domain_angle(seed_text: str) -> str:
    idx = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % len(_DOMAIN_ANGLES)
    return _DOMAIN_ANGLES[idx]


@dataclass
class Draft:
    arm: str
    lens: str
    title: str
    body: str
    topic_hint: str = ""
    hook_id: str = ""
    hook_cat: str = ""


# ---------------------------------------------------------------------------
# Hook identity helpers
# ---------------------------------------------------------------------------

_HOOK_CAT_RULES = [
    (["fail", "mistake", "wrong", "breaks", "rollback"], "failure"),
    (["hidden", "nobody", "secret", "hides"], "hidden_cost"),
    (["cost", "price", "expensive", "cheap", "tax"], "cost"),
    (["measure", "metric", "number", "data"], "measurement"),
    (["stop", "don't", "avoid", "never"], "contrarian"),
    (["replace", "alternative", "simpler", "instead"], "replacement"),
    (["checklist", "question", "test", "before you"], "practical"),
    (["everyone", "most", "nobody", "misunderstand"], "sweeping"),
    (["demo", "production", "real-world", "reality"], "demo_vs_prod"),
    (["90 days", "adoption", "300%", "rate"], "data_driven"),
]


def _classify_hook_cat(template: str) -> str:
    """Derive a hook category from a hook template string."""
    tl = template.lower()
    for keywords, cat in _HOOK_CAT_RULES:
        if any(k in tl for k in keywords):
            return cat
    return "insight"


def _make_hook_id(template: str, all_templates: list[str]) -> str:
    """Create a stable hook_id like 'hook:tpl_07'."""
    try:
        idx = all_templates.index(template)
    except ValueError:
        idx = int(hashlib.md5(template.encode()).hexdigest(), 16) % 9999
    return f"hook:tpl_{idx:02d}"


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

_GENERIC_TITLE_PREFIXES = ["요약", "정리", "최근 동향", "뉴스", "업데이트"]

_SECTION_SIGNALS = [
    "체크리스트", "실전", "실패", "측정", "지표",
    "설계", "아키텍처", "운영", "비용", "대안",
    "thread", "tweet", "1/",
]

_MIN_BODY_LENGTH = 900


def draft_passes_quality(draft: Draft) -> bool:
    """Cheap quality gate for a draft."""
    title_stripped = draft.title.strip()
    for prefix in _GENERIC_TITLE_PREFIXES:
        if title_stripped.lower().startswith(prefix.lower()):
            return False

    body_lower = draft.body.lower()
    signal_count = sum(1 for sig in _SECTION_SIGNALS if sig in body_lower)
    if signal_count < 2:
        return False

    if len(draft.body) < _MIN_BODY_LENGTH:
        return False

    return True


# ---------------------------------------------------------------------------
# LLM integration — OpenAI-compatible API via urllib
# ---------------------------------------------------------------------------

_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_OPENAI_TIMEOUT = 90  # seconds (longer for full blog+thread)
_ENABLE_DETEMPLATE_GUARD = True

_WRITER_SYSTEM_PROMPT = """You write Korean tech insight posts that feel like a sharp engineer’s take someone would share on X.

GOAL:
- Make it interesting even to non-experts.
- Slightly provocative, but still helpful.
- No corporate blog tone. No textbook tone.

STYLE RULES (HARD):
1) Start with a punchy opener (1–2 lines). No “요즘/최근/신호는 이미/무엇이 달라졌고/이 주제/정리/소개/동향”.
2) Include a micro-story (5–10 lines):
   - a team (size), a constraint (budget/latency/SLO), a failure (timeout/rollback/cost 폭주), and the lesson.
3) Include exactly 2 quotable hot-take lines (짧고 세게).
4) Every paragraph must mention at least one concrete object:
   - cost cap, retry rule, timeout, SLO, rollout steps, cache, queue, logs, alerts, canary, budget.
5) If evidence is weak, you MUST say exactly:
   "근거가 약하다. 그래서 이렇게 확인해라:"
   and provide exactly 2 verification steps.
6) Never output metadata fragments:
   - "관련 기술/기업:" "출처:" "URL 컨텍스트:" "source headline" "appears in the source headline"
7) Avoid placeholders like "해당 기술", "이 스택" 남발. Use the subject naturally.

OUTPUT FORMAT (HARD):
## TL;DR
- Exactly 1 sentence.

## Why it matters
- 2–3 sentences. Natural Korean.

## Article
- 5–10 short paragraphs.
- Must include the micro-story.
- Must include 2–3 hot-take lines.

## 근거/출처
- 2–4 bullet links: "domain: url"
- 1–2 short evidence snippets (quotes or numeric windows), if available.

INPUT YOU RECEIVE:
- topic_en, topic_kr, lens_label
- key_claims (claim + evidence + confidence)
- source_urls, source_domains
- topic_primary/topic_secondary

Write the post now."""


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 3000,
    stage: str = "write",
) -> str:
    """Call OpenAI-compatible chat completion API via urllib."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = json.dumps({
        "model": get_model(stage),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
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

    with urlopen(req, timeout=_OPENAI_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def generate_text(prompt: str, hook_line: str | None = None) -> str:
    """Generate text via LLM. Falls back to returning the prompt on error."""
    if hook_line:
        prompt = (
            "The article MUST start with this hook line exactly:\n"
            + hook_line
            + "\n\nDo not rewrite it. Continue the article naturally after it.\n\n"
            + prompt
        )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return prompt
    try:
        return _call_llm(
            system_prompt=(
                "당신은 한국어 테크 블로그 작가입니다. "
                "항상 통찰력 있는 분석 에세이를 작성하며 요약이나 뉴스 전달은 하지 않습니다."
            ),
            user_prompt=prompt,
            stage="write",
        )
    except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError) as exc:
        print(f"[write] WARN  LLM call failed ({exc}); using template fallback",
              file=sys.stderr)
        return prompt


def generate_text_with_hook(prompt: str, hook_line: str | None = None) -> str:
    """Backward-compatible helper for hook-aware text generation."""
    return generate_text(prompt, hook_line=hook_line)


def rewrite_with_feedback(
    draft_body: str,
    topic_kr: str,
    lens_label: str,
    feedback: dict,
) -> str:
    """Rewrite draft body with critic feedback using writer model."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return draft_body

    instr = [str(x).strip() for x in feedback.get("rewrite_instructions", []) if str(x).strip()]
    boring = [str(x).strip() for x in feedback.get("whats_boring", []) if str(x).strip()]
    strong = [str(x).strip() for x in feedback.get("whats_strong", []) if str(x).strip()]

    prompt = (
        "다음 초안을 더 날카롭고 통찰 중심으로 고쳐 써라.\n"
        "주제: " + topic_kr + "\n"
        "렌즈: " + lens_label + "\n\n"
        "유지할 강점:\n- " + ("\n- ".join(strong[:3]) if strong else "핵심 논지의 선명함") + "\n\n"
        "줄일 부분:\n- " + ("\n- ".join(boring[:3]) if boring else "반복적 문장") + "\n\n"
        "개선 지시:\n- " + ("\n- ".join(instr[:7]) if instr else "문장 밀도를 높이고 중복을 제거") + "\n\n"
        "제약:\n"
        "- 섹션 구조(TL;DR, Why it matters, Article, 근거/출처)는 유지\n"
        "- 메타데이터 라벨(출처:, URL 컨텍스트:, 관련 기술/기업:) 금지\n"
        "- 사실 기반 톤 유지\n\n"
        "초안:\n" + draft_body
    )
    try:
        return _call_llm(
            system_prompt="당신은 날카로운 기술 칼럼 편집자다. 군더더기 없이 핵심을 남긴다.",
            user_prompt=prompt,
            max_tokens=2800,
            stage="write",
        )
    except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError):
        return draft_body


# ---------------------------------------------------------------------------
# Evidence extraction + topic repetition cap
# ---------------------------------------------------------------------------

_NO_EVIDENCE_NOTICE = (
    "현재 공개된 정보는 제한적이며, "
    "아래는 관찰 가능한 패턴 중심으로 정리한다."
)

_NUM_CTX_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:engineers?|employees?|users?|teams?|companies|repos?|requests?|tokens?|"
    r"%|percent|ms|seconds?|days?|months?|billion|million|thousand|B\b|M\b|K\b)",
    re.IGNORECASE,
)
_QUOTE_CTX_RE = re.compile(r'"([^"]{10,100})"')
_COMPANY_CTX_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&\-]{1,}(?:\s+[A-Z][A-Za-z0-9&\-]{1,}){0,3})\b"
)
_GENERIC_EVIDENCE_RE = re.compile(
    r"(important|trend|update|overview|introduction|요약|정리|소개|동향)",
    re.IGNORECASE,
)
_METADATA_FRAGMENT_RE = re.compile(r"(관련\s*기술/기업:|출처:|URL\s*컨텍스트:)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EVIDENCE_ARTIFACT_RE = re.compile(
    r"(appears in the source headline|source headline|소스 텍스트에|단서가 반복된다)",
    re.IGNORECASE,
)
_QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def _normalise_quotes(text: str) -> str:
    return text.translate(_QUOTE_TRANSLATION)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[:limit - 1].rstrip() + "…"


def _clean_evidence_text(text: str) -> str:
    t = re.sub(_EVIDENCE_ARTIFACT_RE, "", text or "")
    t = re.sub(r"\s+", " ", t).strip(" ,.;:-")
    return t


def _narrative_evidence_sentence(ev: str, idx: int) -> str:
    e = _clean_evidence_text(ev)
    if not e:
        e = "외부 공개 사례"
    variants = [
        f"{e} 케이스를 보면 병목이 코드보다 운영 규칙에서 먼저 터진다.",
        f"{e}에서도 비용 상한과 복구 규칙을 같이 설계한 팀이 장애 시간을 줄였다.",
        f"{e} 사례는 retry와 timeout을 늦게 고치면 queue 적체가 바로 커진다는 점을 보여준다.",
        f"{e}에서 확인된 패턴은 비슷하다: 지표 없이 확장하면 rollback 빈도부터 오른다.",
        f"{e} 같은 공개 사례를 보면 성능 논쟁보다 운영 제약 설계가 더 먼저 이슈가 된다.",
    ]
    return variants[idx % len(variants)]


def _detemplate_blog_text(text: str) -> str:
    if not _ENABLE_DETEMPLATE_GUARD:
        return text

    banned_patterns = [
        r"승부는\s*기능\s*비교가\s*아니라",
        r"도입\s*속도보다",
        r"\b먼저다\b",
        r"예를\s*들어",
        r"요즘",
        r"최근\s*몇\s*년",
        r"무엇이\s*달라졌고",
        r"이\s*주제",
    ]
    replacements = [
        "현장에서 실제로 갈리는 지점은 따로 있다",
        "대부분 팀이 놓치는 지점은 운영 설계다",
        "문제의 핵심은 모델이 아니라 실행 구조다",
    ]

    out = text
    for p in banned_patterns:
        if re.search(p, out):
            out = re.sub(p, random.choice(replacements), out)
    return out


def _detemplate_output_blog_section(full_text: str) -> str:
    if not _ENABLE_DETEMPLATE_GUARD:
        return full_text
    marker = "## X Thread"
    if marker in full_text:
        head, tail = full_text.split(marker, 1)
        return _detemplate_blog_text(head) + marker + tail
    return _detemplate_blog_text(full_text)


def _extract_evidence(summary_text: str) -> list[str]:
    """Extract high-signal evidence (quotes, numeric claims, company mentions)."""
    text = _normalise_quotes(summary_text or "")
    if not text.strip():
        return []

    candidates: list[str] = []

    for m in _QUOTE_CTX_RE.finditer(text):
        quote = '"' + m.group(1).strip() + '"'
        if len(quote) >= 12:
            candidates.append(quote)

    for m in _NUM_CTX_RE.finditer(text):
        start = max(0, m.start() - 35)
        end = min(len(text), m.end() + 45)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip(" -,:;.")
        if len(snippet) >= 18:
            candidates.append(snippet)

    for m in _COMPANY_CTX_RE.finditer(text):
        phrase = m.group(0).strip()
        if len(phrase) >= 4 and phrase.lower() not in {"the", "this", "that"}:
            candidates.append(phrase)

    if not candidates:
        return []

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for c in candidates:
        if _METADATA_FRAGMENT_RE.search(c):
            continue
        if _GENERIC_EVIDENCE_RE.search(c):
            continue

        norm = " ".join(c.lower().split())
        if norm in seen:
            continue
        seen.add(norm)

        score = 0
        if _QUOTE_CTX_RE.search(c):
            score += 4
        if _NUM_CTX_RE.search(c):
            score += 4
        if _COMPANY_CTX_RE.search(c):
            score += 2
        if len(c) < 14:
            score -= 1

        if score > 0:
            scored.append((score, c[:140]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:3]]


def _cap_topic_repetitions(text: str, topic: str, topic_kr: str, limit: int = 2) -> str:
    """Cap both English/Korean topic mentions beyond `limit` with quote normalization."""

    def _cap_phrase(s: str, phrase: str, replacement: str, ignore_case: bool) -> str:
        if not phrase or not replacement:
            return s
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(re.escape(_normalise_quotes(phrase)), flags)
        count = [0]

        def _repl(m: re.Match) -> str:
            count[0] += 1
            if count[0] <= limit:
                return m.group(0)
            return replacement

        return pattern.sub(_repl, s)

    capped = _normalise_quotes(text)
    if topic:
        capped = _cap_phrase(capped, topic, topic_kr or "이 스택", ignore_case=True)
    if topic_kr:
        capped = _cap_phrase(capped, topic_kr, "이 스택", ignore_case=False)
    return capped


# ---------------------------------------------------------------------------
# Research patterns loader
# ---------------------------------------------------------------------------

def _load_research_patterns() -> dict:
    """Load research patterns if available."""
    path = Path(__file__).resolve().parents[1] / "data" / "research_patterns.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Title builder
# ---------------------------------------------------------------------------

def _build_title(hook: str, lens_id: str, topic: str, topic_hint: str) -> str:
    lens = LENSES[lens_id]

    if hook == "Q":
        base = f"왜 {topic}만으로는 부족한가 — {lens['label']} 관점의 실전 분석"
    elif hook == "S":
        base = f"{topic}, 숫자가 말해주는 것과 말해주지 않는 것 — {lens['label']} 렌즈"
    else:
        base = f"{topic}이 실패하는 진짜 이유 — {lens['label']} 관점에서 재해석"

    if topic_hint:
        base += f" [{topic_hint}]"

    combined = f"{topic} {topic_hint} {base}"
    if not _has_domain_term(combined):
        base += f" {_pick_domain_clause(topic_hint or topic)}"

    return base


_TITLE_LABEL_SUFFIXES = {
    "측정 방법", "비용의 함정", "실패의 이유", "운영 이슈", "대안 전략",
    "시스템 설계 포인트", "핵심 포인트", "대안", "이유",
}

_PRIMARY_FALLBACK_KR = {
    "AI Infra": "AI 인프라",
    "Models": "AI 에이전트",
    "DevTools": "개발 도구",
    "Chips": "AI 칩",
    "Security": "프롬프트 공격",
    "Web": "웹 자동화",
    "Product": "제품 운영",
    "OpenSource": "오픈소스",
    "Data": "데이터 파이프라인",
    "Business": "운영 전략",
}

_TITLE_BAD_START = {"benefits", "how", "why", "what", "when"}
_TITLE_BAD_END = {"of", "to", "for", "a"}


def _english_subject_to_korean(text: str) -> str:
    low = (text or "").lower()
    if "agent" in low:
        return "AI 에이전트"
    if "prompt" in low and ("attack" in low or "injection" in low):
        return "프롬프트 공격"
    if "multi" in low and "agent" in low:
        return "멀티 에이전트 시스템"
    if "model" in low:
        return "AI 모델"
    return ""


def _title_subject(topic_kr: str, fallback: str = "") -> str:
    t = re.sub(r"\s+", " ", (topic_kr or "").strip())
    t = t.replace("'", "").replace('"', "")
    t = re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9\s]+$", "", t).strip()
    if ":" in t:
        t = t.split(":", 1)[0].strip()
    for suf in _TITLE_LABEL_SUFFIXES:
        if t.endswith(suf):
            t = t[: -len(suf)].rstrip(" -_:/")
            break

    english_words = re.findall(r"\b[A-Za-z]+\b", t)
    first_word = english_words[0].lower() if english_words else ""
    ends_in_bad = bool(re.search(r"\b(of|to|for|a)\s*$", t, re.IGNORECASE))
    low = " " + t.lower() + " "
    reject = (
        (" i " in low)
        or (" my " in low)
        or (len(english_words) > 2)
        or (first_word in _TITLE_BAD_START)
        or ("'s" in t.lower())
        or ends_in_bad
        or bool(re.search(r"\b(a|an|the)\s*$", t, re.IGNORECASE))
    )

    # Cleanup english headline fragments into Korean subject when possible.
    if reject:
        mapped = _english_subject_to_korean(t)
        if mapped:
            t = mapped
            reject = False

    if reject:
        t = ""

    if not t:
        t = (fallback or "").strip()
    if not t:
        t = "AI 에이전트"
    return t[:26].strip()


def _build_claim_title(topic_kr: str, hook: str, fallback_subject: str = "") -> str:
    subj = _title_subject(topic_kr, fallback=fallback_subject)
    if hook == "Q":
        return f"{subj}에서 사람들이 착각하는 것"
    if hook == "S":
        return f"{subj}의 진짜 병목"
    return f"{subj}가 망하는 진짜 이유"


# ---------------------------------------------------------------------------
# Template builders (fallback when no LLM)
# ---------------------------------------------------------------------------

def _build_hook(hook: str, topic: str, topic_hint: str, evidence: list[str]) -> str:
    ev = _clean_evidence_text(evidence[0]) if evidence else "하루 비용 상한, 실패 복구 규칙, 응답 시간 SLO"
    if hook == "Q":
        return (
            f"'{topic}'이 성능 문제라고 믿는다면, 아마 원인을 잘못 찍었다.\n"
            f"진짜 병목은 모델이 아니라 운영 제약이다: {ev}."
        )
    if hook == "S":
        return (
            f"나는 '{topic}' 도입 실패를 모델 품질에서 거의 본 적이 없다.\n"
            f"대부분은 예산 상한, 시간 제한, 장애 복구 규칙을 안 정해서 터진다: {ev}."
        )
    extra = f" 추가 맥락: {topic_hint}." if topic_hint and topic_hint.lower() not in topic.lower() else ""
    return (
        f"네 팀의 '{topic}' 실험이 막혔다면, 코드가 아니라 운영 설계가 비어 있을 가능성이 높다.{extra}\n"
        f"실패 모드는 늘 같다: 비용 폭주, 타임아웃, 롤백 루프."
    )


def _build_thesis(hook: str, lens_id: str, topic: str, domain_angle: str) -> str:
    lens = LENSES[lens_id]
    if hook == "Q":
        opinion = (f"핵심은 기능이 아니라 제약 설계다. "
                   f"예산 상한 1개, 실패 복구 규칙 1개, 지연 한도 1개가 없으면 '{topic}'은 바로 불안정해진다.")
    elif hook == "S":
        opinion = (f"채택 속도보다 중요한 건 운영 단위 비용이다. "
                   f"요청당 비용, 재시도 횟수, 장애 복구 시간을 못 재면 '{topic}'은 실전에서 무너진다.")
    else:
        opinion = (f"실패의 다수는 모델 한계가 아니라 실행 설계 부재다. "
                   f"SLO 없이 확장하면 타임아웃과 롤백이 누적된다.")
    return "\n".join([
        "## 핵심 주장 (Thesis)", "",
        opinion, "",
        "핫테이크: 좋은 프롬프트보다 좋은 롤백 규칙이 매출을 지킨다.",
        "핫테이크: 모델 교체보다 장애 플레이북 1장이 더 싸다.",
        f"> {lens['label']} 관점의 핵심 질문: {lens['question']}",
        f"> 이 글은 이 질문을 {domain_angle}의 맥락에서 검증한다.",
    ])


def _build_why_it_matters(topic: str, facts: list[str], domain_angle: str, weak_evidence: bool = False) -> str:
    lines = ["## 변화의 핵심", "",
             "'" + topic + "'의 승패는 기능 개수보다 운영 제약을 먼저 못 박았는지에서 갈린다.", ""]
    if facts:
        for i, ev in enumerate(facts[:3]):
            lines.append(_narrative_evidence_sentence(ev, i))
        lines.append("")
    if weak_evidence or not facts:
        lines.append(_NO_EVIDENCE_NOTICE)
        lines.append("")
    lines.extend([
        "이 변화의 핵심은 " + domain_angle + "에 있다. "
        "도입 여부보다 운영 비용 구조를 먼저 설계한 팀이 이긴다.", "",
        "핫테이크: 기능 데모는 1시간이면 만들고, 운영 안정성은 1주가 걸린다.",
    ])
    return "\n".join(lines)


def _build_checklist(structure: str, topic: str) -> str:
    lines = ["## 실전 체크리스트", ""]
    if structure == "T":
        lines.extend([
            f"'{topic}'을 실전에 적용하기 전에 반드시 점검할 항목:", "",
            "- [ ] **측정 지표 정의**: 성공/실패를 판단할 지표를 먼저 정한다.",
            "- [ ] **최소 루프 설계**: 단일 반복 가능 루프 1개를 먼저 만든다.",
            "- [ ] **실패 시나리오 문서화**: 장애 시 복구 경로를 설계한다.",
            "- [ ] **비용 상한 설정**: 토큰/API 비용의 일일 상한을 설정한다.",
            "- [ ] **1주 회고 실행**: 도입 7일 후 지표 기반 회고를 수행한다.",
        ])
    elif structure == "C":
        lines.extend([
            f"'{topic}' 운영 안정화를 위한 체크리스트:", "",
            "- [ ] **입력/출력 스키마 고정**: 자동화 대상의 형식을 먼저 고정한다.",
            "- [ ] **소스 3개 이하로 제한**: 수집 소스는 점진적으로 확장한다.",
            "- [ ] **중복 제거 로직 검증**: 정규화 기준을 명문화하고 테스트한다.",
            "- [ ] **비용 대비 품질 측정**: 건당 비용과 결과 품질을 매일 기록한다.",
            "- [ ] **장애 대응 플레이북 작성**: 외부 API 장애 시 fallback을 문서화한다.",
            "- [ ] **주간 성과 리뷰**: 조회수, 좋아요 기반으로 주간 리뷰한다.",
            "- [ ] **자동화 범위 재조정**: 2주마다 범위가 적절한지 재검토한다.",
        ])
    else:
        lines.extend([
            f"'{topic}' 케이스에서 배운 실전 교훈:", "",
            "- [ ] **작은 루프부터 자동화**: 반복 빈도가 높은 단일 작업부터 시작한다.",
            "- [ ] **운영 비용을 설계 시점에 계산**: 배포 후 비용 폭발을 방지한다.",
            "- [ ] **실패 케이스를 학습 데이터로 전환**: 실패 로그를 개선 근거로 활용한다.",
            "- [ ] **피드백 주기를 7일 이내로 유지**: 루프가 길어지면 학습 효과가 감소한다.",
        ])
    return "\n".join(lines)


def _build_failure_cases(hook: str, lens_id: str, topic: str, weak_evidence: bool = False) -> str:
    lens = LENSES[lens_id]
    lines = ["## 실패 사례와 한계", ""]
    lines.extend([
        "마이크로 스토리: 4명 팀이 주간 예산 상한과 2초 지연 제한으로 자동화 루프를 붙였다.",
        "첫 주에 실패 모드는 한 가지였다: 재시도 폭주로 큐가 밀리면서 장애가 연쇄 발생했다.",
        "해결은 화려하지 않았다: 일일 비용 컷오프, 재시도 2회 제한, 수동 fallback 버튼을 넣었다.",
        "교훈은 간단했다: 모델 교체보다 운영 제약 3개를 먼저 고정하는 편이 복구 시간을 줄였다.",
    ])
    if weak_evidence:
        lines.append("")
        lines.append("추가 점검: 근거 밀도가 낮으니 로그 수집 기간(최소 7일)을 먼저 확보해야 한다.")
    lines.append("")
    lines.append(f"{lens['label']} 관점의 리스크: SLO 없는 확장은 성공 사례보다 실패 모드를 더 빨리 증폭시킨다.")
    return "\n".join(lines)


def _build_closing(hook: str, topic: str, domain_angle: str) -> str:
    lines = ["## 결론: 남는 한 마디", ""]
    if hook == "Q":
        lines.append(
            f"'{topic}'의 가치는 기능 목록이 아니라 반복 가능한 운영 루프에서 나온다. "
            f"치트코드: 다음 스프린트에서 비용 상한 1개, 실패 복구 규칙 1개, 지연 한도 1개부터 고정하라."
        )
    elif hook == "S":
        lines.append(
            f"데이터는 방향만 보여주고, 운영 설계가 결과를 만든다. "
            f"치트코드: 7일 로그에서 실패 상위 1개만 제거해도 체감 품질이 먼저 오른다."
        )
    else:
        lines.append(
            f"실패는 보통 기술이 아니라 운영 규칙의 빈칸에서 시작된다. "
            f"치트코드: 주간 배포 전에 '비용-지연-복구' 3개 체크리스트를 통과하지 못하면 배포를 멈춰라."
        )
    return "\n".join(lines)


def _build_tldr(topic_kr: str, key_claims: list[dict]) -> str:
    subj = _title_subject(topic_kr)
    sentence = (
        f"사람들은 {subj}의 성패가 모델 성능 때문이라고 보지만 실제 병목은 비용 상한, "
        "재시도 규칙, 지연 한도 같은 운영 제약이다."
    )
    return "## TL;DR\n\n" + _truncate_text(sentence, 130)


def _build_why_it_matters_fixed(
    topic_kr: str,
    key_claims: list[dict],
    domain_angle: str,
    weak_evidence: bool = False,
) -> str:
    lines = ["## Why it matters", ""]
    openers = [
        f"{topic_kr}는 모델 스펙보다 운영 기준선을 어디에 두는지가 결과를 가른다.",
        f"{topic_kr}는 화려한 데모보다 budget cap, retry count, timeout 같은 기본값 설계가 더 중요하다.",
        f"{topic_kr}에서 큰 차이는 기능 수가 아니라 queue depth와 alert rule을 얼마나 빨리 잡느냐에서 난다.",
        f"{topic_kr}는 속도 싸움처럼 보이지만 실제로는 rollback 규칙과 p95 관리에서 승패가 갈린다.",
    ]
    first_ev = ""
    if key_claims:
        evs = key_claims[0].get("evidence", []) or []
        if evs:
            first_ev = str(evs[0]).strip()
    lines.append(random.choice(openers))
    if first_ev:
        lines.append(_truncate_text(_narrative_evidence_sentence(first_ev, 0), 140))
    if weak_evidence:
        lines.append("근거가 약하다. 그래서 이렇게 확인해라:")
        lines.append("1) 7일간 p95 latency, retry count, queue depth를 수집해 병목 구간을 수치로 고정한다.")
        lines.append("2) canary 10%와 전체 배포를 비교해 rollback rate와 alert precision을 계산한다.")
    closers = [
        domain_angle + " 관점에서 성능 논쟁보다 운영 루프의 재현성이 더 큰 차이를 만든다.",
        domain_angle + " 프레임으로 보면 기능 추가보다 SLO/알림 규칙 고정이 실제 장애 시간을 줄인다.",
        domain_angle + "에서는 빠른 도입보다 계측 가능한 실행 구조를 먼저 갖춘 팀이 덜 흔들린다.",
    ]
    lines.append(_truncate_text(random.choice(closers), 130))
    return "\n".join(lines)


def _build_sources_section(summary_urls: list[str], key_claims: list[dict]) -> str:
    lines = ["## 근거/출처", ""]

    seen_domains: set[str] = set()
    src_lines: list[str] = []
    for u in summary_urls:
        if not u:
            continue
        d = urlparse(u).netloc.replace("www.", "")
        if not d or d in seen_domains:
            continue
        seen_domains.add(d)
        src_lines.append(f"- {d}: {u}")
        if len(src_lines) >= 4:
            break
    if not src_lines:
        src_lines.append("- 공개 링크 기반 근거가 제한적이므로 운영 관찰 중심으로 해석했다.")

    ev_lines: list[str] = []
    for kc in key_claims:
        for ev in kc.get("evidence", []) or []:
            t = str(ev).strip()
            if not t or _METADATA_FRAGMENT_RE.search(t):
                continue
            ev_lines.append("- " + _truncate_text(t, 130))
            if len(ev_lines) >= 2:
                break
        if len(ev_lines) >= 2:
            break

    lines.extend(src_lines[:4])
    if ev_lines:
        lines.append("")
        lines.append("근거 스니펫:")
        lines.extend(ev_lines)
    return "\n".join(lines)


def _is_weak_evidence(key_claims: list[dict], facts: list[str]) -> bool:
    if facts:
        return False
    if not key_claims:
        return True
    strong = 0
    for kc in key_claims:
        conf = str(kc.get("confidence", "")).lower().strip()
        evs = [str(e).strip() for e in kc.get("evidence", []) if str(e).strip()]
        has_num_or_quote = any(re.search(r"\b\d[\d,]*(?:\.\d+)?\b|\"[^\"]+\"", e) for e in evs)
        if conf in {"high", "med"} and has_num_or_quote:
            strong += 1
    return strong == 0


# ---------------------------------------------------------------------------
# X thread template (fallback)
# ---------------------------------------------------------------------------

def _build_x_thread_template(topic: str, lens_id: str, hook: str, domain_angle: str) -> str:
    """Build a template X thread (7 tweets, <=260 chars each)."""
    lens = LENSES[lens_id]
    tweets = [
        f"1/ Why does {topic} keep failing in production? A thread on what nobody talks about. 🧵",
        f"2/ The real issue isn't the tech. It's adopting without measuring. "
        f"Most teams skip defining success metrics before deployment.",
        f"3/ {lens['label']} lens: {lens['question']} "
        f"This question alone filters 80% of bad implementations.",
        f"4/ Concrete example: teams that set daily cost caps on API/token usage "
        f"reduced unexpected bills by 60% in the first month.",
        f"5/ Checklist before you ship:\n"
        f"☐ Define 1 success metric\n☐ Build 1 repeatable loop\n"
        f"☐ Document failure recovery\n☐ Set cost ceiling",
        f"6/ Where it breaks: scaling too fast after initial success. "
        f"The architecture that works for 100 requests collapses at 10,000.",
        f"7/ Bottom line: start small, measure everything, iterate weekly. "
        f"{domain_angle} only matters when built on this foundation. Ship less, learn more.",
    ]
    return "\n\n".join(tweets)


def _parse_thread_tweets(thread_text: str) -> list[str]:
    tweets: list[str] = []
    current: list[str] = []
    for raw in (thread_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d+/", line):
            if current:
                tweets.append(" ".join(current).strip())
                current = []
            current.append(re.sub(r"^\d+/\s*", "", line))
        else:
            current.append(line)
    if current:
        tweets.append(" ".join(current).strip())

    if not tweets:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", thread_text or "") if c.strip()]
        tweets = chunks
    return tweets


def _truncate_tweet(body: str, num: int, max_len: int = 240) -> str:
    prefix = f"{num}/ "
    allowed = max_len - len(prefix)
    text = body.strip()
    if len(prefix + text) <= max_len:
        return prefix + text
    return prefix + _truncate_text(text, max(8, allowed))


def _enforce_x_limits(thread_text: str) -> str:
    tweets = _parse_thread_tweets(thread_text)
    if len(tweets) < 7:
        tweets.extend(["Measure one loop at a time before scaling."] * (7 - len(tweets)))
    if len(tweets) > 9:
        tweets = tweets[:9]

    # Links only in last tweet.
    found_links: list[str] = []
    clean_tweets: list[str] = []
    for t in tweets:
        links = _URL_RE.findall(t)
        found_links.extend(links)
        clean = _URL_RE.sub("", t)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        clean_tweets.append(clean)

    if found_links:
        link_tail = " ".join(found_links[:2])
        clean_tweets[-1] = (clean_tweets[-1] + " " + link_tail).strip()

    out = [_truncate_tweet(body, i + 1, max_len=240) for i, body in enumerate(clean_tweets)]
    return "\n\n".join(out)


def _enforce_x_limits_on_full_output(text: str) -> str:
    marker = "## X Thread"
    if marker in text:
        head, tail = text.split(marker, 1)
    else:
        legacy = "## X THREAD (EN copy/paste)"
        if legacy not in text:
            return text
        head, tail = text.split(legacy, 1)
    limited = _enforce_x_limits(tail.strip())
    return head.rstrip() + "\n\n" + marker + "\n\n" + limited


# ---------------------------------------------------------------------------
# Assemble full article body (blog + X thread)
# ---------------------------------------------------------------------------

def _build_article(
    hook: str,
    structure: str,
    lens_id: str,
    topic: str,
    topic_hint: str,
    evidence: list[str],
    research_patterns: dict | None = None,
    narrative_angle: str = "",
    topic_kr: str = "",
    topic_angle: str = "",
    source_urls: list[str] | None = None,
    summary_key_points: list[str] | None = None,
    key_claims: list[dict] | None = None,
    hook_line: str | None = None,
) -> str:
    """Assemble blog article + X thread.

    When OPENAI_API_KEY is set: sends one rich prompt to LLM.
    Fallback: template sections.
    """
    t_kr = topic_kr or topic
    domain_angle = _pick_domain_angle(topic + "|" + lens_id + "|" + topic_hint)
    lens = LENSES[lens_id]

    # Build concrete facts from key_points + evidence
    combined_text = " ".join((summary_key_points or []) + (evidence or []))
    facts = _extract_evidence(combined_text)

    # --- Research patterns context ---
    research_ctx = ""
    if research_patterns:
        kw = research_patterns.get("keywords", [])[:15]
        hp = research_patterns.get("hook_patterns", [])[:8]
        if kw:
            research_ctx += "\n\n참고 키워드 (자연스럽게 녹일 것): " + ", ".join(kw)
        if hp:
            research_ctx += "\n인기 훅 스타일: " + ", ".join(hp)

    narrative_line = ""
    if narrative_angle:
        concrete = narrative_angle.replace("{topic}", t_kr).replace("{metric}", "ROI")
        narrative_line = (
            '\n\n이 글의 서사적 각도 (hook): "' + concrete + '"\n'
            "이 문장을 그대로 쓰지 말고, 이 방향성에 맞는 독창적인 훅을 작성할 것."
        )

    angle_line = ("\n\n분석 각도: " + topic_angle) if topic_angle else ""

    hook_style = {
        "Q": "도발적인 질문으로 시작: 왜 이것이 기대만큼 효과가 없는가?",
        "S": "데이터/사례로 시작: 실제 도입 결과와 기대 사이의 간극을 드러낸다",
        "R": "반론으로 시작: '이게 효과 있다'는 통념을 정면으로 반박한다",
    }
    structure_focus = {
        "T": "TLDR/핵심 우선 구조. 결론을 먼저 주고, 이유를 뒤에 설명한다",
        "C": "체크리스트 중심 구조. 실행 가능한 항목(3~7개)이 핵심이다",
        "K": "케이스 스터디 구조. 실제 사례의 원인-결과-교훈을 연결한다",
    }

    safe_claims = list(key_claims or [])
    weak_evidence = _is_weak_evidence(safe_claims, facts)

    if facts:
        facts_block = "\n\n배경 정보 (분석에 녹여낼 것):\n" + "\n".join("- " + f for f in facts)
    else:
        facts_block = '\n\n배경 정보 없음. 다음 문장으로 한계를 명시할 것: "' + _NO_EVIDENCE_NOTICE + '"'

    hint_line = ""
    if topic_hint and topic_hint.lower() not in topic.lower():
        hint_line = "\n\n추가 렌즈: '" + topic_hint + "' 관점도 자연스럽게 포함할 것."

    prompt = (
        "다음 조건에 따라 두 가지 콘텐츠를 한 번에 작성하세요.\n\n"
        "주제(영문): " + topic + "\n"
        "주제(한국어 제목으로 사용): " + t_kr + "\n"
        "분석 렌즈: " + lens["label"] + " 관점 — " + lens["question"] + "\n"
        "도메인 각도: " + domain_angle + "\n"
        "훅 스타일: " + hook_style.get(hook, hook_style["Q"]) + "\n"
        "본문 구조: " + structure_focus.get(structure, structure_focus["T"])
        + narrative_line + angle_line + facts_block + hint_line + research_ctx + "\n\n"
        "=== 파트 1: 한국어 블로그 글 ===\n\n"
        "규칙:\n"
        "- 톤: 약간 까칠하지만 도움이 되는 X 인사이트 톤 (기업 홍보체/교과서체 금지)\n"
        "- 시작 1~2줄은 '네가 틀렸을 수도 있다' 급의 강한 펀치로 시작\n"
        "- 뉴스 요약 금지. 숨은 메커니즘 중심의 통찰 글로 작성\n"
        "- 인프라/개발 워크플로우/비용 구조/인센티브 중 최소 2개 축으로 설명\n"
        "- 미시 사례 1개 필수: 한 팀, 한 제약(예산/시간), 한 실패, 한 교훈\n"
        "- 짧고 도발적인 hot take 문장 정확히 2개 포함 (X에 바로 인용 가능한 길이)\n"
        "- 마지막은 바로 써먹는 cheat code 한 줄로 마무리\n"
        "- 각 문단에는 최소 1개의 구체 객체를 넣을 것: 숫자/제약/실패모드/비용/시간/운영 디테일 중 하나\n"
        '- 제목/본문에서 "' + topic + '" 원문 표현은 최대 2회. 이후엔 "' + t_kr + '" 또는 동의어 사용.\n'
        '- 금지 표현: "신호는 이미", "요즘", "최근 몇 년", "무엇이 달라졌고", "이 주제", "정리", "소개", "동향"\n'
        '- 메타데이터 라벨 금지: "출처:", "URL 컨텍스트:", "관련 기술/기업:"\n'
        "- 약한 도발/가벼운 유머는 허용하지만 사실성은 유지\n"
        "- 근거가 약하면 반드시 이 문장을 포함: '근거가 약하다. 그래서 이렇게 확인해라:' + 검증 단계 2개\n"
        "- 플레이스홀더 문장 금지 — 모든 문장이 구체적이고 실질적이어야 함\n"
        "- 반드시 다음 구조를 포함:\n"
        "  ## TL;DR\n"
        "  ## Why it matters\n"
        "  ## Article\n"
        "  ## 근거/출처\n"
        "- 다른 초안과 겹치지 않게 관점/비유를 바꿀 것\n\n"
        + ("- 첫 줄은 반드시 다음 문장을 그대로 사용할 것: " + hook_line + "\n" if hook_line else "")
        + "=== 파트 2: 영어 X 스레드 ===\n\n"
        "규칙:\n"
        "- 7~9개 트윗, 각 240자 이하\n"
        "- 번호 형식: 1/ 2/ 3/ ...\n"
        "- 1번: 강한 훅 (topic과 같은 표현 반복 금지)\n"
        "- 반드시 포함: 구체적 사례 1개, 체크리스트 트윗 1개, 한계/실패 트윗 1개\n"
        "- 링크는 마지막 트윗(또는 마지막 2개 트윗)에만 배치\n"
        "- 마지막 트윗: CTA (call to action)\n"
        "- 블로그 글을 요약하지 말고, 독립적으로 읽히는 스레드로 작성\n\n"
        "=== 출력 형식 ===\n\n"
        "## BLOG ARTICLE (KR)\n\n"
        "<한국어 블로그 글 전체>\n\n"
        "---\n\n"
        "## X Thread\n\n"
        "<영어 X 스레드>"
    )

    # Try LLM first
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            result = _call_llm(
                system_prompt=_WRITER_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=3500,
                stage="write",
            )
            if len(result) >= 900:
                if hook_line:
                    blog_marker = "## BLOG ARTICLE (KR)"
                    if blog_marker in result:
                        head, rest = result.split(blog_marker, 1)
                        rest = rest.lstrip()
                        if not rest.startswith(hook_line):
                            result = head + blog_marker + "\n\n" + hook_line + "\n\n" + rest
                result = _cap_topic_repetitions(result, topic, t_kr, limit=2)
                result = _enforce_x_limits_on_full_output(result)
                return _detemplate_output_blog_section(result)
            print("[write] WARN  LLM output too short (%d chars); using template" % len(result),
                  file=sys.stderr)
        except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            print("[write] WARN  LLM call failed (%s); using template fallback" % exc,
                  file=sys.stderr)

    # Template fallback
    hook_text = _build_hook(hook, t_kr, topic_hint, facts)
    tldr = _build_tldr(t_kr, safe_claims)
    why_fixed = _build_why_it_matters_fixed(t_kr, safe_claims, domain_angle, weak_evidence=weak_evidence)
    thesis = _build_thesis(hook, lens_id, t_kr, domain_angle)
    why = _build_why_it_matters(t_kr, facts, domain_angle, weak_evidence=weak_evidence)
    checklist = _build_checklist(structure, t_kr)
    failures = _build_failure_cases(hook, lens_id, t_kr, weak_evidence=weak_evidence)
    closing = _build_closing(hook, t_kr, domain_angle)
    sources_sec = _build_sources_section(source_urls or [], safe_claims)

    blog_section = "\n\n".join([
        "## BLOG ARTICLE (KR)", "",
        hook_line or hook_text, "---",
        tldr, why_fixed,
        "## Article", "",
        thesis, why, checklist, failures, closing,
        sources_sec,
    ])

    x_thread = _enforce_x_limits(_build_x_thread_template(t_kr, lens_id, hook, domain_angle))
    if hook_line:
        tweets = _parse_thread_tweets(x_thread)
        if tweets:
            tweets[0] = hook_line
        x_thread = _enforce_x_limits("\n\n".join(f"{i+1}/ {t}" for i, t in enumerate(tweets)))
    x_section = "## X Thread\n\n" + x_thread
    full_text = blog_section + "\n\n---\n\n" + x_section
    full_text = _cap_topic_repetitions(full_text, topic, t_kr, limit=2)
    return _detemplate_output_blog_section(full_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_drafts(
    summary: TopicSummary,
    count: int = 5,
    topic_hint: str = "",
    research_patterns: dict | None = None,
    hook_line: str | None = None,
) -> List[Draft]:
    """Create *count* draft articles with unique (Arm, Lens) combos.

    Each draft includes a Korean blog article and an English X thread.
    Each draft gets a unique narrative_angle (hook template) for diversity.
    """
    all_combos = [(arm, lens) for arm in ARMS for lens in LENS_IDS]
    chosen = random.sample(all_combos, min(count, len(all_combos)))

    evidence = list(getattr(summary, "key_points", []) or [])
    key_claims = list(getattr(summary, "key_claims", []) or [])

    # Select unique hook templates per draft for diversity
    hook_templates: list[str] = []
    if research_patterns:
        hook_templates = list(research_patterns.get("hook_templates", []))
    if len(hook_templates) < count:
        # Pad with built-in defaults
        from research import _BUILTIN_HOOK_TEMPLATES
        for ht in _BUILTIN_HOOK_TEMPLATES:
            if ht not in hook_templates:
                hook_templates.append(ht)
    random.shuffle(hook_templates)

    topic_angle = getattr(summary, "topic_angle", "") or ""
    topic_primary = getattr(summary, "topic_primary", "") or "DevTools"
    primary_fallback = _PRIMARY_FALLBACK_KR.get(topic_primary, "AI 에이전트")

    drafts: list[Draft] = []
    for idx, (arm, lens_id) in enumerate(chosen):
        hook, structure = arm.split("-")
        # Recompute lens-specific Korean title per-draft for H1 heading diversity.
        lens_label = LENSES[lens_id]["label"]
        per_draft_kr = _heuristic_topic_kr(summary.topic, lens_label)
        title = _build_claim_title(per_draft_kr, hook, fallback_subject=primary_fallback)

        narrative_angle = hook_templates[idx] if idx < len(hook_templates) else ""

        body = _build_article(
            hook, structure, lens_id, summary.topic,
            topic_hint, evidence, research_patterns,
            narrative_angle=narrative_angle,
            topic_kr=per_draft_kr,
            topic_angle=topic_angle,
            source_urls=summary.source_urls,
            summary_key_points=evidence,
            key_claims=key_claims,
            hook_line=hook_line,
        )

        hid  = _make_hook_id(narrative_angle, hook_templates) if narrative_angle else ""
        hcat = _classify_hook_cat(narrative_angle) if narrative_angle else ""

        drafts.append(Draft(
            arm=arm,
            lens=lens_id,
            title=title,
            body=body,
            topic_hint=topic_hint,
            hook_id=hid,
            hook_cat=hcat,
        ))

    return drafts
