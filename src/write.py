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
  ## 핵심 주장 (Thesis)
  ## 무엇이 달라졌고 왜 중요한가
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
from urllib.request import Request, urlopen

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
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_TIMEOUT = 90  # seconds (longer for full blog+thread)


def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
    """Call OpenAI-compatible chat completion API via urllib."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = json.dumps({
        "model": _OPENAI_MODEL,
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


def generate_text(prompt: str) -> str:
    """Generate text via LLM. Falls back to returning the prompt on error."""
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
        )
    except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError) as exc:
        print(f"[write] WARN  LLM call failed ({exc}); using template fallback",
              file=sys.stderr)
        return prompt


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
_QUOTE_TRANSLATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def _normalise_quotes(text: str) -> str:
    return text.translate(_QUOTE_TRANSLATION)


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
            candidates.append(phrase + " announced a concrete change")

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
        capped = _cap_phrase(capped, topic, topic_kr or "이 주제", ignore_case=True)
    if topic_kr:
        capped = _cap_phrase(capped, topic_kr, "이 주제", ignore_case=False)
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


# ---------------------------------------------------------------------------
# Template builders (fallback when no LLM)
# ---------------------------------------------------------------------------

def _build_hook(hook: str, topic: str, topic_hint: str, evidence: list[str]) -> str:
    hint_ref = ""
    if topic_hint and topic_hint.lower() not in topic.lower():
        hint_ref = f" '{topic_hint}'의 흐름 속에서"
    ev_snippet = ""
    if evidence:
        ev_snippet = f" {evidence[0].rstrip('.')}라는 사실이 이를 뒷받침한다."

    if hook == "Q":
        return (
            f"'{topic}'이라는 키워드가 타임라인을 채우고 있지만, "
            f"실제로 업무 루프를 바꾼 팀은 얼마나 될까?{hint_ref}\n"
            f"도구를 도입하는 것과, 도구가 일하게 만드는 것은 완전히 다른 문제다.{ev_snippet}"
        )
    if hook == "S":
        return (
            f"지난 분기{hint_ref} '{topic}' 관련 프로젝트 수가 급증했다.\n"
            f"하지만 증가한 것은 '시도'이지 '성공'이 아니다.{ev_snippet}"
        )
    return (
        f"'{topic}'을 도입했다가 3개월 만에 롤백한 팀이 늘고 있다.{hint_ref}\n"
        f"기술이 문제가 아니라, 적용 방식이 문제였다.{ev_snippet}"
    )


def _build_thesis(hook: str, lens_id: str, topic: str, domain_angle: str) -> str:
    lens = LENSES[lens_id]
    if hook == "Q":
        opinion = (f"'{topic}'의 가치는 기능 자체가 아니라, "
                   f"반복 가능한 자동화 루프를 얼마나 빠르게 만들 수 있느냐에 달려 있다.")
    elif hook == "S":
        opinion = (f"'{topic}' 채택률이 높아진 것은 기술 성숙도 때문이 아니다. "
                   f"기존 워크플로우의 비용이 임계점을 넘었기 때문이다.")
    else:
        opinion = (f"'{topic}'이 실패하는 대부분의 이유는 기술적 한계가 아니라, "
                   f"측정 없는 도입과 설계 없는 확장이다.")
    return "\n".join([
        "## 핵심 주장 (Thesis)", "",
        opinion, "",
        f"> {lens['label']} 관점의 핵심 질문: {lens['question']}",
        f"> 이 글은 이 질문을 {domain_angle}의 맥락에서 검증한다.",
    ])


def _build_why_it_matters(topic: str, facts: list[str], domain_angle: str) -> str:
    lines = ["## 무엇이 달라졌고, 왜 중요한가", "",
             "'" + topic + "'을 둘러싼 환경은 최근 6개월간 구조적으로 변했다.", ""]
    if facts:
        lines.append("실전 근거:")
        lines.append("")
        for ev in facts[:4]:
            lines.append("- " + ev)
        lines.append("")
    else:
        lines.append(_NO_EVIDENCE_NOTICE)
        lines.append("")
    lines.extend([
        "이 변화의 핵심은 " + domain_angle + "에 있다. "
        "단순히 새로운 도구가 등장한 것이 아니라, "
        "기존 운영 비용 구조가 더 이상 지속 불가능해졌다는 점이다.", "",
        "결국 문제는 '도입 여부'가 아니라 '어떤 설계로 도입하느냐'다.",
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


def _build_failure_cases(hook: str, lens_id: str, topic: str) -> str:
    lens = LENSES[lens_id]
    lines = ["## 실패 사례와 한계", ""]
    if hook == "Q":
        lines.extend([
            "### 흔한 실패 패턴", "",
            f"1. **측정 없는 도입**: '{topic}'을 도입했지만 성과 지표를 정의하지 않아 "
            f"3개월 후 '효과 불명'으로 폐기되는 경우가 가장 많다.", "",
            "2. **과도한 자동화 범위**: 처음부터 전체 파이프라인을 자동화하려다 "
            "복잡도가 폭발해 유지보수 비용이 수작업보다 높아지는 역설.", "",
            "3. **컨텍스트 비용 무시**: 토큰 비용, API 호출 빈도, 컨텍스트 윈도우 제한을 "
            "고려하지 않고 설계해 운영 비용이 선형 이상으로 증가.",
        ])
    elif hook == "S":
        lines.extend([
            f"### {lens['label']} 관점에서 본 리스크", "",
            "1. **데이터가 보여주지 않는 것**: 채택률 증가가 곧 성공을 의미하지 않는다. "
            "대부분의 팀이 PoC 단계에서 멈추며, 프로덕션 전환율은 20% 미만이다.", "",
            "2. **벤더 종속 리스크**: 특정 API/모델에 의존하는 설계는 "
            "가격 정책 변경 시 전체 파이프라인이 마비된다.", "",
            "3. **운영 복잡도 과소평가**: 자동화 시스템 자체의 모니터링, "
            "장애 대응, 버전 관리에 드는 숨은 비용을 간과하는 경우가 많다.",
        ])
    else:
        lines.extend([
            "### 왜 롤백하게 되는가", "",
            f"1. **설계 없는 확장**: '{topic}'의 초기 성공에 고무되어 범위를 급격히 늘리면 "
            "아키텍처가 감당하지 못한다.", "",
            "2. **팀 역량과의 불일치**: 도구의 복잡도가 팀의 운영 역량을 초과하면 "
            "자동화가 오히려 병목이 된다.", "",
            "3. **피드백 루프 부재**: 결과를 측정하지 않으면 "
            "개선 방향을 알 수 없고, 동일한 실패를 반복한다.",
        ])
    return "\n".join(lines)


def _build_closing(hook: str, topic: str, domain_angle: str) -> str:
    lines = ["## 결론: 남는 한 마디", ""]
    if hook == "Q":
        lines.append(
            f"'{topic}'의 진짜 가치는 '무엇을 할 수 있느냐'가 아니라, "
            f"'무엇을 반복할 수 있느냐'에 있다. "
            f"{domain_angle}을 이해한 팀만이 자동화를 '운영'으로 전환할 수 있다."
        )
    elif hook == "S":
        lines.append(
            f"숫자는 방향을 보여주지만, 도착지를 보장하지 않는다. "
            f"'{topic}'의 데이터를 읽을 때는 항상 '{domain_angle}'이라는 "
            f"필터를 함께 적용해야 왜곡을 피할 수 있다."
        )
    else:
        lines.append(
            f"실패는 기술 자체의 문제가 아니다. "
            f"'{topic}'이 실패하는 이유는 대부분 측정 부재와 설계 부재로 귀결된다. "
            f"작게 시작하고, 측정하고, 반복하라. "
            f"{domain_angle}은 이 루프 위에서만 의미를 갖는다."
        )
    return "\n".join(lines)


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
        "- 800~1200단어 분량의 분석 에세이 (뉴스 요약 금지)\n"
        '- 제목/본문에서 "' + topic + '" 원문 표현은 최대 2회. 이후엔 "' + t_kr + '" 또는 동의어 사용.\n'
        '- "정리", "최근 동향", "소개" 표현 사용 금지\n'
        "- 플레이스홀더 문장 금지 — 모든 문장이 구체적이고 실질적이어야 함\n"
        "- 반드시 다음 구조를 포함:\n"
        "  ## 핵심 주장 (Thesis)\n"
        "  ## 무엇이 달라졌고 왜 중요한가\n"
        "  ## 실전 체크리스트\n"
        "  ## 실패 사례와 한계\n"
        "  ## 결론: 남는 한 마디\n"
        "- 체크리스트는 `- [ ]` 형식으로 3~7개 항목\n"
        "- 서두는 2줄짜리 훅으로 시작, 그 다음 `---` 구분선\n"
        "- 다른 초안과 완전히 다른 관점/사례/비유를 사용할 것\n\n"
        "=== 파트 2: 영어 X 스레드 ===\n\n"
        "규칙:\n"
        "- 7~9개 트윗, 각 260자 이하\n"
        "- 번호 형식: 1/ 2/ 3/ ...\n"
        "- 1번: 강한 훅 (topic과 같은 표현 반복 금지)\n"
        "- 반드시 포함: 구체적 사례 1개, 체크리스트 트윗 1개, 한계/실패 트윗 1개\n"
        "- 마지막 트윗: CTA (call to action)\n"
        "- 블로그 글을 요약하지 말고, 독립적으로 읽히는 스레드로 작성\n\n"
        "=== 출력 형식 ===\n\n"
        "## BLOG ARTICLE (KR)\n\n"
        "<한국어 블로그 글 전체>\n\n"
        "---\n\n"
        "## X THREAD (EN copy/paste)\n\n"
        "<영어 X 스레드>"
    )

    # Try LLM first
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            result = _call_llm(
                system_prompt=(
                    "당신은 한국어 테크 블로그 작가이자 영어 X(Twitter) 콘텐츠 전략가입니다. "
                    "통찰력 있는 분석 에세이와 임팩트 있는 트윗 스레드를 작성합니다."
                ),
                user_prompt=prompt,
                max_tokens=3500,
            )
            if len(result) >= 900:
                result = _cap_topic_repetitions(result, topic, t_kr, limit=2)
                return result
            print("[write] WARN  LLM output too short (%d chars); using template" % len(result),
                  file=sys.stderr)
        except (URLError, KeyError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            print("[write] WARN  LLM call failed (%s); using template fallback" % exc,
                  file=sys.stderr)

    # Template fallback
    hook_text = _build_hook(hook, t_kr, topic_hint, facts)
    thesis = _build_thesis(hook, lens_id, t_kr, domain_angle)
    why = _build_why_it_matters(t_kr, facts, domain_angle)
    checklist = _build_checklist(structure, t_kr)
    failures = _build_failure_cases(hook, lens_id, t_kr)
    closing = _build_closing(hook, t_kr, domain_angle)

    blog_section = "\n\n".join([
        "## BLOG ARTICLE (KR)", "",
        hook_text, "---",
        thesis, why, checklist, failures, closing,
    ])

    x_thread = _build_x_thread_template(t_kr, lens_id, hook, domain_angle)
    x_section = "## X THREAD (EN copy/paste)\n\n" + x_thread
    full_text = blog_section + "\n\n---\n\n" + x_section
    return _cap_topic_repetitions(full_text, topic, t_kr, limit=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_drafts(
    summary: TopicSummary,
    count: int = 5,
    topic_hint: str = "",
    research_patterns: dict | None = None,
) -> List[Draft]:
    """Create *count* draft articles with unique (Arm, Lens) combos.

    Each draft includes a Korean blog article and an English X thread.
    Each draft gets a unique narrative_angle (hook template) for diversity.
    """
    all_combos = [(arm, lens) for arm in ARMS for lens in LENS_IDS]
    chosen = random.sample(all_combos, min(count, len(all_combos)))

    evidence = list(getattr(summary, "key_points", []) or [])

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

    drafts: list[Draft] = []
    for idx, (arm, lens_id) in enumerate(chosen):
        hook, structure = arm.split("-")
        # Recompute lens-specific Korean title per-draft for H1 heading diversity.
        lens_label = LENSES[lens_id]["label"]
        per_draft_kr = _heuristic_topic_kr(summary.topic, lens_label)
        title = per_draft_kr  # Korean headline ≤45 chars; used as H1 by render.py

        narrative_angle = hook_templates[idx] if idx < len(hook_templates) else ""

        body = _build_article(
            hook, structure, lens_id, summary.topic,
            topic_hint, evidence, research_patterns,
            narrative_angle=narrative_angle,
            topic_kr=per_draft_kr,
            topic_angle=topic_angle,
            source_urls=summary.source_urls,
            summary_key_points=evidence,
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
