"""write.py – Generate insight/essay-style tech blog drafts using (Arm × Lens).

Arms  = Hook (Q/S/R) × Structure (T/C/K)  →  9 arms
Lenses = L1–L6 from docs/arms.md

generate_text() is a placeholder for an LLM call—replace it later.

Article structure:
  # Title
  Hook (2 lines)
  ---
  ## Thesis
  ## What changed / why it matters
  ## Practical takeaways (checklist)
  ## Failure cases / caveats
  ## Closing insight

Hard rules:
  - Not a news summary
  - Must contain a clear thesis (one strong opinion)
  - Must include one practical checklist (3–7 items)
  - Must mention limitations or failure cases
  - Avoid generic phrases: "정리", "최근 동향", "소개"
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List

from summarize import TopicSummary


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

# Domain-anchoring phrases injected into narrative sections
_DOMAIN_ANGLES = [
    "AI 오케스트레이션 파이프라인의 비용 구조",
    "컨텍스트 윈도우 압축과 토큰 비용 절감",
    "MCP 기반 도구 호출과 스킬 등록 패턴",
    "에이전틱 워크플로우 자동화의 실전 한계",
    "로컬 AI 추론과 온디바이스 배포 전략",
    "멀티 에이전트 시스템의 설계 트레이드오프",
]


def _has_domain_term(text: str) -> bool:
    """Check if text contains at least one DOMAIN_TERMS entry."""
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
    lens: str      # e.g. "L3"
    title: str
    body: str
    topic_hint: str = ""  # retained for metadata/traceability


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

_GENERIC_TITLE_PREFIXES = ["요약", "정리", "최근 동향", "뉴스", "업데이트"]

_SECTION_SIGNALS = [
    "체크리스트", "실전", "실패", "측정", "지표",
    "설계", "아키텍처", "운영", "비용", "대안",
]

_MIN_BODY_LENGTH = 900


def draft_passes_quality(draft: Draft) -> bool:
    """Cheap quality gate for a draft.

    Rejects if:
      - Title starts with a generic word
      - Body has fewer than 2 section signals
      - Body is shorter than 900 chars
    """
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
# Placeholder LLM function (swap later)
# ---------------------------------------------------------------------------

def generate_text(prompt: str) -> str:
    """Placeholder: return the prompt itself.

    Replace this with an LLM API call that accepts a prompt string and
    returns the generated text string.
    """
    return prompt


# ---------------------------------------------------------------------------
# Insight/essay builders
# ---------------------------------------------------------------------------

def _build_title(hook: str, lens_id: str, topic: str, topic_hint: str) -> str:
    """Build an opinionated, insight-style title (never starts with generic words)."""
    lens = LENSES[lens_id]

    if hook == "Q":
        base = f"왜 {topic}만으로는 부족한가 — {lens['label']} 관점의 실전 분석"
    elif hook == "S":
        base = f"{topic}, 숫자가 말해주는 것과 말해주지 않는 것 — {lens['label']} 렌즈"
    else:  # R
        base = f"{topic}이 실패하는 진짜 이유 — {lens['label']} 관점에서 재해석"

    if topic_hint:
        base += f" [{topic_hint}]"

    # Domain anchoring: append clause if no domain term present
    combined = f"{topic} {topic_hint} {base}"
    if not _has_domain_term(combined):
        base += f" {_pick_domain_clause(topic_hint or topic)}"

    return base


def _build_hook(hook: str, topic: str, topic_hint: str, evidence: list[str]) -> str:
    """2-line hook that sets up the thesis. Opinionated, not a summary."""
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
    """## Thesis – one strong opinion."""
    lens = LENSES[lens_id]

    if hook == "Q":
        opinion = (
            f"'{topic}'의 가치는 기능 자체가 아니라, "
            f"반복 가능한 자동화 루프를 얼마나 빠르게 만들 수 있느냐에 달려 있다."
        )
    elif hook == "S":
        opinion = (
            f"'{topic}' 채택률이 높아진 것은 기술 성숙도 때문이 아니다. "
            f"기존 워크플로우의 비용이 임계점을 넘었기 때문이다."
        )
    else:
        opinion = (
            f"'{topic}'이 실패하는 대부분의 이유는 기술적 한계가 아니라, "
            f"측정 없는 도입과 설계 없는 확장이다."
        )

    return "\n".join([
        "## 핵심 주장 (Thesis)",
        "",
        opinion,
        "",
        f"> {lens['label']} 관점의 핵심 질문: {lens['question']}",
        f"> 이 글은 이 질문을 {domain_angle}의 맥락에서 검증한다.",
    ])


def _build_why_it_matters(topic: str, evidence: list[str], domain_angle: str) -> str:
    """## What changed / why it matters – evidence-backed narrative."""
    lines = [
        "## 무엇이 달라졌고, 왜 중요한가",
        "",
        f"'{topic}'을 둘러싼 환경은 최근 6개월간 구조적으로 변했다.",
        "",
    ]

    # Use evidence bullets as supporting data, not as a list dump
    if evidence:
        lines.append("실전 근거:")
        lines.append("")
        for i, ev in enumerate(evidence[:4], 1):
            lines.append(f"{i}. {ev}")
        lines.append("")

    lines.extend([
        f"이 변화의 핵심은 {domain_angle}에 있다. "
        "단순히 새로운 도구가 등장한 것이 아니라, "
        "기존 운영 비용 구조가 더 이상 지속 불가능해졌다는 점이다.",
        "",
        "결국 문제는 '도입 여부'가 아니라 '어떤 설계로 도입하느냐'다.",
    ])
    return "\n".join(lines)


def _build_checklist(structure: str, topic: str) -> str:
    """## Practical takeaways – actionable checklist (3–7 items)."""
    lines = ["## 실전 체크리스트"]
    lines.append("")

    if structure == "T":  # TLDR-style: tight, 5 items
        lines.extend([
            f"'{topic}'을 실전에 적용하기 전에 반드시 점검할 항목:",
            "",
            "- [ ] **측정 지표 정의**: 성공/실패를 판단할 지표(비용 절감율, 처리 시간, 에러율)를 먼저 정한다.",
            "- [ ] **최소 루프 설계**: 전체 파이프라인이 아닌, 단일 반복 가능 루프 1개를 먼저 만든다.",
            "- [ ] **실패 시나리오 문서화**: '정상 작동할 때'가 아닌 '장애 시 복구 경로'를 설계한다.",
            "- [ ] **비용 상한 설정**: 토큰/API/컨텍스트 비용의 일일 상한을 설정하고 알림을 건다.",
            "- [ ] **1주 회고 실행**: 도입 7일 후 반드시 지표 기반 회고를 수행한다.",
        ])
    elif structure == "C":  # Checklist-heavy: 7 items
        lines.extend([
            f"'{topic}' 운영 안정화를 위한 체크리스트:",
            "",
            "- [ ] **입력/출력 스키마 고정**: 자동화 대상의 입력·출력 형식을 먼저 고정한다.",
            "- [ ] **소스 3개 이하로 제한**: 수집 소스는 3개 이하에서 시작해 점진적으로 확장한다.",
            "- [ ] **중복 제거 로직 검증**: 정규화 기준(URL, 제목)을 명문화하고 테스트한다.",
            "- [ ] **비용 대비 품질 측정**: 건당 비용과 결과 품질을 매일 기록한다.",
            "- [ ] **장애 대응 플레이북 작성**: 외부 API 장애 시 fallback 경로를 문서화한다.",
            "- [ ] **주간 성과 리뷰**: 조회수, 좋아요, 댓글 기반으로 주간 리뷰를 수행한다.",
            "- [ ] **자동화 범위 재조정**: 2주마다 자동화 범위가 적절한지 재검토한다.",
        ])
    else:  # K = Case-study style: 4 items
        lines.extend([
            f"'{topic}' 케이스에서 배운 실전 교훈:",
            "",
            "- [ ] **작은 루프부터 자동화**: 전체 대신 가장 반복 빈도가 높은 단일 작업부터 자동화한다.",
            "- [ ] **운영 비용을 설계 단계에서 계산**: 배포 후 비용 폭발을 막으려면 아키텍처 설계 시점에 비용 모델을 검증한다.",
            "- [ ] **실패 케이스를 학습 데이터로 전환**: 실패한 실행 로그를 버리지 말고, 다음 반복의 개선 근거로 활용한다.",
            "- [ ] **측정 → 실험 → 반복 주기를 7일 이내로 유지**: 피드백 루프가 길어지면 학습 효과가 급격히 감소한다.",
        ])

    return "\n".join(lines)


def _build_failure_cases(hook: str, lens_id: str, topic: str) -> str:
    """## Failure cases / caveats – honest limitations."""
    lens = LENSES[lens_id]

    lines = [
        "## 실패 사례와 한계",
        "",
    ]

    if hook == "Q":
        lines.extend([
            f"### 흔한 실패 패턴",
            "",
            f"1. **측정 없는 도입**: '{topic}'을 도입했지만 성과 지표를 정의하지 않아 "
            f"3개월 후 '효과 불명'으로 폐기되는 경우가 가장 많다.",
            "",
            f"2. **과도한 자동화 범위**: 처음부터 전체 파이프라인을 자동화하려다 "
            f"복잡도가 폭발해 유지보수 비용이 수작업보다 높아지는 역설.",
            "",
            f"3. **컨텍스트 비용 무시**: 토큰 비용, API 호출 빈도, 컨텍스트 윈도우 제한을 "
            f"고려하지 않고 설계해 운영 비용이 선형 이상으로 증가.",
        ])
    elif hook == "S":
        lines.extend([
            f"### {lens['label']} 관점에서 본 리스크",
            "",
            f"1. **데이터가 보여주지 않는 것**: 채택률 증가가 곧 성공을 의미하지 않는다. "
            f"대부분의 팀이 PoC 단계에서 멈추며, 프로덕션 전환율은 20% 미만이다.",
            "",
            f"2. **벤더 종속 리스크**: 특정 API/모델에 의존하는 설계는 "
            f"가격 정책 변경이나 서비스 중단 시 전체 파이프라인이 마비된다.",
            "",
            f"3. **운영 복잡도 과소평가**: 자동화 시스템 자체의 모니터링, "
            f"장애 대응, 버전 관리에 드는 숨은 비용을 간과하는 경우가 많다.",
        ])
    else:
        lines.extend([
            f"### 왜 롤백하게 되는가",
            "",
            f"1. **설계 없는 확장**: '{topic}'의 초기 성공에 고무되어 범위를 급격히 늘리면 "
            f"아키텍처가 감당하지 못한다. 확장은 반드시 측정 이후에.",
            "",
            f"2. **팀 역량과의 불일치**: 도구의 복잡도가 팀의 운영 역량을 초과하면 "
            f"자동화가 오히려 병목이 된다.",
            "",
            f"3. **피드백 루프 부재**: 결과를 측정하지 않으면 "
            f"개선 방향을 알 수 없고, 동일한 실패를 반복한다.",
        ])

    return "\n".join(lines)


def _build_closing(hook: str, topic: str, domain_angle: str) -> str:
    """## Closing insight – one memorable takeaway."""
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
# Assemble full article body
# ---------------------------------------------------------------------------

def _build_article(
    hook: str,
    structure: str,
    lens_id: str,
    topic: str,
    topic_hint: str,
    evidence: list[str],
) -> str:
    """Assemble the full insight/essay article body."""
    domain_angle = _pick_domain_angle(f"{topic}|{lens_id}|{topic_hint}")

    hook_text = _build_hook(hook, topic, topic_hint, evidence)
    thesis = _build_thesis(hook, lens_id, topic, domain_angle)
    why = _build_why_it_matters(topic, evidence, domain_angle)
    checklist = _build_checklist(structure, topic)
    failures = _build_failure_cases(hook, lens_id, topic)
    closing = _build_closing(hook, topic, domain_angle)

    # Assemble with the prescribed structure
    sections = [
        hook_text,
        "---",
        thesis,
        why,
        checklist,
        failures,
        closing,
    ]

    # Wrap each section through generate_text for future LLM enhancement
    processed = []
    for section in sections:
        if section == "---":
            processed.append(section)
        else:
            processed.append(generate_text(section))

    return "\n\n".join(processed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_drafts(
    summary: TopicSummary,
    count: int = 3,
    topic_hint: str = "",
) -> List[Draft]:
    """Create *count* draft articles with unique (Arm, Lens) combos.

    Parameters
    ----------
    summary : TopicSummary  – the top-ranked summary to base drafts on
    count : int  – number of drafts to generate (caller may pass >3)
    topic_hint : str  – optional topic_hint injected into title and lead

    When random has been seeded (via --seed in run_once.py), the combo
    selection is deterministic.
    """
    all_combos = [(arm, lens) for arm in ARMS for lens in LENS_IDS]
    chosen = random.sample(all_combos, min(count, len(all_combos)))

    # Extract evidence bullets from summary key_points
    evidence = list(getattr(summary, "key_points", []) or [])

    drafts: list[Draft] = []
    for arm, lens_id in chosen:
        hook, structure = arm.split("-")
        title = _build_title(hook, lens_id, summary.topic, topic_hint)
        body = _build_article(hook, structure, lens_id, summary.topic,
                              topic_hint, evidence)

        drafts.append(Draft(
            arm=arm,
            lens=lens_id,
            title=title,
            body=body,
            topic_hint=topic_hint,
        ))

    return drafts
