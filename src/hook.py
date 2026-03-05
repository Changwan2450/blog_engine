import random
import re

OPS_OBJECTS = ["예산 상한", "재시도", "지연", "큐", "로그", "롤백", "SLO"]


BUCKETS = {
    "AI Infra", "Models", "DevTools", "Chips", "Security",
    "Web", "Product", "OpenSource", "Data", "Business",
}


_EN_VERBS = {"using", "develop", "create", "endpoints", "multimodal", "agents", "vlm"}
_VAGUE_SUBJECTS = {"이 글", "이번 글", "이 스택", "공개 장애 사례", "이 케이스", "이 이슈", "해당 기술"}


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text or ""))


def _is_valid_subject(topic: str) -> bool:
    t = (topic or "").strip()
    if not t:
        return False
    if t in BUCKETS:
        return False
    if t in _VAGUE_SUBJECTS:
        return False
    if _english_word_count(t) > 0:
        return False
    tl = t.lower()
    if any(v in tl for v in _EN_VERBS):
        return False
    if "|" in t or "…" in t or t.endswith("..."):
        return False
    return bool(re.search(r"[가-힣]", t))


def generate_hook(subject: str) -> str:
    topic = (subject or "").strip()
    if (not topic) or (topic in _VAGUE_SUBJECTS) or (not _is_valid_subject(topic)):
        topic = "AI 에이전트 운영"
    obj = random.choice(["retry", "timeout", "budget cap", "p95", "queue", "로그"])
    return f"요즘 {topic} 얘기 많은데, 진짜 터지는 건 {obj}다."
