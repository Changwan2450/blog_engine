import random
import re

OPS_OBJECTS = ["예산 상한", "재시도", "지연", "큐", "로그", "롤백", "SLO"]


BUCKETS = {
    "AI Infra", "Models", "DevTools", "Chips", "Security",
    "Web", "Product", "OpenSource", "Data", "Business",
}


_EN_VERBS = {"using", "develop", "create", "endpoints", "multimodal", "agents", "vlm"}


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text or ""))


def _is_valid_subject(topic: str) -> bool:
    t = (topic or "").strip()
    if not t:
        return False
    if t in BUCKETS:
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
    if not _is_valid_subject(topic):
        topic = "이번 글"
    obj = random.choice(OPS_OBJECTS)
    return f"사람들이 {topic}에서 제일 많이 착각하는 건 {obj}이다."
