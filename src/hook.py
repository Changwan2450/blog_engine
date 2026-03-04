import random
import re

HOOK_PATTERNS = [
    "{topic}가 망하는 진짜 이유는 모델이 아니라 {object}다.",
    "사람들은 {topic}을 모델 문제로 보지만 실제 병목은 {object}다.",
    "{topic} 프로젝트가 터지는 순간은 항상 {object}에서 시작된다.",
    "대부분의 {topic} 실패는 코드가 아니라 {object} 설정에서 나온다.",
    "{topic}에서 사람들이 가장 많이 착각하는 건 {object}다.",
]

OBJECTS = [
    "retry 규칙",
    "timeout",
    "SLO",
    "budget cap",
    "queue 길이",
    "cache 전략",
    "rollback 규칙",
    "canary 롤아웃",
    "alerts 설정",
    "로그 설계"
]


BUCKETS = {
    "AI Infra", "Models", "DevTools", "Chips", "Security",
    "Web", "Product", "OpenSource", "Data", "Business",
}


def generate_hook(subject: str) -> str:
    topic = (subject or "").strip()
    if not topic:
        topic = "AI 에이전트"
    if topic in BUCKETS:
        topic = "AI 에이전트"

    # Optional safety: English-heavy subject fallback.
    english_words = re.findall(r"\b[A-Za-z]+\b", topic)
    if len(english_words) > 3:
        topic = "AI 에이전트"

    pattern = random.choice(HOOK_PATTERNS)
    obj = random.choice(OBJECTS)
    return pattern.format(topic=topic, object=obj)
