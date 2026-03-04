"""research.py – Extract writing patterns from collected source items.

Analyzes titles, hooks, keywords, section patterns, and topic categories
from external content (Reddit, Google News, RSS blogs) to inform writing
prompts and improve content diversity.

Saves results to data/research_patterns.json (accumulates over runs).
Also generates data/topic_categories.json.

This data influences writing prompts but does NOT act as reward.

Stdlib only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_KR_WORD_RE = re.compile(r"[\uac00-\ud7a3]{2,}")
# Detect tech product / project names (capitalized or ALL-CAPS, 2+ chars)
_TECH_NAME_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-\.]+|[A-Z]{2,}[0-9]*)\b")

_STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any",
    "can", "her", "was", "one", "our", "out", "has", "have", "had",
    "with", "this", "that", "from", "they", "been", "said", "each",
    "will", "how", "its", "may", "use", "would", "make", "like",
    "just", "about", "over", "such", "more", "than", "what", "when",
    "who", "why", "new", "now", "also", "into", "some", "get",
    "your", "most", "first", "don", "does", "using", "used",
    "every", "here", "best", "need", "top", "way", "things",
    "really", "still", "much", "look", "part", "last",
}

# Technology/product stop words (too generic to be useful)
_TECH_STOPS = {"API", "SDK", "CPU", "GPU", "RAM", "SSD", "URL", "HTTP", "JSON",
               "HTML", "CSS", "SQL", "PDF", "RSS", "CLI"}


# ---------------------------------------------------------------------------
# Hook template library (seeded + extracted from content)
# ---------------------------------------------------------------------------

_BUILTIN_HOOK_TEMPLATES = [
    # Failure framing
    "The real reason {topic} fails",
    "{topic} works in demos but fails in production",
    "The biggest mistake teams make with {topic}",
    "Most {topic} projects fail — and not because of the tech",
    "{topic} looked great in the PoC, then reality hit",

    # Contrarian / challenge
    "Why most teams misunderstand {topic}",
    "What most people get wrong about {topic}",
    "Everyone talks about {topic}, but nobody measures the cost",
    "Stop treating {topic} as a silver bullet",
    "{topic} is powerful, but it hides a major cost",

    # Hidden cost / risk
    "The hidden cost of {topic}",
    "{topic} is cheaper than you think — until it isn't",
    "The operational tax of {topic} nobody warns you about",
    "You adopted {topic}. Now the real work begins",
    "The true price of scaling {topic}",

    # Data driven
    "We ran {topic} for 90 days. Here's what the numbers say",
    "{topic} adoption is up 300%. Success rate? Still 20%",
    "The data behind {topic}: what it shows and what it hides",

    # Measurement / metrics
    "Everyone talks about {topic}, but nobody measures {metric}",
    "If you can't measure it, {topic} is just a buzzword",
    "The one metric that separates {topic} winners from losers",

    # Practical / tactical
    "Before you adopt {topic}, ask these 5 questions",
    "The 7-day {topic} test: measure before you commit",
    "A checklist for surviving {topic} in production",

    # Question hooks
    "Is {topic} solving a real problem — or creating a new one?",
    "Why does {topic} keep failing in teams that should know better?",
    "What happens when {topic} meets real-world constraints?",

    # Reversal / surprise
    "We replaced {topic} with a spreadsheet. It worked better",
    "The simplest alternative to {topic} nobody is talking about",
    "Most developers think {topic} solves the problem. It doesn't",
]


# ---------------------------------------------------------------------------
# Title pattern extraction (expanded)
# ---------------------------------------------------------------------------

def _extract_title_patterns(titles: list[str], top_n: int = 30) -> list[str]:
    """Extract frequent title structures from collected titles."""
    patterns: list[str] = []

    for title in titles:
        tl = title.lower().strip()

        if any(tl.startswith(q) for q in ["why ", "how ", "what ", "when ", "왜 ", "어떻게 "]):
            patterns.append("question_hook")
        elif re.match(r"^\d+\s", tl):
            patterns.append("numbered_list")
        elif " vs " in tl or " vs. " in tl:
            patterns.append("comparison")
        elif ":" in title and title.index(":") < len(title) // 2:
            patterns.append("topic_colon_subtitle")
        elif " — " in title or " - " in title:
            patterns.append("topic_dash_angle")
        elif re.match(r"^(the|a)\s+(real|hidden|true|biggest|one)\s", tl):
            patterns.append("superlative_claim")
        elif any(w in tl for w in ["mistake", "wrong", "fail", "실패"]):
            patterns.append("failure_framing")
        elif any(w in tl for w in ["guide", "tutorial", "가이드"]):
            patterns.append("practical_guide")
        elif tl.endswith("?"):
            patterns.append("trailing_question")
        elif any(w in tl for w in ["surprising", "unexpected", "놀라운"]):
            patterns.append("surprise_reveal")
        elif any(w in tl for w in ["stop", "don't", "avoid", "하지 마"]):
            patterns.append("contrarian_warning")
        elif any(w in tl for w in ["cost", "price", "비용", "expensive"]):
            patterns.append("cost_analysis")
        elif any(w in tl for w in ["every ", "all ", "most ", "nobody"]):
            patterns.append("sweeping_statement")
        elif any(w in tl for w in ["better", "faster", "simpler"]):
            patterns.append("improvement_claim")
        else:
            patterns.append("declarative")

    counter = Counter(patterns)
    return [p for p, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Hook pattern extraction (expanded, template-based)
# ---------------------------------------------------------------------------

def _extract_hook_patterns(titles: list[str], top_n: int = 30) -> list[str]:
    """Classify titles into hook pattern categories and return unique patterns."""
    pattern_counter: Counter = Counter()

    _hook_rules = [
        (["?"], "provocative_question"),
        (["실패", "fail", "wrong", "mistake", "problem", "break"], "failure_framing"),
        (["hidden", "숨겨진", "nobody", "아무도"], "hidden_insight"),
        (["stop", "멈춰", "하지 마", "don't", "avoid", "never"], "contrarian"),
        (["guide", "가이드", "checklist", "체크리스트", "step"], "practical_guide"),
        (["cost", "비용", "price", "expensive", "cheap"], "cost_hook"),
        (["vs", "versus", "compared", "비교", "alternative"], "comparison_hook"),
        (["real reason", "진짜 이유", "actual", "truth"], "truth_reveal"),
        (["biggest", "most", "worst", "best", "top"], "superlative"),
        (["everyone", "nobody", "모두", "아무도"], "sweeping_claim"),
        (["works in", "demo", "poc", "prototype"], "demo_vs_reality"),
        (["replace", "대체", "instead", "better than"], "replacement_narrative"),
        (["measure", "측정", "metric", "지표", "number"], "measurement_focus"),
        (["simple", "간단", "easy", "minimal", "lightweight"], "simplicity_angle"),
        (["scale", "확장", "grow", "production", "프로덕션"], "scale_challenge"),
        (["we ", "our ", "우리", "I ", "my "], "first_person_narrative"),
        (["lesson", "교훈", "learned", "배운"], "lessons_learned"),
        (["before you", "하기 전에", "first"], "prerequisite_warning"),
    ]

    for title in titles:
        tl = title.lower()
        matched = False
        for keywords, pattern_name in _hook_rules:
            if any(k in tl for k in keywords):
                pattern_counter[pattern_name] += 1
                matched = True
                break
        if not matched:
            # Check for data patterns
            if re.search(r"\d+[%x×]|\d+k|\d+m|\d+\s*(day|week|month)", tl):
                pattern_counter["data_driven"] += 1
            else:
                pattern_counter["insight_statement"] += 1

    return [p for p, _ in pattern_counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Keyword extraction (expanded: tech names + bigrams)
# ---------------------------------------------------------------------------

def _extract_keywords(titles: list[str], top_n: int = 200) -> list[str]:
    """Extract technology names, product names, concepts from titles."""
    word_counter: Counter = Counter()

    for title in titles:
        # Tech/product names (capitalized)
        for m in _TECH_NAME_RE.finditer(title):
            name = m.group(0)
            if name not in _TECH_STOPS and len(name) >= 2:
                word_counter[name] += 1

        # Regular English words
        for m in _WORD_RE.finditer(title):
            w = m.group(0).lower()
            if w not in _STOP_WORDS and len(w) >= 3:
                word_counter[w] += 1

        # Korean words (2+ syllables)
        for m in _KR_WORD_RE.finditer(title):
            word_counter[m.group(0)] += 1

        # Bigrams (English, lowered)
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", title)
                 if len(w) >= 3 and w.lower() not in _STOP_WORDS]
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) >= 7:
                word_counter[bigram] += 1

    # Filter: require at least 2 occurrences
    return [w for w, c in word_counter.most_common(top_n * 2) if c >= 2][:top_n]


# ---------------------------------------------------------------------------
# Section pattern extraction (expanded)
# ---------------------------------------------------------------------------

def _extract_section_patterns(titles: list[str]) -> list[str]:
    theme_counter: Counter = Counter()

    _theme_rules = [
        (["cost", "비용", "price", "pricing", "budget"], "cost_analysis"),
        (["fail", "실패", "risk", "리스크", "problem", "issue"], "failure_cases"),
        (["how", "guide", "tutorial", "가이드", "방법", "step"], "practical_how_to"),
        (["vs", "compare", "비교", "alternative", "instead"], "comparison"),
        (["benchmark", "performance", "성능", "측정", "metric"], "measurement"),
        (["architecture", "아키텍처", "design", "설계", "system"], "system_design"),
        (["checklist", "체크리스트", "list", "tip"], "checklist"),
        (["scale", "확장", "production", "deploy", "배포"], "production_readiness"),
        (["security", "보안", "privacy", "취약"], "security"),
        (["team", "팀", "hire", "culture", "조직"], "team_org"),
        (["open source", "오픈소스", "github", "oss"], "open_source"),
        (["automat", "자동", "pipeline", "파이프라인", "workflow"], "automation"),
    ]

    for title in titles:
        tl = title.lower()
        for keywords, theme in _theme_rules:
            if any(k in tl for k in keywords):
                theme_counter[theme] += 1

    return [p for p, _ in theme_counter.most_common(15)]


# ---------------------------------------------------------------------------
# Topic category discovery
# ---------------------------------------------------------------------------

_CATEGORY_RULES: dict[str, list[str]] = {
    "agents": ["agent", "multi-agent", "agentic", "autogpt", "crewai",
               "autonomous", "에이전트"],
    "tools": ["tool", "framework", "library", "sdk", "mcp",
              "langchain", "llamaindex", "도구"],
    "productivity": ["workflow", "automation", "pipeline", "효율",
                     "자동화", "생산성", "productivity"],
    "models": ["llm", "gpt", "claude", "gemini", "mistral", "llama",
               "model", "모델", "fine-tune", "training"],
    "infrastructure": ["deploy", "kubernetes", "docker", "cloud",
                       "server", "인프라", "배포", "hosting"],
    "systems": ["architecture", "orchestration", "system", "설계",
                "아키텍처", "microservice", "distributed"],
    "cost": ["cost", "pricing", "token", "비용", "budget", "billing"],
    "failures": ["fail", "mistake", "wrong", "risk", "실패",
                 "rollback", "downtime", "incident"],
    "measurement": ["metric", "benchmark", "performance", "측정",
                    "evaluation", "성능", "monitoring"],
    "local_ai": ["local", "on-device", "edge", "mlx", "gguf",
                 "quantize", "로컬", "온디바이스"],
}


def _discover_topic_categories(titles: list[str]) -> dict[str, list[str]]:
    """Infer topic categories from titles and return category→keyword mapping."""
    category_keywords: dict[str, Counter] = {
        cat: Counter() for cat in _CATEGORY_RULES
    }

    for title in titles:
        tl = title.lower()
        for category, trigger_words in _CATEGORY_RULES.items():
            for tw in trigger_words:
                if tw in tl:
                    # Extract nearby keywords as category members
                    for m in _WORD_RE.finditer(title):
                        w = m.group(0).lower()
                        if w not in _STOP_WORDS and len(w) >= 3:
                            category_keywords[category][w] += 1

    # Build final mapping: top keywords per category
    result: dict[str, list[str]] = {}
    for cat, counter in category_keywords.items():
        top = [w for w, c in counter.most_common(20) if c >= 2]
        if top:
            result[cat] = top

    return result


# ---------------------------------------------------------------------------
# Pattern accumulation (merge new with existing)
# ---------------------------------------------------------------------------

def _merge_lists(existing: list, new: list, max_items: int) -> list:
    """Merge two lists, dedup, cap at max_items."""
    seen: set = set()
    merged: list = []
    for item in existing + new:
        key = item if isinstance(item, str) else str(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged[:max_items]


def _load_existing(path: Path) -> dict:
    """Load existing patterns file if it exists."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_patterns(items: list, output_path: str | Path) -> dict:
    """Extract writing patterns from collected source items.

    Accumulates patterns over runs (merges with existing file).

    Parameters
    ----------
    items : list of SourceItem (or any object with .title attribute)
    output_path : path to write research_patterns.json

    Returns
    -------
    dict with keys: title_patterns, hook_patterns, keywords, section_patterns,
                    hook_templates
    """
    titles = [getattr(it, "title", "") for it in items if getattr(it, "title", "")]

    if not titles:
        print("[research] WARN  No titles to analyze", file=sys.stderr)
        return _load_existing(Path(output_path)) or {
            "title_patterns": [], "hook_patterns": [],
            "keywords": [], "section_patterns": [], "hook_templates": [],
        }

    # Extract new patterns
    new_title_patterns = _extract_title_patterns(titles)
    new_hook_patterns = _extract_hook_patterns(titles)
    new_keywords = _extract_keywords(titles)
    new_section_patterns = _extract_section_patterns(titles)

    # Load existing and merge (accumulate over time)
    path = Path(output_path)
    existing = _load_existing(path)

    patterns = {
        "title_patterns": _merge_lists(
            existing.get("title_patterns", []), new_title_patterns, 50),
        "hook_patterns": _merge_lists(
            existing.get("hook_patterns", []), new_hook_patterns, 50),
        "keywords": _merge_lists(
            existing.get("keywords", []), new_keywords, 300),
        "section_patterns": _merge_lists(
            existing.get("section_patterns", []), new_section_patterns, 20),
        "hook_templates": _merge_lists(
            existing.get("hook_templates", []), list(_BUILTIN_HOOK_TEMPLATES), 60),
    }

    # Save
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(patterns, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Topic categories (separate file)
    cat_path = path.parent / "topic_categories.json"
    categories = _discover_topic_categories(titles)
    existing_cats = _load_existing(cat_path)
    for cat, kws in categories.items():
        existing_kws = existing_cats.get(cat, [])
        categories[cat] = _merge_lists(existing_kws, kws, 30)
    merged_cats = {**existing_cats, **categories}
    cat_path.write_text(
        json.dumps(merged_cats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[research] Extracted {len(patterns['keywords'])} keywords, "
          f"{len(patterns['title_patterns'])} title patterns, "
          f"{len(patterns['hook_patterns'])} hook patterns, "
          f"{len(patterns['hook_templates'])} hook templates, "
          f"{len(merged_cats)} topic categories",
          file=sys.stderr)

    return patterns


def load_patterns(path: str | Path) -> dict:
    """Load previously saved research patterns."""
    return _load_existing(Path(path))
