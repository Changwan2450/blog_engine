# Plan: DeTemplate + Defuddle Integration

## Scope
- Remove recurring template phrase patterns in generated blog text.
- Strengthen insight-meme voice in writer (LLM prompt + fallback behavior).
- Add Defuddle Markdown extraction stage before summarize with fallback reader.
- Keep CLI/Makefile/pipeline behavior intact.
- Stdlib only; minimal edits.

## Target Files (minimal)
- `src/write.py`
- `src/run_once.py`
- `src/critic.py`
- `src/summarize.py` (only if needed for summarize input hook; prefer `run_once.py` integration)

No directory/CLI/Makefile changes.

---

## Part 1 — Remove template phrases at source (`src/write.py`)

### 1) Source phrase removal
Update these functions to remove direct phrase stems:
- `_narrative_evidence_sentence(...)`
- `_build_why_it_matters_fixed(...)`

Remove hard-coded templates containing:
- `승부는 기능 비교가 아니라`
- `도입 속도보다 ... 먼저다`
- `예를 들어`

### 2) Deterministic de-template guard
Add constant:
- `_ENABLE_DETEMPLATE_GUARD = True`

Add function:
- `_detemplate_blog_text(text: str) -> str`
  - apply banned regex patterns and deterministic replacements (stdlib `re`, `random`)
  - no extra LLM calls

Planned banned regex list:
- `r"승부는\s*기능\s*비교가\s*아니라"`
- `r"도입\s*속도보다"`
- `r"\b먼저다\b"`
- `r"예를\s*들어"`
- `r"요즘"`
- `r"최근\s*몇\s*년"`
- `r"무엇이\s*달라졌고"`
- `r"이\s*주제"`

Planned replacement pool:
- `현장에서 실제로 갈리는 지점은 따로 있다`
- `대부분 팀이 놓치는 지점은 운영 설계다`
- `문제의 핵심은 모델이 아니라 실행 구조다`

### 3) Integration point
Call `_detemplate_blog_text(...)` on the assembled blog section right before final return (just before thread concat or final return path for both LLM/fallback where practical).

---

## Part 2 — Insight meme voice enforcement (`src/write.py`)

### Prompt update
Update writer system prompt constraints to explicitly require:
- opener 1–2 lines, human-rant style
- micro-story with 5 required fields:
  - team size
  - constraint (budget/latency/retry)
  - failure mode
  - fix
  - numeric outcome
- exactly 2 hot-take sentences, one-line each
- each paragraph includes concrete object (cost cap/retry count/timeout/queue depth/p95/canary%/alert rule/cache hit ratio/SLO)
- weak evidence exact phrase:
  - `근거가 약하다. 그래서 이렇게 확인해라:`
  - exactly 2 verification steps

### Fallback alignment
Adjust fallback builders to match prompt style (no corporate/textbook tone, concrete objects in each paragraph) while preserving section structure:
- `## TL;DR`
- `## Why it matters`
- `## Article`
- `## 근거/출처`

---

## Part 3 — Defuddle Markdown extraction stage

### Pipeline target
Current: `collect -> summarize -> write`

Planned internal behavior:
- `collect -> defuddle_extract -> summarize -> write`

### Minimal integration strategy
Prefer adding extraction in `run_once.py` right before summarize (least invasive):
- Build enriched text per item URL via helper:
  - first: `https://defuddle.md/<original_url>`
  - fallback: `https://r.jina.ai/http://<original_url>`
- If defuddle content length `< 500`, treat as fail and fallback.
- On both fail, keep existing item title/url-only behavior.

### Helper functions
Add in `run_once.py` (or `summarize.py` only if cleaner):
- `_fetch_reader_markdown(url: str, timeout: int = 10) -> str`
- `_enrich_items_with_reader_text(items) -> items`

Data handling:
- store extracted text in in-memory field or temporary attribute only.
- no new tracked runtime files.

---

## Part 4 — Critic checks (`src/critic.py`)

Prompt update only:
- explicitly check banned phrase presence
- explicitly check micro-story completeness (5 fields)
- explicitly list generic reusable lines
- keep strict JSON schema unchanged.

No output schema changes.

---

## Part 5 — Verification commands

Run:
1. `python3 -m py_compile src/*.py`
2. `make run`

Validate final output:
3. `rg -n "승부는 기능 비교가 아니라|예를 들어|도입 속도보다|먼저다" out/*_final.md`
4. `rg -n "요즘|최근 몇 년|무엇이 달라졌고|이 주제" out/*_final.md`
5. `rg -n "^## TL;DR|^## Why it matters|^## Article|^## 근거/출처" out/*_final.md`

Runtime data hygiene (if changed by run):
6. `git restore data/research_patterns.json data/topic_categories.json`

---

## Part 6 — Commit plan

Commit message:
- `feat: detemplate writer + add defuddle reader`

Then push:
- `git push`

---

## Rollback plan
- Single switch: set `_ENABLE_DETEMPLATE_GUARD = False` to disable deterministic phrase rewriting while keeping other logic.
- Defuddle extraction failures already fallback to existing summarize inputs, so pipeline remains operational without extra rollback.
