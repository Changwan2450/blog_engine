# Plan: Model Routing + Critic Rewrite + Dotenv Loading

## 1) Current OPENAI env usage map (today)

- `src/summarize.py:208`
  - `api_key = os.environ.get("OPENAI_API_KEY", "")`
- `src/summarize.py:223`
  - model set inline: `os.environ.get("OPENAI_MODEL", "gpt-4o-mini")`

- `src/write.py:204`
  - module-level model constant: `_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")`
- `src/write.py:210`
  - `api_key = os.environ.get("OPENAI_API_KEY", "")` in `_call_llm`
- `src/write.py:242`
  - `api_key = os.environ.get("OPENAI_API_KEY", "")` in `generate_text`
- `src/write.py:861`
  - API key checked before long-form article generation call

Note: `run_once.py` currently imports summarize/write early but does not load `.env` explicitly.

---

## 2) .env loading design (stdlib-only, minimal)

### File to add
- `src/env_loader.py` (new)

### Function
- `load_dotenv(path: str | Path = ".env") -> None`

### Behavior
- Read local `.env` as plain text (UTF-8).
- Ignore:
  - empty lines
  - lines starting with `#`
- Parse `KEY=VALUE` only.
- Trim outer whitespace around key/value.
- Strip optional matching quotes (`"..."` or `'...'`) around value.
- Do **not** override pre-existing process env vars (`if key in os.environ: continue`).
- Fail-safe on malformed lines (skip, no crash).

### Integration point
- Call `load_dotenv(project_root / ".env")` at the earliest safe point inside `run_once.main()` before calling `run_once(...)`.
- Keep existing CLI/Make behavior unchanged.

Reasoning: dotenv load must happen before summarize/write LLM calls, while avoiding import-order complexity around existing stdlib `select` workaround.

---

## 3) Env var spec and model routing

### Env vars
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (global fallback)
- `OPENAI_MODEL_SUMMARIZE`
- `OPENAI_MODEL_WRITE`
- `OPENAI_MODEL_CRITIC`

### Helper to add
- `get_model(stage: str) -> str`
  - location: `src/env_loader.py` (or small dedicated `src/model_routing.py` if cleaner)
  - mapping:
    - `summarize` -> `OPENAI_MODEL_SUMMARIZE` or `OPENAI_MODEL` or `gpt-4o-mini`
    - `write` -> `OPENAI_MODEL_WRITE` or `OPENAI_MODEL` or `gpt-4o-mini`
    - `critic` -> `OPENAI_MODEL_CRITIC` or `OPENAI_MODEL` or `gpt-4o-mini`

### File updates
- `src/summarize.py`
  - Replace inline `OPENAI_MODEL` read with `get_model("summarize")`.
- `src/write.py`
  - Remove static `_OPENAI_MODEL` dependency for routing.
  - `_call_llm(..., stage="write")` picks model via `get_model(stage)`.

Backward compatibility:
- If only `OPENAI_MODEL` is set, behavior remains equivalent.
- If no model env vars set, default remains `gpt-4o-mini`.

---

## 4) Critic + Rewrite step design (minimal integration)

### New file
- `src/critic.py`

### API
- `critique_draft(draft_text: str, topic_kr: str, lens_label: str) -> dict`
  - returns keys:
    - `punchline` (str)
    - `whats_boring` (list[str], 1-3)
    - `whats_strong` (list[str], 1-3)
    - `rewrite_instructions` (list[str], 3-7)
    - `risk_flags` (list[str], 0-3)

### Implementation details
- Use OpenAI chat completion via urllib (same style as existing modules).
- Use critic-stage model routing: `get_model("critic")`.
- Keep output strict JSON parse with robust fallback on error:
  - fallback critique with empty/minimal lists and neutral instructions.
- Low token budget for critic request.

### run_once integration
- In `src/run_once.py`, after `generate_drafts(...)` and before quality/select:
  1. critique each draft body (or body+title)
  2. attach critique to draft (via dynamic attrs, e.g. `draft.critique = ...`)
  3. pick top-1 draft candidate for rewrite (minimal cost path)
     - use current bandit score pre-pass or first passing draft as practical heuristic
  4. rewrite that one draft using writer LLM with critique instructions
- Keep existing quality gate + select + render intact.
- If critic/rewrite fails, continue with original drafts (no pipeline break).

---

## 5) Insight voice prompt refresh in writer

### `src/write.py` prompt updates
- Replace rigid section forcing language with voice constraints:
  - start with surprising concrete observation (1-2 lines)
  - explain hidden mechanism
  - include one vivid scenario (5-7 lines)
  - end with one actionable takeaway
- Keep required output sections:
  - `## TL;DR`
  - `## Why it matters`
  - `## Article`
  - `## 근거/출처`
- Ban phrases in generation instruction:
  - `무엇이 달라졌고`
  - `이 주제`
  - `최근 6개월`
  - `정리`, `소개`, `동향`
- Ban metadata fragments in output instruction:
  - `출처:`
  - `URL 컨텍스트:`
  - `관련 기술/기업:`
- Keep factual, allow light humor/slight provocation.

---

## 6) Files to edit

- New:
  - `src/env_loader.py`
  - `src/critic.py`

- Update:
  - `src/run_once.py`
  - `src/summarize.py`
  - `src/write.py`

- Optional tiny update:
  - `.gitignore` only if needed (currently `.env` is already ignored).

---

## 7) Verification plan

1. Compile:
   - `python3 -m py_compile src/*.py`
2. Run pipeline:
   - `make run`
3. Backtest smoke (optional but recommended):
   - `make backtest`
4. Secret hygiene checks:
   - ensure no key-like values committed
   - confirm `.env` is ignored (`git check-ignore -v .env`)
5. Metadata leakage check in outputs:
   - `grep -R "출처:\|URL 컨텍스트\|관련 기술/기업" out/*_candidates/*.md`
6. Git hygiene:
   - `git status --short`
   - if runtime tracked files changed:
     - `git restore data/research_patterns.json data/topic_categories.json`

---

## 8) Rollback / safety points

- Critic step guarded by try/except; if unavailable, pipeline continues with original drafts.
- Rewrite only top-1 to bound latency/cost and reduce regression surface.
- Keep existing quality gate/select signatures compatible (only additive metadata/kwargs).
- Model routing falls back to `OPENAI_MODEL` then default `gpt-4o-mini`.
