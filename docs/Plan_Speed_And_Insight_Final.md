# Plan: Speed + Insight Final Stabilization

## Objective
- Cut end-to-end runtime by aggressively limiting downstream workload.
- Keep article quality high with stronger voice constraints and safer fallback behavior.
- Preserve existing CLI / Makefile / pipeline structure.

## Where time is currently spent
1) **High item volume after collect**
- `collect` yields ~3k unique items.
- Downstream summarize/reader/write quality steps still see too many candidates.

2) **Reader extraction latency (defuddle/jina)**
- Network fetch per URL can stall due to timeout/retry.
- Serial fetch path amplifies wall-clock delay.

3) **LLM call path + retries**
- Long prompts and occasional under-produced outputs trigger fallback loops.
- If LLM output is short, immediate fallback reduces quality and wastes effort.

4) **Critic/rewrite chain cost**
- Extra stage is useful but amplifies latency when upstream volume is high.

---

## Exact limits to introduce

### A) Summary candidate hard cap
- Add env-driven cap before summarize input:
  - `BLOG_MAX_ITEMS_SUMMARY` (default `120`)
- Select best 120 by simple deterministic ranking:
  - source-quality proxy (domain weighting if available, else source score)
  - recency (`published_at`) descending

### B) Reader extraction hard cap
- Add env-driven cap for defuddle/jina fetch list:
  - `BLOG_MAX_ITEMS_READER` (default `40`)
- Only top 40 get reader markdown; others remain title+url-only.

### C) Reader concurrency
- Reader fetch only via `ThreadPoolExecutor`:
  - `BLOG_READER_MAX_WORKERS` (default `12`)

---

## Fallback logic changes (fast-fail)

### Reader fetch (defuddle/jina)
- Order:
  1. `https://defuddle.md/<original_url>`
  2. fallback `https://r.jina.ai/http://<host+path>`
- Timeouts:
  - connect/request timeout env-driven default `5`
  - read timeout env-driven default `10` (single timeout if urllib-only)
- Retries:
  - max `1` retry for reader fetch path
- Content threshold:
  - if extracted text `< 500 chars` treat as fail and fallback/skip.

### Write short output handling
- If LLM output under target quality/length:
  - perform **one retry** with explicit expansion instruction
  - then fallback template only if still short

---

## Caching strategy (minimal, optional but included)
- Cache dir: `data/reader_cache/`
- Key: `sha1(url).md`
- TTL: 24h
- On valid cache hit, skip network fetch.
- Add ignore rule: `data/reader_cache/` to `.gitignore`.

---

## Concurrency plan
- Concurrency only for reader extraction (safe isolated network I/O).
- Use thread pool with bounded workers.
- Exceptions are swallowed per-item and never fail whole pipeline.

---

## Insight voice and de-template enforcement plan

### Writer (`src/write.py`)
- Tighten system prompt:
  - exact sections required: TL;DR / Why it matters / Article / 근거/출처 / X Thread
  - Article >= 900 chars target
  - no empty sections
  - one retry before fallback when output is too short
- Keep de-template guard active and extend to fallback output path.
- Ensure banned phrase list remains blocked in final blog text.

### Critic (`src/critic.py`)
- Prompt checks updated to enforce:
  - banned phrase absence
  - micro-story completeness (team size + constraint + failure + fix + numeric outcome)
  - evidence junk token detection
  - exactly 2 hot-takes

### Evidence cleanup (`src/summarize.py` / `src/write.py`)
- Drop evidence snippets if:
  - len < 12
  - single English token-like (`^[A-Za-z]{1,12}$`)
  - stop-word-only tokens (There/New/In/Town...)
- If no valid evidence remains:
  - keep links in 근거/출처
  - enforce exact weak-evidence phrase + exactly 2 verification steps.

---

## Verification commands
1. `python3 -m py_compile src/*.py`
2. `make run`
3. confirm speed logs in run output:
   - `[speed] summary_items=...`
   - `[speed] reader_items=...`
4. banned phrases:
   - `rg -n "승부는 기능 비교가 아니라|예를 들어|도입 속도보다|먼저다|요즘|최근 몇 년|무엇이 달라졌고|이 주제" out/*_final.md`
5. evidence snippet quality:
   - `rg -n "근거 스니펫:" out/*_final.md`
6. git hygiene:
   - `git status --short`
   - if drift:
     - `git restore data/research_patterns.json data/topic_categories.json`

---

## Rollback toggles
- `BLOG_MAX_ITEMS_SUMMARY` / `BLOG_MAX_ITEMS_READER` can be raised quickly without code changes.
- Reader cache can be bypassed by setting TTL to 0 (constant/env toggle).
- Keep de-template guard behind existing toggle so it can be disabled quickly if needed.
