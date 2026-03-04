# Plan: SaaS-grade Quality Upgrade

## Scope
- Upgrade quality and reliability across `collect -> summarize -> write -> select -> run_once` with minimal architectural change.
- Keep stdlib-only and preserve existing CLI/Make targets.
- Maintain backward compatibility for existing summary consumers.

## File-by-file changes

### 1) `src/collect.py`
#### Changes
- Strengthen canonical URL normalization used **only** for dedupe key:
  - lower-case scheme/host
  - remove fragment
  - normalize path (strip trailing slash, preserve meaningful path)
  - parse query params, remove tracking keys and `utm_*`
  - retain only allowlist params: `id`, `p`, `page`, `v`, `t`
- Keep original `SourceItem.url` unchanged in outputs.
- Optional canonical extraction helper for HTML pages (top-N=50):
  - fetch with short timeout (2s)
  - parse `<link rel="canonical" href="...">`
  - cache per-run canonical lookup
  - fallback to normalized URL when unavailable/failure

#### Helper functions to add
- `_is_tracking_param(name: str) -> bool`
- `_normalise_query_for_key(query: str) -> str`
- `_canonical_url_key(raw: str) -> str` (replace current `_canonical_url` usage for dedupe)
- `_maybe_extract_html_canonical(url: str) -> str | None`
- `_dedupe_key_for_item(item: SourceItem, ...) -> str`

#### Rollback toggle points
- `_ENABLE_HTML_CANONICAL = True` toggle
- `_HTML_CANONICAL_MAX = 50` and `_HTML_CANONICAL_TIMEOUT = 2`

---

### 2) `src/summarize.py`
#### Schema updates
- Extend `TopicSummary` with:
  - `topic_en: str`
  - `topic_kr: str`
  - `topic_angle: str`
  - `key_claims: list[dict]` where each dict is:
    - `claim: str`
    - `evidence: list[str]`
    - `confidence: "high" | "med" | "low"`
  - `source_domains: list[str]`
  - `topic_primary: str`
  - `topic_secondary: str`
- Backward compatibility:
  - keep existing `topic` field and set `topic = topic_en`
  - keep `key_points` populated from flattened claims/evidence for existing writers during transition

#### Key-claim generation
- Build grounded claim candidates from title/URL-only signals:
  - quoted phrase extraction
  - numeric context windows
  - named entity + action verb phrases
- Guarantee at least one claim via fallback claim from cleaned title.

#### Confidence heuristic
- `high`: (`quote` or `number`) + named entity
- `med`: named entity or strong action verb
- `low`: fallback/general claim

#### Topic classification
- `topic_primary` from keyword buckets:
  - `AI Infra`, `Models`, `DevTools`, `Chips`, `Security`, `Web`, `Product`, `OpenSource`, `Data`, `Business`
- `topic_secondary` from strongest entity token (e.g., OpenAI/NVIDIA/Apple/AWS/Kubernetes/Rust)
- fallback: `topic_primary="DevTools"`, `topic_secondary=""`

#### Metadata leakage prevention
- Global banned fragments filter on claims/evidence/key_points:
  - `관련 기술/기업:`
  - `출처:`
  - `URL 컨텍스트:`

#### Helper functions to add
- `_extract_source_domains(urls: list[str]) -> list[str]`
- `_extract_claim_candidates(item: SourceItem) -> list[dict]`
- `_claim_confidence(claim: str, evidence: list[str]) -> str`
- `_classify_topic(topic_en: str, item: SourceItem) -> tuple[str, str]`
- `_sanitize_text_fragments(values: list[str]) -> list[str]`

---

### 3) `src/write.py`
#### Article quality upgrade
- Consume upgraded summary fields (`key_claims`, `source_domains`, `topic_primary`, `topic_secondary`).
- Add fixed early sections:
  - `## TL;DR` (1 sentence)
  - `## Why it matters` (2-3 sentences)
- Add end section:
  - `## 근거/출처`
  - include 2-4 `domain + URL` bullets
  - include 1-2 evidence bullets from `key_claims`
- Reduce repetitive boilerplate by deriving intro/thesis from top claim + evidence.

#### Topic repetition control
- Keep quote-normalized repetition cap and enforce exact phrase max=2 for both:
  - `topic_en`
  - `topic_kr`

#### X thread enforcement
- Add `_enforce_x_limits(thread_text: str) -> str`:
  - normalize to 7-9 tweets
  - preserve numbering `1/ ...`
  - enforce each tweet <=240 chars
  - links only in last tweet (or last 2), strip/move earlier links
  - truncate overflow with `…`
- Apply to both LLM output and fallback template thread.

#### Prompt upgrade
- Replace writing system prompt with requested style guidance:
  - insightful, slightly provocative, non-expert friendly
  - avoid generic AI-blog phrasing
  - surprising observation -> mechanism -> concrete example -> forward insight

#### Helper functions to add
- `_build_tldr(summary: TopicSummary) -> str`
- `_build_why_it_matters(summary: TopicSummary) -> str`
- `_build_evidence_sources_section(summary: TopicSummary) -> str`
- `_split_thread_tweets(text: str) -> list[str]`
- `_truncate_tweet(tweet: str, max_len: int = 240) -> str`
- `_enforce_x_limits(thread_text: str) -> str`

---

### 4) `src/select.py`
#### Scoring model upgrade
- Extend score composition:
  - `EV_bandit + source_quality + novelty - duplication_penalty + slot_bonus`
- Add component calculators:
  - `source_quality`: domain weights (`high/med/low`)
  - `novelty`: Jaccard-vs-recent-topic-memory bonus
  - `duplication_penalty`: overlap with recent claim tokens
  - `slot_bonus`: underrepresented `topic_primary` in recent runs
- Print debug component line to `stderr`:
  - `baseEV=?, src=?, nov=?, dup=?, slot=?`

#### Helper functions to add
- `_score_source_quality(draft, summary) -> float`
- `_score_novelty(summary, topic_memory) -> float`
- `_score_duplication(summary, topic_memory) -> float`
- `_score_slot_bonus(summary, runs_recent) -> float`

---

### 5) `src/run_once.py`
#### Integration updates
- Pass richer context into selection call:
  - summary-level `source_domains`, `topic_primary`, `key_claims`
  - recent topic memory entries
  - recent run distribution for slot balancing
- Keep pipeline behavior unchanged externally (same CLI/make).

#### Backward compatibility strategy
- If upgraded fields are missing, selection falls back to existing EV-only behavior.
- Keep existing `choose_best_draft(...)` interface compatible via optional kwargs/defaults.

## Backward compatibility summary
- Preserve `TopicSummary.topic` while introducing `topic_en`.
- Preserve `key_points` as flattened, sanitized derivative of `key_claims`.
- Keep draft/render pipeline inputs stable to avoid downstream breakage.

## Verification commands (post-implementation)
1. `python3 -m py_compile src/*.py`
2. `make run`
3. `make backtest`
4. `grep -R "출처:\|URL 컨텍스트\|관련 기술/기업" out/*_candidates/*.md`
   - must return no matches
5. `for f in out/*_candidates/*.md; do head -1 "$f"; done`
   - verify Korean/natural H1 titles
6. `git status --short`
   - if runtime tracked files changed:
   - `git restore data/research_patterns.json data/topic_categories.json`

## Runtime-state / gitignore check
- No new runtime state file is planned.
- If any new runtime file is introduced under `data/`, append to `.gitignore` before finalizing.

## Rollback strategy
- Toggle off optional HTML canonical extraction (`_ENABLE_HTML_CANONICAL=False`) if fetch overhead/noise appears.
- In `summarize`, fallback to old `key_points` only if new claim extraction fails.
- In `select`, guard each new component; if context missing, component returns `0.0` and existing EV path remains active.
- If needed, revert task-by-task in commit order (collect -> summarize -> write -> select/run_once).
