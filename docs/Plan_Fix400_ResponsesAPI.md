# Plan: Fix OpenAI 400 with Responses API client

## Current request locations
- `src/write.py`
  - `_call_llm(...)` posts to `https://api.openai.com/v1/chat/completions`
  - used by `generate_text(...)`, `rewrite_with_feedback(...)`, and article build path.
- `src/critic.py`
  - `_call_critic_llm(...)` posts to `https://api.openai.com/v1/chat/completions`.
- `src/summarize.py`
  - `rewrite_topic(...)` posts to `https://api.openai.com/v1/chat/completions`.

## Why HTTP 400 happens
- GPT-5-class models can require the Responses API (`/v1/responses`) or payload shape differences.
- Current code hardcodes Chat Completions payload/endpoint, causing model-vs-endpoint mismatch and 400 fallback behavior.

## Exact endpoint/payload changes

### New shared wrapper
- Add `src/openai_client.py` with:
  - `call_openai(stage, system, user, max_tokens, temperature) -> str`
  - Uses `get_model(stage)` from `env_loader`.
  - Env switch `OPENAI_USE_RESPONSES` (default on if unset/"1").

### Responses API path (preferred)
- Endpoint: `POST https://api.openai.com/v1/responses`
- Payload:
  - `model`
  - `input`: system/user content arrays
  - include `max_output_tokens` and `temperature`
- Robust text extraction from common shapes:
  - `output_text`
  - `output[].content[].text`
  - fallback scan for text-like fields

### Chat Completions fallback path
- Endpoint: `POST https://api.openai.com/v1/chat/completions`
- Payload:
  - `model`, `messages`, `temperature`, `max_tokens`
- Parse `choices[0].message.content`.

### Error visibility
- On `HTTPError`, read response body and print to stderr:
  - include stage + endpoint + status + body text
  - do not print API key.

## Integration changes
- `src/write.py`
  - remove direct endpoint calls from `_call_llm`
  - route all LLM calls through `openai_client.call_openai(stage="write", ...)`
- `src/critic.py`
  - route through `call_openai(stage="critic", ...)`
- `src/summarize.py`
  - route `rewrite_topic(...)` through `call_openai(stage="summarize", ...)`

## Env compatibility
- Keep existing env routing untouched:
  - `OPENAI_MODEL_SUMMARIZE`
  - `OPENAI_MODEL_WRITE`
  - `OPENAI_MODEL_CRITIC`
  - `OPENAI_MODEL`
- Add optional:
  - `OPENAI_USE_RESPONSES` (default `1`).

## Rollback plan
- Set `OPENAI_USE_RESPONSES=0` to force legacy Chat Completions without code rollback.
- If issues persist, module-level change can be reverted by switching caller imports back to local `_call_llm` implementation.

## Verification commands
1. `python3 -m py_compile src/*.py`
2. `make run`
3. Confirm no writer 400 line:
   - absence of `[write] WARN  LLM call failed (HTTP Error 400...)`
4. If error occurs, confirm stderr includes HTTP response body JSON (without secrets).
