"""Shared OpenAI caller with Responses API preference (stdlib only).

Env vars (network/reliability):
- OPENAI_USE_RESPONSES=1           # default on
- OPENAI_FORCE_CHAT=0              # if 1, skip Responses API
- OPENAI_HTTP_TIMEOUT_SEC=30       # connection/request timeout
- OPENAI_READ_TIMEOUT_SEC=60       # read timeout budget (combined with HTTP timeout)
- OPENAI_MAX_RETRIES=3             # retriable attempts after first request
- OPENAI_BACKOFF_BASE_SEC=1.0      # exponential backoff base
- OPENAI_BACKOFF_MAX_SEC=12.0      # max backoff cap
- OPENAI_MAX_OUTPUT_TOKENS_WRITE     (optional cap)
- OPENAI_MAX_OUTPUT_TOKENS_SUMMARIZE (optional cap)
- OPENAI_MAX_OUTPUT_TOKENS_CRITIC    (optional cap)
- OPENAI_TEMPERATURE=0.7           # default temperature when supported by model
"""
from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_loader import get_model


_RESPONSES_URL = "https://api.openai.com/v1/responses"
_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_MAX_RESPONSE_BYTES = 2_000_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _network_settings() -> dict:
    http_timeout = _env_int("OPENAI_HTTP_TIMEOUT_SEC", 30)
    read_timeout = _env_int("OPENAI_READ_TIMEOUT_SEC", 60)
    timeout = max(1, max(http_timeout, read_timeout))
    retries = max(0, _env_int("OPENAI_MAX_RETRIES", 3))
    backoff_base = max(0.1, _env_float("OPENAI_BACKOFF_BASE_SEC", 1.0))
    backoff_max = max(backoff_base, _env_float("OPENAI_BACKOFF_MAX_SEC", 12.0))
    return {
        "timeout": timeout,
        "retries": retries,
        "backoff_base": backoff_base,
        "backoff_max": backoff_max,
    }


def _use_responses_api() -> bool:
    raw = os.environ.get("OPENAI_USE_RESPONSES", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _force_chat() -> bool:
    raw = os.environ.get("OPENAI_FORCE_CHAT", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body[:2000]


def _is_retriable_http(exc: HTTPError) -> bool:
    return int(getattr(exc, "code", 0) or 0) in {429, 500, 502, 503, 504}


def _is_timeout_like(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        if reason and "timed out" in str(reason).lower():
            return True
    return "timed out" in str(exc).lower()


def _is_retriable_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return _is_retriable_http(exc)
    if isinstance(exc, URLError):
        return _is_timeout_like(exc)
    if isinstance(exc, (TimeoutError, socket.timeout, OSError)):
        return _is_timeout_like(exc)
    return False


def _err_label(exc: Exception) -> str:
    if _is_timeout_like(exc):
        return "timeout"
    if isinstance(exc, HTTPError):
        return f"http_{exc.code}"
    return exc.__class__.__name__.lower()


def _stage_max_tokens(stage: str, requested: int) -> int:
    key = f"OPENAI_MAX_OUTPUT_TOKENS_{stage.strip().upper()}"
    cap = _env_int(key, requested)
    if cap <= 0:
        cap = requested
    return max(32, min(int(requested), int(cap)))


def _extract_text_from_responses(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if isinstance(direct, list):
        joined = "\n".join(str(x).strip() for x in direct if str(x).strip())
        if joined.strip():
            return joined.strip()

    out = data.get("output")
    if isinstance(out, list):
        texts: list[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                txt = part.get("text")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())
                elif isinstance(txt, dict):
                    val = txt.get("value")
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
                out_txt = part.get("output_text")
                if isinstance(out_txt, str) and out_txt.strip():
                    texts.append(out_txt.strip())
        if texts:
            return "\n".join(texts).strip()

    # recursive fallback scan for text-like fields
    def _walk(obj) -> list[str]:
        acc: list[str] = []
        if isinstance(obj, str):
            if obj.strip():
                acc.append(obj.strip())
            return acc
        if isinstance(obj, list):
            for it in obj:
                acc.extend(_walk(it))
            return acc
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in {"text", "output_text", "value"}:
                    if isinstance(v, str) and v.strip():
                        acc.append(v.strip())
                    elif isinstance(v, dict):
                        vval = v.get("value")
                        if isinstance(vval, str) and vval.strip():
                            acc.append(vval.strip())
                else:
                    acc.extend(_walk(v))
            return acc
        return acc

    found = _walk(data)
    if found:
        joined = "\n".join(x for x in found if len(x) > 1).strip()
        if joined:
            return joined
    return ""


def _post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read(_MAX_RESPONSE_BYTES).decode("utf-8", errors="ignore")
        return json.loads(raw)


def _post_json_with_retry(
    *,
    stage: str,
    endpoint_label: str,
    url: str,
    payload: dict,
    api_key: str,
    timeout: int,
    retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict:
    attempts_total = retries + 1
    last_exc: Exception | None = None

    for attempt in range(attempts_total):
        try:
            return _post_json(url, payload, api_key=api_key, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            if isinstance(exc, HTTPError):
                body = _read_http_error_body(exc)
                print(
                    f"[openai] WARN  stage={stage} endpoint={endpoint_label} status={exc.code} body={body}",
                    file=sys.stderr,
                )

            retriable = _is_retriable_error(exc)
            if (not retriable) or attempt >= retries:
                raise

            delay = min(backoff_max, backoff_base * (2 ** attempt))
            delay *= 0.7 + random.random() * 0.6
            print(
                "[openai] stage=%s attempt=%d/%d err=%s backoff=%.1fs"
                % (stage, attempt + 2, attempts_total, _err_label(exc), delay),
                file=sys.stderr,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI request failed without exception")


def call_openai(
    stage: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI using Responses API by default; fallback to Chat Completions."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = get_model(stage)
    net = _network_settings()
    timeout = int(net["timeout"])
    retries = int(net["retries"])
    backoff_base = float(net["backoff_base"])
    backoff_max = float(net["backoff_max"])
    supports_temperature = not model.lower().startswith("gpt-5")
    max_tokens_eff = _stage_max_tokens(stage, max_tokens)
    temp_eff = _env_float("OPENAI_TEMPERATURE", temperature)
    use_responses_first = _use_responses_api() and not _force_chat()

    # Preferred path: Responses API
    if use_responses_first:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_tokens_eff,
        }
        if supports_temperature:
            payload["temperature"] = temp_eff
        try:
            data = _post_json_with_retry(
                stage=stage,
                endpoint_label="/v1/responses",
                url=_RESPONSES_URL,
                payload=payload,
                api_key=api_key,
                timeout=timeout,
                retries=retries,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
            )
            text = _extract_text_from_responses(data)
            if text:
                return text
            print(
                f"[openai] WARN  stage={stage} responses returned empty text; falling back to chat",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_retriable_error(exc):
                print(f"[openai] WARN  stage={stage} responses retriable failure; falling back to chat", file=sys.stderr)
            else:
                raise

    # Legacy fallback: Chat Completions
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens_eff,
    }
    if supports_temperature:
        payload["temperature"] = temp_eff
    try:
        data = _post_json_with_retry(
            stage=stage,
            endpoint_label="/v1/chat/completions",
            url=_CHAT_URL,
            payload=payload,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
        )
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict):
                    txt = p.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            return "\n".join(parts).strip()
        return ""
    except Exception:
        raise
