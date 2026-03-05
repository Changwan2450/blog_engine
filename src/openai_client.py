"""Shared OpenAI caller with Responses API preference (stdlib only)."""
from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_loader import get_model


_RESPONSES_URL = "https://api.openai.com/v1/responses"
_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_DEFAULT_TIMEOUT = 35


def _use_responses_api() -> bool:
    raw = os.environ.get("OPENAI_USE_RESPONSES", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    return body[:4000]


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
        return json.loads(resp.read().decode("utf-8"))


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
    timeout = _DEFAULT_TIMEOUT
    supports_temperature = not model.lower().startswith("gpt-5")

    # Preferred path: Responses API
    if _use_responses_api():
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_tokens,
        }
        if supports_temperature:
            payload["temperature"] = temperature
        try:
            data = _post_json(_RESPONSES_URL, payload, api_key=api_key, timeout=timeout)
            text = _extract_text_from_responses(data)
            if text:
                return text
            raise RuntimeError("Responses API returned empty text")
        except HTTPError as exc:
            body = _read_http_error_body(exc)
            print(
                f"[openai] WARN  stage={stage} endpoint=/v1/responses status={exc.code} body={body}",
                file=sys.stderr,
            )
            # fall through to chat completions fallback
        except (URLError, json.JSONDecodeError, OSError, RuntimeError) as exc:
            print(f"[openai] WARN  stage={stage} responses failed: {exc}", file=sys.stderr)

    # Legacy fallback: Chat Completions
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
    }
    if supports_temperature:
        payload["temperature"] = temperature
    try:
        data = _post_json(_CHAT_URL, payload, api_key=api_key, timeout=timeout)
        return data["choices"][0]["message"]["content"].strip()
    except HTTPError as exc:
        body = _read_http_error_body(exc)
        print(
            f"[openai] WARN  stage={stage} endpoint=/v1/chat/completions status={exc.code} body={body}",
            file=sys.stderr,
        )
        raise
