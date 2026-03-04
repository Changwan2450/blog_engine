"""Environment loading and model routing helpers (stdlib-only)."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
            value = value[1:-1]
        os.environ[key] = value


def get_model(stage: str) -> str:
    """Return stage-routed model name with global/default fallback."""
    stage_map = {
        "summarize": "OPENAI_MODEL_SUMMARIZE",
        "write": "OPENAI_MODEL_WRITE",
        "critic": "OPENAI_MODEL_CRITIC",
    }
    env_key = stage_map.get(stage.strip().lower(), "")
    if env_key:
        value = os.environ.get(env_key, "").strip()
        if value:
            return value

    base = os.environ.get("OPENAI_MODEL", "").strip()
    if base:
        return base
    return "gpt-4o-mini"
