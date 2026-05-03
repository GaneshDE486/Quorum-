"""
Append-only log of LLM requests and responses (JSON Lines).

Writes to ``dataset/llm_output/llm_calls.jsonl`` under the project root.
Each line is one JSON object: timestamp, call site, what was sent (instructions + user
input), and what came back (or an error).

Thread-safe within one process via a module lock (sufficient for uvicorn worker threads).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_BASE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _BASE / "dataset" / "llm_output"
DEFAULT_LOG_FILE = OUTPUT_DIR / "llm_calls.jsonl"

_append_lock = threading.Lock()


def get_llm_log_path() -> Path:
    """Path to the JSONL log; override with env ``TI_LLM_LOG_FILE`` (absolute or cwd-relative)."""
    override = os.getenv("TI_LLM_LOG_FILE")
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = _BASE / p
        return p
    return DEFAULT_LOG_FILE


def append_llm_call_log(
    *,
    call_site: str,
    agent_name: str,
    agent_instructions: str,
    user_input: str,
    response: str | None = None,
    error: str | None = None,
    extras: Mapping[str, Any] | None = None,
    log_path: Path | None = None,
) -> None:
    """
    Append one JSON line with full prompt/context and raw model output (or failure).

    ``extras`` may include identifiers like conversation_id, parsed_category, temperature, model_id.
    """
    path = log_path if log_path is not None else get_llm_log_path()
    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_site": call_site,
        "agent_name": agent_name,
        "agent_instructions": agent_instructions,
        "user_input": user_input,
        "response": response,
        "error": error,
    }
    if extras:
        row["extras"] = dict(extras)

    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with _append_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
