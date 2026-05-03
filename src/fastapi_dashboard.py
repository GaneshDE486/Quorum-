"""
FastAPI dashboard for Transcript Intelligence.

Routes: GET /, GET /architecture, GET /api/dashboard, GET /api/conversation/{id},
        POST /api/conversation/{id}/llm-summarize.

Run:
  uvicorn src.fastapi_dashboard:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import markdown as markdown_lib
except ImportError:  # pragma: no cover - optional until pip install -r requirements.txt
    markdown_lib = None  # type: ignore[misc, assignment]

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from agents import Agent, ModelSettings, Runner  # noqa: E402

from llm_call_logger import append_llm_call_log  # noqa: E402
from llm_model_config import ModelSingleton  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CLASSIFICATION_CSV = BASE_DIR / "dataset/classification_output" / "all_classified.csv"
DATASET_ROOT = BASE_DIR / "dataset"
ARCHITECTURE_MD = BASE_DIR / "docs" / "ARCHITECTURE.md"
ARCHITECTURE_HTML = BASE_DIR / "docs" / "ARCHITECTURE.html"

CONVERSATION_ID_PATTERN = re.compile(r"^[0-9A-Z]{20,48}$")

LLM_AGENT_INSTRUCTIONS = (
    """You are an expert analyst of enterprise meetings. """
    """Transcript lines that begin with [HOST] were spoken by the meeting host / organizer."""
    """

Produce ONLY this Markdown (no preamble):
### Executive summary
3–6 sentences grounded in what was actually said.

### Key outcomes
Bulleted explicit decisions or commitments.

### Risks / open issues
Bulleted gaps, deadlines, escalations.

### Tone & dynamics
One short paragraph.

Rules: do not invent. If unsure, write "Not stated."
"""
)


def validate_conversation_id(conversation_id: str) -> str:
    cid = conversation_id.strip()
    if not CONVERSATION_ID_PATTERN.fullmatch(cid):
        raise HTTPException(status_code=400, detail="Invalid conversation id.")
    folder = DATASET_ROOT / cid
    if folder.is_dir() and (folder / "meeting-info.json").is_file():
        return cid
    raise HTTPException(status_code=404, detail="Conversation bundle not found.")


def email_local_parts_to_guess(email: str) -> str:
    local = email.split("@", 1)[0].strip().lower().replace("_", ".")
    parts = [p for p in local.split(".") if p]
    return " ".join(p.capitalize() for p in parts)


def resolve_host_display_name(host_email: str, speaker_meta_names: list[str]) -> str:
    if not host_email:
        return "unknown"
    if "@" not in host_email:
        return host_email
    guess = email_local_parts_to_guess(host_email)
    by_lower = {n.lower(): n for n in speaker_meta_names}
    if guess.lower() in by_lower:
        return by_lower[guess.lower()]
    gp = "".join(guess.lower().split())
    for n in speaker_meta_names:
        if "".join(n.lower().split()) == gp:
            return n
    return guess or email_local_parts_to_guess(host_email)


def aggregate_join_leave(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    joins: dict[str, list[float]] = defaultdict(list)
    leaves: dict[str, list[float]] = defaultdict(list)
    join_rows = leave_rows = 0
    for ev in events:
        name = str(ev.get("participantName") or "").strip()
        if not name:
            continue
        typ = ev.get("type")
        try:
            t_sec = float(ev.get("time")) if ev.get("time") is not None else None
        except (TypeError, ValueError):
            t_sec = None
        if t_sec is None:
            continue
        if typ == "Join":
            joins[name].append(t_sec)
            join_rows += 1
        elif typ == "Leave":
            leaves[name].append(t_sec)
            leave_rows += 1
    timeline: list[dict[str, Any]] = []
    all_names = set(joins) | set(leaves)
    for name in sorted(all_names):
        j_times = joins.get(name, [])
        l_times = leaves.get(name, [])
        timeline.append(
            {
                "participant_name": name,
                "first_join_seconds": min(j_times) if j_times else None,
                "last_leave_seconds": max(l_times) if l_times else None,
            }
        )
    return timeline, join_rows, leave_rows


def build_plain_transcript(transcript_obj: dict[str, Any], host_name: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    host_l = host_name.strip().lower() if host_name else ""
    for item in transcript_obj.get("data") or []:
        text = item.get("sentence")
        if text is None:
            continue
        sp_raw = item.get("speaker_name")
        sp = str(sp_raw).strip() if sp_raw is not None else ""
        lines.append(
            {
                "speaker": sp or str(item.get("speaker_id", "")),
                "text": str(text),
                "is_host": bool(host_l and sp.lower() == host_l),
            }
        )
    return lines


def format_transcript_plaintext(turns: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for t in turns:
        pref = "[HOST] " if t.get("is_host") else ""
        out.append(pref + str(t["speaker"]) + ": " + str(t["text"]))
    return "\n".join(out)


def load_conversation_detail(conversation_id: str) -> dict[str, Any]:
    root = DATASET_ROOT / conversation_id
    mi_path = root / "meeting-info.json"
    sm_path = root / "speaker-meta.json"
    ev_path = root / "events.json"
    su_path = root / "summary.json"
    tr_path = root / "transcript.json"

    with open(mi_path, encoding="utf-8") as f:
        meeting_info = json.load(f)

    meta_raw: dict[str, Any] = {}
    if sm_path.is_file():
        with open(sm_path, encoding="utf-8") as f:
            meta_raw = json.load(f)
    participant_ids = list(meta_raw.keys())
    speaker_names_ordered: list[str] = []
    for sid in sorted(participant_ids, key=lambda x: int(x) if str(x).isdigit() else 0):
        speaker_names_ordered.append(str(meta_raw[sid]).strip())

    host_email = str(meeting_info.get("host") or meeting_info.get("organizerEmail") or "").strip()
    host_name = resolve_host_display_name(host_email, speaker_names_ordered)

    events_loaded: list[dict[str, Any]] = []
    if ev_path.is_file():
        with open(ev_path, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                events_loaded = loaded
    timeline, join_rows, leave_rows = aggregate_join_leave(events_loaded)

    summary_block: dict[str, Any] = {}
    if su_path.is_file():
        with open(su_path, encoding="utf-8") as f:
            summary_block = json.load(f)

    turns: list[dict[str, Any]] = []
    if tr_path.is_file():
        with open(tr_path, encoding="utf-8") as f:
            turns = build_plain_transcript(json.load(f), host_name)

    return {
        "conversation_id": conversation_id,
        "title": meeting_info.get("title"),
        "participant_count_speaker_meta": len(participant_ids),
        "participant_names_speaker_meta": speaker_names_ordered,
        "host_email": host_email or None,
        "host_resolved_name": host_name,
        "start_time": meeting_info.get("startTime"),
        "end_time": meeting_info.get("endTime"),
        "duration_minutes": meeting_info.get("duration"),
        "participant_timeline": timeline,
        "aggregate_join_records": join_rows,
        "aggregate_leave_records": leave_rows,
        "distinct_names_in_events": len({row["participant_name"] for row in timeline}),
        "saved_summary_text": summary_block.get("summary"),
        "topics": summary_block.get("topics"),
        "overall_sentiment": summary_block.get("overallSentiment"),
        "sentiment_score_saved": summary_block.get("sentimentScore"),
        "transcript_turns": len(turns),
        "transcript_plaintext_for_llm": format_transcript_plaintext(turns),
        "transcript_lines": [
            {"speaker": str(t["speaker"]), "text": str(t["text"]), "is_host": bool(t.get("is_host"))}
            for t in turns
        ],
    }


def summarize_transcript_via_llm(
    *,
    conversation_id: str,
    detail: dict[str, Any],
) -> str:
    model = ModelSingleton.get_instance()
    if model is None:
        raise HTTPException(status_code=503, detail="LLM is not configured (check .env keys).")

    transcript = detail.get("transcript_plaintext_for_llm") or ""
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="No transcript text loaded for summarization.")

    max_chars = 120_000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n...[truncated for model context window]…"

    user_blob = (
        f"conversation_id={conversation_id}\n"
        f"title={detail.get('title')}\n"
        f"host_resolved_name={detail.get('host_resolved_name')}\n"
        f"host_email={detail.get('host_email')}\n\n"
        f"Transcript:\n{transcript}"
    )

    agent = Agent(
        name="summarizer",
        instructions=LLM_AGENT_INSTRUCTIONS,
        model=model,
        model_settings=ModelSettings(temperature=0.2),
    )
    model_id = os.getenv("model_name")
    extras: dict[str, Any] = {"conversation_id": conversation_id, "temperature": 0.2}
    if model_id:
        extras["model_name"] = model_id

    result = None
    try:
        result = Runner.run_sync(agent, user_blob)
    except Exception as exc:
        append_llm_call_log(
            call_site="summarize_transcript_via_llm",
            agent_name="summarizer",
            agent_instructions=LLM_AGENT_INSTRUCTIONS,
            user_input=user_blob,
            response=None,
            error=str(exc),
            extras=extras,
        )
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc!s}") from exc
    text = getattr(result, "final_output", None) if result else None
    if text is None:
        append_llm_call_log(
            call_site="summarize_transcript_via_llm",
            agent_name="summarizer",
            agent_instructions=LLM_AGENT_INSTRUCTIONS,
            user_input=user_blob,
            response=None,
            error="Empty model response (final_output is None)",
            extras=extras,
        )
        raise HTTPException(status_code=502, detail="Empty model response.")
    out = str(text).strip()
    append_llm_call_log(
        call_site="summarize_transcript_via_llm",
        agent_name="summarizer",
        agent_instructions=LLM_AGENT_INSTRUCTIONS,
        user_input=user_blob,
        response=out,
        error=None,
        extras=extras,
    )
    return out

STAKEHOLDER_VIEWS: list[dict[str, Any]] = [
    {
        "key": "support_leaders",
        "label": "Support Operations",
        "categories": ["A", "B", "C"],
    },
    {
        "key": "sales_managers",
        "label": "Revenue & Growth",
        "categories": ["D", "E", "F"],
    },
    {
        "key": "product_managers",
        "label": "Product Strategy",
        "categories": ["G", "E", "F"],
    },
    {
        "key": "engineering_leads",
        "label": "Engineering Execution",
        "categories": ["H", "B", "I"],
    },
]

CATEGORY_LABELS: dict[str, str] = {
    "A": "Technical Support & Troubleshooting",
    "B": "Platform Outage & Incident Response",
    "C": "Customer Outage Fallout & Churn Risk",
    "D": "Contract Renewal & Commercial Negotiation",
    "E": "Comply v2 Launch & Go-to-Market",
    "F": "Competitive Threat & Win/Loss Analysis",
    "G": "Customer Product Feedback & Feature Requests",
    "H": "Engineering Planning & Execution",
    "I": "Compliance & Audit Preparation",
}

# Sub-themes shown under each letter (curated taxonomy; meetings are not auto-tagged to sub-lines).
CATEGORY_SUB_THEMES: dict[str, list[str]] = {
    "A": [
        "A1 — Break-fix resolution",
        "A2 — Known issue management",
        "A3 — Integration failures",
    ],
    "B": [
        "B1 — War room & active response",
        "B2 — Root cause & post-mortem",
        "B3 — Remediation execution",
    ],
    "C": [
        "C1 — Immediate escalation (during outage)",
        "C2 — SLA credit & make-right",
        "C3 — Competitive evaluation triggered",
    ],
    "D": [
        "D1 — Standard renewal (healthy)",
        "D2 — At-risk renewal",
        "D3 — Expansion-led renewal",
    ],
    "E": [
        "E1 — Demos & preview / onboarding",
        "E2 — Launch readiness & deployment",
        "E3 — Early adoption & stabilization",
    ],
    "F": [
        "F1 — Active competitive deal",
        "F2 — Win / loss review",
        "F3 — Landscape intelligence",
    ],
    "G": [
        "G1 — Feature gaps & missing functionality",
        "G2 — Usability & workflow pain",
        "G3 — Framework-specific asks (CMMC, templates)",
    ],
    "H": [
        "H1 — Sprint cadence",
        "H2 — Roadmap & architecture",
        "H3 — Cross-team dependencies",
    ],
    "I": [
        "I1 — Internal audit readiness",
        "I2 — Evidence packaging & automation",
        "I3 — Control gap remediation",
    ],
}

app = FastAPI(title="Transcript Intelligence Dashboard", docs_url=None, redoc_url=None)


# ========= START OF load_classification_rows =========
def load_classification_rows() -> list[dict[str, Any]]:
    """
    Read all_classified.csv into a list of row dicts.

    Expected input: CLASSIFICATION_CSV path on disk (UTF-8 CSV).

    Output: list of plain dicts with string keys matching CSV columns; empty list if missing file.
    """
    if not CLASSIFICATION_CSV.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(CLASSIFICATION_CSV, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {k: (v if v != "" else None) for k, v in raw.items()}
            rows.append(row)
    return rows


# ========= END OF load_classification_rows =========


# ========= START OF build_dashboard_payload =========
def build_dashboard_payload() -> dict[str, Any]:
    """
    Aggregate rows into stakeholder summaries for the SPA.

    Expected input: uses load_classification_rows() internally.

    Output: dict with keys stakeholders (list), rows (list), csv_path (str), row_count (int).

    Gives: counts per category letter inside each stakeholder view plus global stats.
    """
    rows = load_classification_rows()
    stakeholders_out: list[dict[str, Any]] = []
    rules_n = llm_n = 0
    for row in rows:
        src = (row.get("classification_source") or "").lower()
        if src == "llm":
            llm_n += 1
        elif src == "rules":
            rules_n += 1
    for view in STAKEHOLDER_VIEWS:
        allowed = set(view["categories"])
        filtered = [r for r in rows if r.get("category") in allowed]
        counts: dict[str, int] = {c: 0 for c in view["categories"]}
        for r in filtered:
            cat = r.get("category")
            if cat in counts:
                counts[str(cat)] += 1
        c_sent = []
        for r in filtered:
            if r.get("category") == "C" and r.get("sentiment_score") is not None:
                try:
                    c_sent.append(float(r["sentiment_score"]))
                except (ValueError, TypeError):
                    pass
        avg_c = sum(c_sent) / len(c_sent) if c_sent else None
        stakeholders_out.append(
            {
                "key": view["key"],
                "label": view["label"],
                "categories": view["categories"],
                "counts": counts,
                "meeting_count": len(filtered),
                "avg_sentiment_category_c": round(avg_c, 3) if avg_c is not None else None,
            }
        )
    return {
        "stakeholders": stakeholders_out,
        "rows": rows,
        "row_count": len(rows),
        "classification_source": {"rules": rules_n, "llm": llm_n},
        "csv_path": str(CLASSIFICATION_CSV.relative_to(BASE_DIR)),
    }


# ========= END OF build_dashboard_payload =========


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transcript Intelligence — Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400..800;1,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-deep: #0f172a;
      --bg: #1e293b;
      --bg-elevated: #243547;
      --panel: rgba(30, 41, 59, 0.72);
      --panel-solid: #2d3f54;
      --glass: rgba(51, 65, 85, 0.45);
      --border: rgba(148, 163, 184, 0.22);
      --border-strong: rgba(148, 163, 184, 0.38);
      --text: #f1f5f9;
      --muted: #cbd5e1;
      --hint: #94a3b8;
      --accent: #38bdf8;
      --accent-bright: #7dd3fc;
      --accent-dim: rgba(56, 189, 248, 0.22);
      --accent-glow: rgba(56, 189, 248, 0.35);
      --violet: #a78bfa;
      --teal: #2dd4bf;
      --amber: #fbbf24;
      --danger: #f87171;
      --danger-bg: rgba(248, 113, 113, 0.12);
      --danger-border: rgba(248, 113, 113, 0.45);
      --radius-lg: 20px;
      --radius-md: 14px;
      --radius-sm: 10px;
      --font: "Plus Jakarta Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, monospace;
      --shadow-lg: 0 25px 50px -12px rgba(15, 23, 42, 0.55);
      --shadow-card: 0 4px 24px rgba(15, 23, 42, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    *,
    *::before,
    *::after {
      box-sizing: border-box;
    }
    html {
      scroll-behavior: smooth;
    }
    @media (prefers-reduced-motion: reduce) {
      html {
        scroll-behavior: auto;
      }
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
    body.ti-app {
      margin: 0;
      font-family: var(--font);
      background-color: var(--bg-deep);
      background-image:
        radial-gradient(ellipse 120% 80% at 0% -20%, rgba(56, 189, 248, 0.15) 0%, transparent 55%),
        radial-gradient(ellipse 90% 50% at 100% 0%, rgba(167, 139, 250, 0.12) 0%, transparent 45%),
        radial-gradient(ellipse 70% 40% at 50% 100%, rgba(45, 212, 191, 0.08) 0%, transparent 40%),
        linear-gradient(180deg, #0f172a 0%, #1e293b 40%, #172033 100%);
      background-attachment: fixed;
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }
    ::selection {
      background: rgba(56, 189, 248, 0.35);
      color: var(--text);
    }
    .ti-shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header.ti-header {
      position: sticky;
      top: 0;
      z-index: 40;
      padding: 0;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(20px) saturate(140%);
      -webkit-backdrop-filter: blur(20px) saturate(140%);
      box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .ti-header-inner {
      max-width: 1440px;
      margin: 0 auto;
      padding: 1.1rem clamp(1.25rem, 4vw, 2.25rem);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      flex-wrap: wrap;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .brand-mark {
      width: 2.85rem;
      height: 2.85rem;
      border-radius: var(--radius-sm);
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 0.82rem;
      letter-spacing: 0.06em;
      color: #0f172a;
      background: linear-gradient(135deg, var(--accent-bright) 0%, var(--accent) 45%, var(--teal) 100%);
      box-shadow:
        0 4px 14px rgba(56, 189, 248, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.35);
    }
    header.ti-header h1 {
      margin: 0;
      font-size: clamp(1.15rem, 2.5vw, 1.35rem);
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 1.2;
    }
    header.ti-header p {
      margin: 0.3rem 0 0;
      color: var(--hint);
      font-size: 0.875rem;
      font-weight: 450;
      max-width: 28rem;
      line-height: 1.45;
    }
    .ti-header-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
    }
    .ti-header-tag {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent-bright);
      padding: 0.38rem 0.85rem;
      border-radius: 999px;
      border: 1px solid rgba(56, 189, 248, 0.35);
      background: rgba(56, 189, 248, 0.1);
      white-space: nowrap;
    }
    .arch-doc-link {
      font-size: 0.78rem;
      font-weight: 650;
      color: var(--accent-bright);
      text-decoration: none;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      border: 1px solid rgba(56, 189, 248, 0.38);
      background: rgba(56, 189, 248, 0.1);
      white-space: nowrap;
      transition: background 0.15s ease, color 0.15s ease;
    }
    .arch-doc-link:hover {
      background: rgba(56, 189, 248, 0.22);
      color: var(--text);
    }
    main.ti-main {
      flex: 1;
      width: 100%;
      max-width: 1440px;
      margin: 0 auto;
      padding: clamp(1.25rem, 4vw, 2rem) clamp(1.25rem, 4vw, 2.25rem)
        clamp(2.5rem, 6vw, 3.75rem);
    }
    #status.empty.status-card {
      border-radius: var(--radius-lg);
      padding: 2.5rem 2rem;
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-card);
      max-width: 28rem;
      margin: 2rem auto;
    }
    .ti-app-shell {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .glass-panel--subcats {
      padding: 0.65rem clamp(0.85rem, 2vw, 1.15rem);
    }
    .subcat-matrix {
      display: grid;
      grid-template-columns: repeat(9, minmax(0, 1fr));
      gap: 0.45rem 0.55rem;
      align-items: start;
    }
    @media (max-width: 1100px) {
      .subcat-matrix {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }
    @media (max-width: 520px) {
      .subcat-matrix {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    .subcat-col {
      display: flex;
      flex-direction: column;
      gap: 0.32rem;
      min-width: 0;
      padding: 0.4rem 0.38rem;
      border-radius: var(--radius-sm);
      background: rgba(15, 23, 42, 0.35);
      border: 1px solid var(--border);
      border-top: 3px solid #38bdf8;
    }
    .subcat-col.ac1 {
      border-top-color: #2dd4bf;
    }
    .subcat-col.ac2 {
      border-top-color: #fb923c;
    }
    .subcat-col-hd {
      font-size: 0.78rem;
      font-weight: 800;
      text-align: center;
      color: var(--accent-bright);
      letter-spacing: 0.04em;
      line-height: 1;
      padding-bottom: 0.15rem;
      border-bottom: 1px solid var(--border);
    }
    .subcat-col-chips {
      display: flex;
      flex-direction: column;
      gap: 0.28rem;
    }
    .subcat-chip {
      font-size: 0.62rem;
      line-height: 1.38;
      color: var(--muted);
      padding: 0.28rem 0.35rem;
      border-radius: 6px;
      background: rgba(15, 23, 42, 0.45);
      border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .glass-panel {
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      background: var(--panel);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      box-shadow: var(--shadow-card);
      padding: 1rem clamp(1rem, 2.5vw, 1.35rem);
    }
    .glass-panel-board {
      padding-bottom: clamp(1rem, 2.5vw, 1.35rem);
    }
    .glass-panel--dense {
      padding: 0.65rem clamp(0.85rem, 2vw, 1.15rem);
    }
    .glass-panel .meta {
      margin-bottom: 0;
    }
    code {
      font-family: var(--font-mono);
      font-size: 0.9em;
      padding: 0.12rem 0.35rem;
      border-radius: 6px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
    }
    .tabs {
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
      align-items: center;
      margin: 0;
      padding: 0;
    }
    .tabs.track {
      gap: 0.25rem;
      padding: 0.35rem;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.45);
      border: 1px solid var(--border);
      width: 100%;
      justify-content: flex-start;
    }
    .tabs button {
      border: none;
      background: transparent;
      color: var(--hint);
      padding: 0.55rem 1.05rem;
      border-radius: 999px;
      cursor: pointer;
      font-size: 0.875rem;
      font-family: inherit;
      font-weight: 500;
      transition:
        color 0.18s ease,
        background 0.18s ease,
        box-shadow 0.18s ease;
    }
    .tabs button:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.05);
    }
    .tabs button:focus-visible {
      outline: none;
      box-shadow:
        0 0 0 2px var(--bg-deep),
        0 0 0 4px var(--accent);
    }
    .tabs button.active {
      background: linear-gradient(
        135deg,
        rgba(56, 189, 248, 0.25) 0%,
        rgba(56, 189, 248, 0.12) 100%
      );
      color: var(--text);
      font-weight: 650;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.62rem;
      align-items: center;
      margin-bottom: 0.85rem;
      font-size: 0.82rem;
      color: var(--hint);
      line-height: 1.4;
    }
    .meta strong {
      color: var(--text);
      font-weight: 650;
    }
    .meta-pill {
      padding: 0.42rem 0.92rem;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.22);
      font-size: 0.78rem;
      transition:
        border-color 0.15s ease,
        transform 0.15s ease;
    }
    .meta-pill:hover {
      border-color: rgba(56, 189, 248, 0.4);
    }
    .board-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem 1.5rem;
      align-items: flex-end;
      justify-content: space-between;
      margin: 0;
    }
    .board-toolbar label {
      font-size: 0.75rem;
      color: var(--hint);
      display: block;
      margin-bottom: 0.4rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .toolbar-group {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      min-width: 0;
    }
    .sort-select {
      padding: 0.62rem 0.95rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-strong);
      background: var(--panel-solid);
      color: var(--text);
      font-size: 0.875rem;
      font-family: inherit;
      min-width: 210px;
      cursor: pointer;
      transition:
        border-color 0.15s ease,
        box-shadow 0.15s ease;
    }
    .sort-select:hover {
      border-color: rgba(56, 189, 248, 0.45);
    }
    .sort-select:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .search {
      flex: 1;
      min-width: min(100%, 240px);
      max-width: 440px;
      padding: 0.65rem 1rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-strong);
      background: var(--panel-solid);
      color: var(--text);
      font-size: 0.925rem;
      font-family: inherit;
      transition:
        border-color 0.15s ease,
        box-shadow 0.15s ease;
    }
    .search::placeholder {
      color: var(--hint);
      opacity: 0.85;
    }
    .search:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
      gap: clamp(1rem, 2.5vw, 1.35rem);
      align-items: stretch;
    }
    .cat-card {
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      background: linear-gradient(
        165deg,
        rgba(51, 65, 85, 0.55) 0%,
        rgba(30, 41, 59, 0.85) 100%
      );
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-height: 300px;
      box-shadow: var(--shadow-lg);
      transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        border-color 0.2s ease;
      position: relative;
    }
    .cat-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      opacity: 0.5;
      background: radial-gradient(
        ellipse 85% 50% at 15% -5%,
        rgba(56, 189, 248, 0.12),
        transparent 55%
      );
    }
    .cat-card:hover {
      transform: translateY(-3px);
      border-color: rgba(56, 189, 248, 0.25);
      box-shadow:
        var(--shadow-lg),
        0 0 40px rgba(56, 189, 248, 0.08);
    }
    .cat-card.ac0 {
      border-top: 4px solid #38bdf8;
    }
    .cat-card.ac1 {
      border-top: 4px solid #2dd4bf;
    }
    .cat-card.ac2 {
      border-top: 4px solid #fb923c;
    }
    .cat-card-head {
      padding: 1.05rem 1.2rem;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.35);
      position: relative;
      z-index: 1;
    }
    .cat-card-head-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.85rem;
      margin-bottom: 0.55rem;
    }
    .letter-mark {
      font-size: 1.95rem;
      font-weight: 800;
      line-height: 1;
      letter-spacing: -0.06em;
      background: linear-gradient(
        135deg,
        #e2e8f0 0%,
        var(--accent-bright) 50%,
        var(--teal) 115%
      );
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .count-pill {
      font-variant-numeric: tabular-nums;
      font-weight: 700;
      font-size: 0.78rem;
      padding: 0.32rem 0.72rem;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent-bright);
      border: 1px solid rgba(56, 189, 248, 0.35);
      white-space: nowrap;
    }
    .cat-title {
      font-size: 0.935rem;
      font-weight: 600;
      color: var(--text);
      line-height: 1.4;
      margin: 0;
    }
    .sub-list {
      margin: 0;
      padding: 0.8rem 1.2rem 0.92rem;
      list-style: none;
      font-size: 0.76rem;
      color: var(--muted);
      line-height: 1.6;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.25);
      position: relative;
      z-index: 1;
    }
    .sub-list li {
      padding: 0.22rem 0 0.22rem 1.05rem;
      position: relative;
    }
    .sub-list li::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0.68em;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), var(--violet));
      opacity: 0.92;
      box-shadow: 0 0 8px var(--accent-glow);
    }
    .meetings-box {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
      position: relative;
      z-index: 1;
    }
    .meetings-box h4 {
      margin: 0;
      padding: 0.72rem 1.2rem 0.42rem;
      font-size: 0.66rem;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: var(--hint);
      font-weight: 700;
    }
    .meetings-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 0 0.5rem 0.9rem;
      max-height: 520px;
      scrollbar-width: thin;
      scrollbar-color: rgba(56, 189, 248, 0.35) transparent;
    }
    .meetings-scroll::-webkit-scrollbar {
      width: 6px;
    }
    .meetings-scroll::-webkit-scrollbar-thumb {
      background: rgba(56, 189, 248, 0.25);
      border-radius: 99px;
    }
    .meeting {
      padding: 0.85rem 0.92rem;
      margin: 0.42rem 0.35rem;
      border-radius: var(--radius-sm);
      background: rgba(15, 23, 42, 0.45);
      border: 1px solid var(--border);
      font-size: 0.81rem;
      transition:
        border-color 0.18s ease,
        background 0.18s ease,
        box-shadow 0.18s ease;
    }
    .meeting:hover {
      border-color: rgba(56, 189, 248, 0.35);
      background: rgba(56, 189, 248, 0.06);
      box-shadow: 0 4px 20px rgba(15, 23, 42, 0.3);
    }
    .meeting.meeting--alert {
      border-color: rgba(248, 113, 113, 0.45);
      background: linear-gradient(
        145deg,
        rgba(248, 113, 113, 0.1) 0%,
        rgba(15, 23, 42, 0.55) 100%
      );
      box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.15);
    }
    .signal-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      align-items: center;
      margin-bottom: 0.48rem;
    }
    .tag {
      font-size: 0.62rem;
      font-weight: 750;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      padding: 0.22rem 0.52rem;
      border-radius: 999px;
    }
    .tag-red {
      background: var(--danger-bg);
      color: #fecaca;
      border: 1px solid var(--danger-border);
    }
    .meeting-title {
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.45rem;
      line-height: 1.42;
      font-size: 0.9rem;
      letter-spacing: -0.01em;
    }
    .meeting-details {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.32rem 0.92rem;
      font-size: 0.735rem;
      color: var(--hint);
      margin-top: 0.48rem;
      padding-top: 0.52rem;
      border-top: 1px dashed var(--border);
    }
    .meeting-details dt {
      font-weight: 650;
      color: var(--accent-bright);
      font-size: 0.67rem;
      text-transform: uppercase;
      letter-spacing: 0.055em;
    }
    .meeting-details dd {
      margin: 0;
      color: var(--text);
      font-weight: 480;
      word-break: break-word;
    }
    .meeting-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.42rem;
      align-items: center;
      font-size: 0.73rem;
      color: var(--hint);
      margin-bottom: 0.12rem;
    }
    .badge {
      display: inline-block;
      padding: 0.14rem 0.48rem;
      border-radius: 999px;
      font-size: 0.67rem;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .badge.rules {
      background: rgba(45, 212, 191, 0.15);
      color: #99f6e4;
      border: 1px solid rgba(45, 212, 191, 0.4);
    }
    .badge.llm {
      background: rgba(251, 191, 36, 0.14);
      color: #fef08a;
      border: 1px solid rgba(251, 191, 36, 0.42);
    }
    .empty {
      color: var(--muted);
      padding: 2rem;
      text-align: center;
    }
    .empty-mini {
      padding: 1.65rem;
      margin: 0.42rem;
      border-radius: var(--radius-sm);
      border: 1px dashed var(--border-strong);
      text-align: center;
      font-size: 0.82rem;
      color: var(--hint);
      background: rgba(15, 23, 42, 0.25);
    }
    .meet-actions {
      margin-top: 0.72rem;
      padding-top: 0.62rem;
      border-top: 1px dashed var(--border);
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .btn-explore {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.45rem 0.92rem;
      border-radius: 999px;
      border: 1px solid rgba(56, 189, 248, 0.45);
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent-bright);
      cursor: pointer;
      font-family: inherit;
      transition:
        background 0.18s ease,
        border-color 0.18s ease,
        color 0.18s ease,
        transform 0.15s ease;
    }
    .btn-explore:hover {
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--accent-bright);
      color: var(--text);
    }
    .btn-explore:focus-visible {
      outline: none;
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .btn-explore-active {
      border-color: var(--accent-bright);
      background: rgba(56, 189, 248, 0.28);
      color: #fff;
    }
    .meet-flyout {
      margin-top: 0.72rem;
      padding: 1rem 1.05rem;
      border-radius: var(--radius-md);
      border: 1px solid rgba(56, 189, 248, 0.28);
      background: rgba(15, 23, 42, 0.88);
      font-size: 0.745rem;
      box-shadow: 0 12px 40px rgba(15, 23, 42, 0.5);
    }
    .fly-hint {
      margin: 0 0 0.65rem;
      color: var(--muted);
      font-size: 0.73rem;
      line-height: 1.5;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.42rem;
      margin-bottom: 0.72rem;
    }
    .q-chip {
      font-size: 0.68rem;
      font-weight: 600;
      padding: 0.32rem 0.72rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(30, 41, 59, 0.8);
      color: var(--muted);
      cursor: pointer;
      font-family: inherit;
      transition:
        color 0.15s ease,
        border-color 0.15s ease,
        background 0.15s ease;
    }
    .q-chip:hover {
      color: var(--text);
      border-color: rgba(56, 189, 248, 0.4);
      background: rgba(56, 189, 248, 0.08);
    }
    .q-chip.on {
      border-color: var(--accent);
      color: var(--text);
      background: linear-gradient(
        135deg,
        rgba(56, 189, 248, 0.28) 0%,
        rgba(56, 189, 248, 0.1) 100%
      );
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .fly-section {
      display: none;
      margin-top: 0.4rem;
    }
    .fly-section.vis {
      display: block;
      animation: tiFade 0.2s ease;
    }
    @keyframes tiFade {
      from {
        opacity: 0;
        transform: translateY(4px);
      }
      to {
        opacity: 1;
        transform: none;
      }
    }
    .fly-section h5 {
      margin: 0 0 0.48rem;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--hint);
      font-weight: 750;
    }
    .fly-section ul {
      margin: 0;
      padding-left: 1.1rem;
      color: var(--text);
      font-size: 0.74rem;
    }
    .fly-section li {
      margin: 0.22rem 0;
    }
    .kv-grid {
      display: grid;
      grid-template-columns: minmax(10rem, 34%) 1fr;
      gap: 0.45rem 1rem;
      align-items: start;
      margin: 0.4rem 0 0.6rem;
      font-size: 0.74rem;
    }
    .kv-key {
      color: var(--accent-bright);
      font-weight: 650;
      letter-spacing: 0.035em;
      line-height: 1.42;
      border-radius: var(--radius-sm);
      padding: 0.15rem 0.35rem;
      background: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent);
    }
    .kv-val {
      color: #f1f5f9;
      font-weight: 450;
      line-height: 1.48;
      word-break: break-word;
      padding: 0.15rem 0;
    }
    .party-list-wrap {
      margin: 0.28rem 0 0;
      grid-column: 1 / -1;
    }
    .party-list-wrap span.name-tag {
      display: inline-block;
      margin: 0.28rem 0.38rem 0 0;
      padding: 0.22rem 0.55rem;
      border-radius: 999px;
      background: rgba(167, 139, 250, 0.12);
      color: #ddd6fe;
      font-size: 0.72rem;
      border: 1px solid rgba(167, 139, 250, 0.38);
    }
    .join-block {
      font-size: 0.74rem;
      margin: 0.4rem 0;
      padding: 0.58rem 0.72rem;
      border-radius: var(--radius-sm);
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid var(--border);
      display: grid;
      grid-template-columns: minmax(8rem, 28%) 1fr;
      gap: 0.42rem 0.92rem;
    }
    .join-block .jn-name {
      color: #67e8f9;
      font-weight: 650;
      border-left: 3px solid #22d3ee;
      padding-left: 0.48rem;
    }
    .join-block .jn-detail {
      color: #e2e8f0;
    }
    .jn-sub {
      display: inline-block;
      margin-right: 0.88rem;
    }
    .jn-sub .jn-lbl {
      color: var(--amber);
      font-size: 0.64rem;
      font-weight: 700;
      margin-right: 0.28rem;
      letter-spacing: 0.03em;
    }
    .jn-sub .jn-num {
      color: #fef9c3;
    }
    .summary-prose {
      margin: 0 0 0.82rem;
      padding: 0.75rem 0.82rem;
      border-radius: var(--radius-sm);
      background: rgba(52, 211, 153, 0.08);
      border: 1px solid rgba(52, 211, 153, 0.26);
      color: #d1fae5;
      line-height: 1.58;
      font-size: 0.795rem;
    }
    .transcript-stream {
      max-height: 320px;
      overflow-y: auto;
      padding: 0.58rem;
      border-radius: var(--radius-sm);
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--border-strong);
      scrollbar-width: thin;
      scrollbar-color: rgba(167, 139, 250, 0.35) transparent;
    }
    .transcript-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 0.42rem;
      margin-bottom: 0.62rem;
      padding: 0.48rem 0.58rem;
      border-radius: var(--radius-sm);
      background: rgba(30, 41, 59, 0.55);
      border: 1px dashed var(--border-strong);
      font-size: 0.695rem;
      color: var(--muted);
      font-weight: 500;
    }
    .transcript-legend .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.38rem;
      padding: 0.18rem 0.52rem 0.18rem 0.42rem;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.45);
      border: 1px solid var(--border);
    }
    .legend-swatch {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
    }
    .t-turn {
      margin-bottom: 0.65rem;
      padding: 0.58rem 0.68rem;
      border-radius: var(--radius-sm);
      border-left: 4px solid transparent;
      font-size: 0.735rem;
      line-height: 1.48;
    }
    .t-turn-is-host {
      box-shadow: inset 0 0 0 1px rgba(251, 211, 141, 0.5);
      border-radius: var(--radius-sm);
    }
    .t-sp0 {
      border-left-color: #38bdf8;
      background: linear-gradient(
        105deg,
        rgba(56, 189, 248, 0.15) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp1 {
      border-left-color: #f472b6;
      background: linear-gradient(
        105deg,
        rgba(244, 114, 182, 0.14) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp2 {
      border-left-color: #a3e635;
      background: linear-gradient(
        105deg,
        rgba(163, 230, 53, 0.13) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp3 {
      border-left-color: #a78bfa;
      background: linear-gradient(
        105deg,
        rgba(167, 139, 250, 0.15) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp4 {
      border-left-color: #fb923c;
      background: linear-gradient(
        105deg,
        rgba(251, 146, 60, 0.14) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp5 {
      border-left-color: #2dd4bf;
      background: linear-gradient(
        105deg,
        rgba(45, 212, 191, 0.13) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp6 {
      border-left-color: #facc15;
      background: linear-gradient(
        105deg,
        rgba(250, 204, 21, 0.12) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp7 {
      border-left-color: #f87171;
      background: linear-gradient(
        105deg,
        rgba(248, 113, 113, 0.12) 0%,
        rgba(30, 41, 59, 0.7) 100%
      );
    }
    .t-sp0 .legend-swatch {
      background: #38bdf8;
    }
    .t-sp1 .legend-swatch {
      background: #f472b6;
    }
    .t-sp2 .legend-swatch {
      background: #a3e635;
    }
    .t-sp3 .legend-swatch {
      background: #a78bfa;
    }
    .t-sp4 .legend-swatch {
      background: #fb923c;
    }
    .t-sp5 .legend-swatch {
      background: #2dd4bf;
    }
    .t-sp6 .legend-swatch {
      background: #facc15;
    }
    .t-sp7 .legend-swatch {
      background: #f87171;
    }
    .t-speaker-line {
      display: flex;
      align-items: center;
      gap: 0.48rem;
      flex-wrap: wrap;
      margin-bottom: 0.38rem;
    }
    .t-role-badge {
      font-size: 0.565rem;
      font-weight: 800;
      letter-spacing: 0.07em;
      padding: 0.15rem 0.4rem;
      border-radius: 999px;
    }
    .t-sp0 .t-role-badge {
      background: rgba(56, 189, 248, 0.28);
      color: #bae6fd;
      border: 1px solid rgba(56, 189, 248, 0.45);
    }
    .t-sp1 .t-role-badge {
      background: rgba(244, 114, 182, 0.25);
      color: #fbcfe8;
      border: 1px solid rgba(244, 114, 182, 0.45);
    }
    .t-sp2 .t-role-badge {
      background: rgba(163, 230, 53, 0.22);
      color: #ecfccb;
      border: 1px solid rgba(163, 230, 53, 0.4);
    }
    .t-sp3 .t-role-badge {
      background: rgba(167, 139, 250, 0.26);
      color: #e9d5ff;
      border: 1px solid rgba(167, 139, 250, 0.45);
    }
    .t-sp4 .t-role-badge {
      background: rgba(251, 146, 60, 0.24);
      color: #fed7aa;
      border: 1px solid rgba(251, 146, 60, 0.45);
    }
    .t-sp5 .t-role-badge {
      background: rgba(45, 212, 191, 0.22);
      color: #ccfbf1;
      border: 1px solid rgba(45, 212, 191, 0.45);
    }
    .t-sp6 .t-role-badge {
      background: rgba(250, 204, 21, 0.22);
      color: #fef08a;
      border: 1px solid rgba(250, 204, 21, 0.45);
    }
    .t-sp7 .t-role-badge {
      background: rgba(248, 113, 113, 0.22);
      color: #fecaca;
      border: 1px solid rgba(248, 113, 113, 0.45);
    }
    .t-speaker-name {
      font-weight: 700;
      color: #f8fafc;
      font-size: 0.77rem;
    }
    .t-sp0 .t-speaker-name {
      color: #bae6fd;
    }
    .t-sp1 .t-speaker-name {
      color: #fbcfe8;
    }
    .t-sp2 .t-speaker-name {
      color: #ecfccb;
    }
    .t-sp3 .t-speaker-name {
      color: #ddd6fe;
    }
    .t-sp4 .t-speaker-name {
      color: #ffedd5;
    }
    .t-sp5 .t-speaker-name {
      color: #ccfbf1;
    }
    .t-sp6 .t-speaker-name {
      color: #fef9c3;
    }
    .t-sp7 .t-speaker-name {
      color: #fecaca;
    }
    .t-turn-text {
      white-space: pre-wrap;
      word-break: break-word;
      color: #e2e8f0;
    }
    .t-sp0 .t-turn-text {
      color: #f0f9ff;
    }
    .t-sp1 .t-turn-text {
      color: #fdf2f8;
    }
    .t-sp2 .t-turn-text {
      color: #f7fee7;
    }
    .t-sp3 .t-turn-text {
      color: #f5f3ff;
    }
    .t-sp4 .t-turn-text {
      color: #fffbeb;
    }
    .t-sp5 .t-turn-text {
      color: #f0fdfa;
    }
    .t-sp6 .t-turn-text {
      color: #fefce8;
    }
    .t-sp7 .t-turn-text {
      color: #fef2f2;
    }
    .trans-plain {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      padding: 0.72rem;
      max-height: 280px;
      overflow: auto;
      border-radius: var(--radius-sm);
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid var(--border-strong);
      font-size: 0.73rem;
      line-height: 1.44;
      color: #dae4f5;
      font-family: var(--font-mono);
    }
    .btn-llm {
      margin-top: 0.58rem;
      font-size: 0.75rem;
      font-weight: 650;
      padding: 0.45rem 0.92rem;
      border-radius: 999px;
      border: 1px solid rgba(251, 191, 36, 0.45);
      background: rgba(251, 191, 36, 0.12);
      color: var(--amber);
      cursor: pointer;
      font-family: inherit;
      transition:
        background 0.15s ease,
        transform 0.12s ease;
    }
    .btn-llm:hover:not(:disabled) {
      background: rgba(251, 191, 36, 0.22);
    }
    .btn-llm:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .llm-out {
      margin-top: 0.58rem;
      padding: 0.72rem;
      border-radius: var(--radius-sm);
      background: rgba(30, 41, 59, 0.95);
      border: 1px solid rgba(251, 191, 36, 0.32);
      font-size: 0.74rem;
      color: var(--text);
      white-space: pre-wrap;
      max-height: 360px;
      overflow: auto;
      font-family: var(--font-mono);
      line-height: 1.5;
    }
    .fly-err {
      color: #fecaca;
      font-size: 0.74rem;
    }
    .footnote {
      margin-top: 1rem;
      padding-top: 0.65rem;
      border-top: 1px solid var(--border);
      font-size: 0.78rem;
      color: var(--muted);
      line-height: 1.48;
      font-style: italic;
    }
  </style>
</head>
<body class="ti-app">
  <div class="ti-shell">
    <header class="ti-header">
      <div class="ti-header-inner">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">TI</span>
          <div>
            <h1>Transcript Intelligence</h1>
            <p>Stakeholder dashboards from classification output.</p>
          </div>
        </div>
        <div class="ti-header-actions">
          <span class="ti-header-tag" title="Unified meeting taxonomy">Insights · Taxonomy A–I</span>
          <a href="/architecture" class="arch-doc-link" title="Classification rules, LLM usage, data fields">Architecture plan</a>
        </div>
      </div>
    </header>
    <main class="ti-main">
      <div id="status" class="empty status-card">Loading…</div>
      <div id="app" class="ti-app-shell" style="display:none;">
        <section
          class="glass-panel glass-panel--dense glass-panel--subcats"
          aria-label="All taxonomy sub-themes A1 through I3 in columns A to I"
        >
          <div id="subcat-matrix"></div>
        </section>
        <section class="glass-panel glass-panel--dense">
          <div class="meta" id="global-meta"></div>
        </section>
        <section class="glass-panel glass-panel--dense">
          <nav class="tabs track" aria-label="Stakeholder views" id="tabs"></nav>
        </section>
        <section class="glass-panel">
          <div class="board-toolbar">
            <div class="toolbar-group" style="flex:1;">
              <label for="filter">Filter meetings</label>
              <input type="search" class="search" id="filter" placeholder="Title, conversation ID, or category…" autocomplete="off" />
            </div>
            <div class="toolbar-group">
              <label for="sort-date">Sort by meeting date</label>
              <select class="sort-select" id="sort-date" aria-label="Sort meetings by date">
                <option value="asc">Date — oldest first (A → Z)</option>
                <option value="desc">Date — newest first (Z → A)</option>
              </select>
            </div>
          </div>
        </section>
        <section class="glass-panel glass-panel-board">
          <section class="card-grid" id="card-grid"></section>
          <p class="footnote" id="footnote"></p>
        </section>
      </div>
    </main>
  </div>
  <script>
    const LABELS = __CATEGORY_LABELS_JSON__;
    const SUB_THEMES = __SUB_THEMES_JSON__;
    const TAXONOMY_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I"];

    function renderSubcategoryMatrix() {
      const root = document.getElementById("subcat-matrix");
      if (!root) return;
      let html = '<div class="subcat-matrix">';
      for (let i = 0; i < TAXONOMY_ORDER.length; i++) {
        const L = TAXONOMY_ORDER[i];
        const subs = SUB_THEMES[L] || [];
        const ac = "ac" + (i % 3);
        html += '<div class="subcat-col ' + ac + '">';
        html += '<div class="subcat-col-hd">' + escapeHtml(L) + "</div>";
        html += '<div class="subcat-col-chips">';
        if (subs.length) {
          for (let s = 0; s < subs.length; s++) {
            html += '<span class="subcat-chip">' + escapeHtml(String(subs[s])) + "</span>";
          }
        } else {
          html += '<span class="subcat-chip">—</span>';
        }
        html += "</div></div>";
      }
      html += "</div>";
      root.innerHTML = html;
    }

    let payload = null;
    let activeKey = null;
    const detailCache = Object.create(null);

    function fmtSecs(sec) {
      if (sec === null || sec === undefined || sec === "") return "—";
      const n = Number(sec);
      if (!Number.isFinite(n)) return String(sec);
      const m = Math.floor(n / 60);
      const s = Math.floor(n % 60);
      return m + ":" + String(s).padStart(2, "0") + " into call (" + Math.round(n) + "s)";
    }

    async function hydrateFlyout(cid, flyout) {
      const stack = flyout.querySelector("[data-explore-mount]");
      if (!stack || !cid) return;
      if (stack.dataset.rendered === "1") {
        return;
      }
      if (detailCache[cid]) {
        renderExploreInto(stack, detailCache[cid], cid);
        stack.dataset.rendered = "1";
        return;
      }
      stack.innerHTML =
        '<p class="fly-hint" style="color:#eab308">Reading speaker-meta.json, events.json, meeting-info.json, summary.json, transcript.json…</p>';
      try {
        const res = await fetch("/api/conversation/" + encodeURIComponent(cid));
        if (!res.ok) {
          const err = await res.text();
          throw new Error(err || "HTTP " + res.status);
        }
        detailCache[cid] = await res.json();
        renderExploreInto(stack, detailCache[cid], cid);
        stack.dataset.rendered = "1";
      } catch (err) {
        stack.innerHTML = '<div class="fly-err">Could not load: ' + escapeHtml(err.message) + "</div>";
      }
    }

    function fillTranscriptPane(paneTalk, d) {
      if (!paneTalk || paneTalk.querySelector(".transcript-stream")) return;
      const holder = document.createElement("div");
      holder.className = "transcript-stream";
      const lines = Array.isArray(d.transcript_lines) ? d.transcript_lines : [];
      const PALETTE = 8;
      if (!lines.length) {
        holder.textContent = "(No transcript lines in bundle.)";
        paneTalk.appendChild(holder);
        return;
      }
      const speakersFirst = [];
      const slotBySpeaker = Object.create(null);
      for (let si = 0; si < lines.length; si++) {
        const raw = String(lines[si].speaker || "").trim() || "—";
        if (!(raw in slotBySpeaker)) {
          slotBySpeaker[raw] = speakersFirst.length;
          speakersFirst.push(raw);
        }
      }
      const legend = document.createElement("div");
      legend.className = "transcript-legend";
      legend.appendChild(document.createTextNode("Speakers: "));
      for (let si = 0; si < speakersFirst.length; si++) {
        const name = speakersFirst[si];
        const idx = si % PALETTE;
        const speaksAsHost = lines.some(function (L) {
          return (String(L.speaker || "").trim() || "—") === name && Boolean(L.is_host);
        });
        const chip = document.createElement("span");
        chip.className = "legend-chip t-sp" + idx;
        const dot = document.createElement("span");
        dot.className = "legend-swatch";
        const lab = document.createElement("span");
        lab.textContent = name + (speaksAsHost ? " · host" : "");
        chip.appendChild(dot);
        chip.appendChild(lab);
        legend.appendChild(chip);
      }
      holder.appendChild(legend);
      for (let i = 0; i < lines.length; i++) {
        const turn = lines[i];
        const name = String(turn.speaker || "").trim() || "—";
        const slotIdx = slotBySpeaker[name];
        const lane = slotIdx % PALETTE;
        const isH = Boolean(turn.is_host);
        const wrap = document.createElement("div");
        wrap.className = "t-turn t-sp" + lane + (isH ? " t-turn-is-host" : "");
        const top = document.createElement("div");
        top.className = "t-speaker-line";
        const badge = document.createElement("span");
        badge.className = "t-role-badge";
        badge.textContent = (isH ? "HOST " : "") + "S" + (slotIdx + 1);
        const nm = document.createElement("span");
        nm.className = "t-speaker-name";
        nm.textContent = name;
        top.appendChild(badge);
        top.appendChild(nm);
        const body = document.createElement("div");
        body.className = "t-turn-text";
        body.textContent = String(turn.text || "");
        wrap.appendChild(top);
        wrap.appendChild(body);
        holder.appendChild(wrap);
      }
      paneTalk.appendChild(holder);
    }

    function renderExploreInto(mount, d, cid) {
      const namesHtml = Array.isArray(d.participant_names_speaker_meta)
        ? d.participant_names_speaker_meta
            .map(function (n) {
              return '<span class="name-tag">' + escapeHtml(String(n)) + "</span>";
            })
            .join("")
        : "—";
      const joinHtml = Array.isArray(d.participant_timeline)
        ? d.participant_timeline
            .map(function (row) {
              return (
                '<div class="join-block">' +
                '<div class="jn-name">' +
                escapeHtml(String(row.participant_name)) +
                "</div>" +
                '<div class="jn-detail">' +
                '<span class="jn-sub"><span class="jn-lbl">First join</span> <span class="jn-num">' +
                fmtSecs(row.first_join_seconds) +
                '</span></span><span class="jn-sub"><span class="jn-lbl">Last leave</span> <span class="jn-num">' +
                fmtSecs(row.last_leave_seconds) +
                "</span></span></div></div>"
              );
            })
            .join("")
        : '<div class="kv-val">—</div>';

      const tpl =
        '<p class="fly-hint">Data loaded from <code>dataset/' +
        escapeHtml(cid) +
        "/</code> — choose a lane.</p>" +
        '<div class="chip-row">' +
        '<button type="button" class="q-chip on" data-explore-pane="people">People</button>' +
        '<button type="button" class="q-chip" data-explore-pane="join">Joined / left</button>' +
        '<button type="button" class="q-chip" data-explore-pane="meet">Meeting & host</button>' +
        '<button type="button" class="q-chip" data-explore-pane="summary">Saved summary</button>' +
        '<button type="button" class="q-chip" data-explore-pane="talk">Plain transcript</button>' +
        '<button type="button" class="q-chip" data-explore-pane="llm">LLM summary</button>' +
        "</div>" +
        '<div data-pane="people" class="fly-section vis">' +
        "<h5>People (speaker-meta.json)</h5>" +
        '<div class="kv-grid">' +
        '<div class="kv-key">Roster count</div><div class="kv-val">' +
        escapeHtml(String(d.participant_count_speaker_meta != null ? d.participant_count_speaker_meta : "—")) +
        "</div>" +
        '<div class="kv-key">Distinct in events</div><div class="kv-val">' +
        escapeHtml(String(d.distinct_names_in_events != null ? d.distinct_names_in_events : "—")) +
        "</div>" +
        '<div class="kv-key">Join rows (events)</div><div class="kv-val">' +
        escapeHtml(String(d.aggregate_join_records != null ? d.aggregate_join_records : "—")) +
        "</div>" +
        '<div class="kv-key">Leave rows (events)</div><div class="kv-val">' +
        escapeHtml(String(d.aggregate_leave_records != null ? d.aggregate_leave_records : "—")) +
        "</div>" +
        '<div class="kv-key">Names on roster</div><div class="kv-val party-list-wrap">' +
        namesHtml +
        "</div></div></div>" +
        '<div data-pane="join" class="fly-section">' +
        "<h5>Join & leave timeline (events.json)</h5>" +
        joinHtml +
        "</div>" +
        '<div data-pane="meet" class="fly-section">' +
        "<h5>Scheduling (meeting-info.json)</h5>" +
        '<div class="kv-grid">' +
        '<div class="kv-key">Duration</div><div class="kv-val">' +
        escapeHtml(String(d.duration_minutes != null ? d.duration_minutes : "—")) +
        " minutes</div>" +
        '<div class="kv-key">Starts</div><div class="kv-val">' +
        escapeHtml(String(d.start_time || "—")) +
        "</div>" +
        '<div class="kv-key">Ends</div><div class="kv-val">' +
        escapeHtml(String(d.end_time || "—")) +
        "</div>" +
        '<div class="kv-key">Host (resolved)</div><div class="kv-val">' +
        escapeHtml(String(d.host_resolved_name || "—")) +
        "</div>" +
        '<div class="kv-key">Host email</div><div class="kv-val">' +
        escapeHtml(String(d.host_email || "—")) +
        "</div></div></div>" +
        '<div data-pane="summary" class="fly-section">' +
        "<h5>Saved narrative (summary.json)</h5>" +
        '<div class="summary-prose">' +
        escapeHtml(String(d.saved_summary_text || "—")) +
        "</div>" +
        '<div class="kv-grid">' +
        '<div class="kv-key">Topics</div><div class="kv-val">' +
        escapeHtml(Array.isArray(d.topics) ? d.topics.join(", ") : "—") +
        "</div>" +
        '<div class="kv-key">Overall sentiment</div><div class="kv-val">' +
        escapeHtml(String(d.overall_sentiment || "—")) +
        "</div>" +
        '<div class="kv-key">Sentiment score</div><div class="kv-val">' +
        escapeHtml(String(d.sentiment_score_saved != null ? d.sentiment_score_saved : "—")) +
        "</div>" +
        '<div class="kv-key">Transcript turns</div><div class="kv-val">' +
        escapeHtml(String(d.transcript_turns != null ? d.transcript_turns : "—")) +
        "</div></div></div>" +
        '<div data-pane="talk" class="fly-section">' +
        "<h5>Plain transcript (color per participant)</h5>" +
        '<p class="fly-hint" style="margin-top:0">Each distinct speaker gets a lane (S1&#8230;n). Over 8 people, colors repeat. Organizer lines also show HOST on the badge and a subtle frame.</p>' +
        "</div>" +
        '<div data-pane="llm" class="fly-section">' +
        "<h5>Fresh LLM synopsis</h5>" +
        '<div class="kv-grid" style="margin-bottom:0.65rem">' +
        '<div class="kv-key">Input</div><div class="kv-val">Full plain transcript + host metadata (server-side)</div>' +
        '<div class="kv-key">Output</div><div class="kv-val">Structured Markdown brief</div></div>' +
        '<button type="button" class="btn-llm" data-llm-cid="' +
        escapeHtml(cid) +
        '">Process transcript with LLM</button>' +
        '<pre class="llm-out" data-llm-out style="display:none;"></pre>' +
        "</div>";

      mount.innerHTML = tpl;

      const paneTalk = mount.querySelector('[data-pane="talk"]');
      fillTranscriptPane(paneTalk, d);

      mount.querySelectorAll("[data-explore-pane]").forEach(function (chip) {
        chip.addEventListener("click", function () {
          const key = chip.getAttribute("data-explore-pane");
          mount.querySelectorAll(".q-chip").forEach(function (c) {
            c.classList.toggle("on", c === chip);
          });
          mount.querySelectorAll("[data-pane]").forEach(function (sec) {
            sec.classList.toggle("vis", sec.getAttribute("data-pane") === key);
          });
        });
      });

      mount.querySelectorAll(".btn-llm").forEach(function (btn) {
        btn.addEventListener("click", async function () {
          const outNode = btn.parentElement.querySelector("[data-llm-out]");
          if (!outNode) return;
          btn.disabled = true;
          outNode.style.display = "block";
          outNode.textContent = "Calling model…";
          try {
            const res = await fetch(
              "/api/conversation/" + encodeURIComponent(cid) + "/llm-summarize",
              { method: "POST", headers: { Accept: "application/json" }, body: "" }
            );
            if (!res.ok) throw new Error((await res.text()) || String(res.status));
            const jr = await res.json();
            outNode.textContent = jr.markdown || JSON.stringify(jr);
          } catch (e) {
            outNode.textContent = "LLM error: " + e.message;
          } finally {
            btn.disabled = false;
          }
        });
      });

      mount.querySelectorAll("[data-pane]").forEach(function (sec) {
        sec.classList.toggle("vis", sec.getAttribute("data-pane") === "people");
      });
      mount.querySelectorAll(".q-chip").forEach(function (c) {
        c.classList.toggle("on", c.getAttribute("data-explore-pane") === "people");
      });
    }

    const URGENT_RE =
      /\\b(urgent|p0\\b|prio(?:rity)?\\s*0\\b|hotfix|hot-fix|hot fix|immediate|emergency|critical outage|severity[\\s:-]*(?:0|1)\\b|sev[\\s:-]*(?:0|1)\\b|war room|rollback|production down)\\b/i;

    function parseSentiment(row) {
      if (row.sentiment_score === null || row.sentiment_score === undefined || row.sentiment_score === "")
        return null;
      const n = Number(row.sentiment_score);
      return Number.isFinite(n) ? n : null;
    }

    function isUrgentMeeting(row) {
      return URGENT_RE.test(String(row.title || ""));
    }

    function isLowSentiment(row) {
      const n = parseSentiment(row);
      return n !== null && n < 3;
    }

    function getDateSortKey(row) {
      const d = String(row.start_date || "");
      return d || "~~~";
    }

    function compareByDateAsc(a, b) {
      return getDateSortKey(a).localeCompare(getDateSortKey(b));
    }

    function sortRowsByPreference(rows, order) {
      const copy = rows.slice();
      copy.sort((a, b) => {
        const c = compareByDateAsc(a, b);
        return order === "desc" ? -c : c;
      });
      return copy;
    }

    async function load() {
      const status = document.getElementById("status");
      try {
        const res = await fetch("/api/dashboard");
        if (!res.ok) throw new Error("HTTP " + res.status);
        payload = await res.json();
        status.style.display = "none";
        document.getElementById("app").style.display = "block";
        renderGlobal();
        renderTabs();
        if (payload.stakeholders.length) {
          selectTab(payload.stakeholders[0].key);
        }
      } catch (e) {
        status.textContent = "Could not load dashboard: " + e.message;
      }
    }

    function renderGlobal() {
      const el = document.getElementById("global-meta");
      const s = payload.classification_source;
      el.innerHTML =
        '<span class="meta-pill"><strong>Meetings</strong> ' + payload.row_count + "</span>" +
        '<span class="meta-pill"><strong>Rules</strong> ' + (s.rules || 0) + "</span>" +
        '<span class="meta-pill"><strong>LLM</strong> ' + (s.llm || 0) + "</span>" +
        '<span class="meta-pill"><strong>CSV</strong> ' + escapeHtml(payload.csv_path) + "</span>";
    }

    function renderTabs() {
      const tabs = document.getElementById("tabs");
      tabs.innerHTML = "";
      for (const sh of payload.stakeholders) {
        const b = document.createElement("button");
        b.textContent = sh.label + " (" + sh.meeting_count + ")";
        b.dataset.key = sh.key;
        b.onclick = () => selectTab(sh.key);
        tabs.appendChild(b);
      }
    }

    function selectTab(key) {
      activeKey = key;
      document.querySelectorAll(".tabs button").forEach((b) => {
        b.classList.toggle("active", b.dataset.key === key);
      });
      const sh = payload.stakeholders.find((x) => x.key === key);
      renderCategoryBoard(sh);
    }

    function renderCategoryBoard(sh) {
      const grid = document.getElementById("card-grid");
      const foot = document.getElementById("footnote");
      grid.innerHTML = "";
      foot.textContent = "";
      const q = (document.getElementById("filter").value || "").toLowerCase();
      let cardIndex = 0;
      if (sh.avg_sentiment_category_c != null) {
        foot.textContent =
          "Avg sentiment for category C in this stakeholder view: " + sh.avg_sentiment_category_c;
      }
      for (const c of sh.categories) {
        const subs = SUB_THEMES[c] || [];
        let rowsAll = payload.rows.filter((r) => r.category === c);
        if (q) {
          rowsAll = rowsAll.filter((r) => {
            const t = String(r.title || "").toLowerCase();
            const id = String(r.conversation_id || "").toLowerCase();
            const lab = String(r.category_label || "").toLowerCase();
            return t.includes(q) || id.includes(q) || lab.includes(q);
          });
        }
        const displayed = rowsAll.length;
        const total = sh.counts[c] || 0;
        const pillText =
          q && q.length
            ? displayed + " match · " + total + " in category"
            : total + (total === 1 ? " meeting" : " meetings");

        const art = document.createElement("article");
        art.className = "cat-card ac" + (cardIndex % 3);
        let subHtml = "";
        for (const line of subs) {
          subHtml += "<li>" + escapeHtml(line) + "</li>";
        }

        art.innerHTML =
          '<header class="cat-card-head">' +
          '<div class="cat-card-head-top">' +
          '<span class="letter-mark" aria-hidden="true">' +
          escapeHtml(c) +
          "</span>" +
          '<span class="count-pill">' + pillText + "</span></div>" +
          '<p class="cat-title">' +
          escapeHtml(LABELS[c] || "") +
          "</p></header>" +
          '<ul class="sub-list">' +
          subHtml +
          '</ul><div class="meetings-box"><h4>Meetings</h4><div class="meetings-scroll"></div></div>';

        const scroll = art.querySelector(".meetings-scroll");
        const sortOrder = document.getElementById("sort-date").value || "asc";
        if (!rowsAll.length) {
          const empty = document.createElement("div");
          empty.className = "empty-mini";
          empty.textContent = q ? "No matches in this bucket." : "No meetings classified here.";
          scroll.appendChild(empty);
        } else {
          const sorted = sortRowsByPreference(rowsAll, sortOrder);
          for (const r of sorted) {
            const urgent = isUrgentMeeting(r);
            const lowSent = isLowSentiment(r);
            const alerts = urgent || lowSent;
            const src = (r.classification_source || "").toLowerCase();
            const badge = src === "llm" ? "llm" : "rules";
            const row = document.createElement("div");
            row.className = "meeting" + (alerts ? " meeting--alert" : "");

            let signals = "";
            if (urgent) {
              signals += '<span class="tag tag-red">Urgent</span>';
            }
            if (lowSent) {
              signals += '<span class="tag tag-red">Sentiment below 3</span>';
            }

            const idVal = escapeHtml(String(r.conversation_id || "—"));
            const catLbl = escapeHtml(String(r.category_label || LABELS[r.category] || "—"));
            const stake = escapeHtml(String(r.stakeholder_primary || "—"));
            const dur =
              r.duration_min !== null && r.duration_min !== undefined && r.duration_min !== ""
                ? escapeHtml(String(r.duration_min)) + " min"
                : "—";

            row.innerHTML =
              (signals
                ? '<div class="signal-row">' + signals + "</div>"
                : "") +
              '<div class="meeting-title">' + escapeHtml(String(r.title || "")) + "</div>" +
              '<div class="meeting-meta">' +
              '<span class="badge ' +
              badge +
              '">' +
              escapeHtml(String(r.classification_source || "")) +
              '</span><span aria-hidden="true">·</span><span>' +
              escapeHtml(String(r.start_date || "—")) +
              '</span><span aria-hidden="true">·</span><span>' +
              escapeHtml(String(r.category || "")) +
              '</span></div>' +
              "<dl class=\\"meeting-details\\">" +
              "<dt>Conversation ID</dt><dd>" +
              idVal +
              "</dd>" +
              "<dt>Primary category</dt><dd>" +
              catLbl +
              "</dd>" +
              "<dt>Primary stakeholder</dt><dd>" +
              stake +
              "</dd>" +
              "<dt>Duration</dt><dd>" +
              dur +
              "</dd>" +
              "<dt>Sentiment score</dt><dd>" +
              (parseSentiment(r) !== null
                ? escapeHtml(String(r.sentiment_score))
                : "—") +
              "</dd>" +
              "</dl>";
            const cidRaw = String(r.conversation_id || "").trim();
            const actWrap = document.createElement("div");
            actWrap.className = "meet-actions";
            const btnEx = document.createElement("button");
            btnEx.type = "button";
            btnEx.className = "btn-explore";
            btnEx.textContent = "Participants, schedule, summary, transcript & LLM";
            if (!cidRaw) btnEx.disabled = true;
            const flyoutEl = document.createElement("div");
            flyoutEl.className = "meet-flyout";
            flyoutEl.hidden = true;
            const mountEl = document.createElement("div");
            mountEl.setAttribute("data-explore-mount", "");
            flyoutEl.appendChild(mountEl);
            btnEx.addEventListener("click", function () {
              if (!cidRaw) return;
              flyoutEl.hidden = !flyoutEl.hidden;
              btnEx.classList.toggle("btn-explore-active", !flyoutEl.hidden);
              if (!flyoutEl.hidden) hydrateFlyout(cidRaw, flyoutEl);
            });
            actWrap.appendChild(btnEx);
            row.appendChild(actWrap);
            row.appendChild(flyoutEl);
            scroll.appendChild(row);
          }
        }
        grid.appendChild(art);
        cardIndex++;
      }
    }

    function escapeHtml(s) {
      const d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }

    document.getElementById("filter").addEventListener("input", () => {
      const sh = payload.stakeholders.find((x) => x.key === activeKey);
      if (sh) renderCategoryBoard(sh);
    });

    document.getElementById("sort-date").addEventListener("change", () => {
      const sh = payload.stakeholders.find((x) => x.key === activeKey);
      if (sh) renderCategoryBoard(sh);
    });

    renderSubcategoryMatrix();
    load();
  </script>
</body>
</html>"""


# ========= START OF dashboard_html =========
def dashboard_html() -> str:
    """
    Produce the SPA shell HTML with embedded category labels for client-side legends.

    Expected input: none (uses CATEGORY_LABELS and DASHBOARD_HTML template).

    Output: UTF-8 HTML string safe to return as HTMLResponse.
    """
    labels = json.dumps(CATEGORY_LABELS, ensure_ascii=False)
    themes = json.dumps(CATEGORY_SUB_THEMES, ensure_ascii=False)
    return (
        DASHBOARD_HTML.replace("__CATEGORY_LABELS_JSON__", labels).replace(
            "__SUB_THEMES_JSON__",
            themes,
        )
    )


# ========= END OF dashboard_html =========


_MERMAID_BLOCK = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    re.DOTALL,
)


def _promote_mermaid_blocks(html: str) -> str:
    """Turn fenced ```mermaid blocks into div.mermaid for Mermaid.js."""

    def _repl(m: re.Match[str]) -> str:
        return f'<div class="mermaid">\n{m.group(1).strip()}\n</div>'

    return _MERMAID_BLOCK.sub(_repl, html)


ARCHITECTURE_PAGE_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Architecture — Transcript Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400..800;1,600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --text: #f4f1ff;
      --muted: #9d94b8;
      --accent: #5eead4;
      --accent-2: #f0abfc;
      --border: rgba(167, 139, 250, 0.24);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      background-color: #08070f;
      background-image:
        radial-gradient(ellipse 100% 80% at 0% -20%, rgba(139, 92, 246, 0.34), transparent 55%),
        radial-gradient(ellipse 90% 55% at 100% 8%, rgba(20, 184, 166, 0.18), transparent 52%),
        radial-gradient(ellipse 70% 45% at 85% 100%, rgba(251, 113, 133, 0.11), transparent 42%),
        linear-gradient(180deg, #0a0912 0%, #14122a 38%, #0b0a12 100%);
      color: var(--text);
      line-height: 1.62;
      min-height: 100vh;
    }
    .doc-top {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      width: 100%;
      padding: 0.9rem clamp(1rem, 3vw, 2.5rem);
      background: rgba(8, 7, 15, 0.9);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid rgba(94, 234, 212, 0.22);
    }
    .doc-top a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      font-size: 0.9rem;
    }
    .doc-top a:hover { color: var(--accent-2); text-decoration: underline; }
    .doc-top span {
      font-size: 0.78rem;
      color: var(--muted);
      font-weight: 500;
    }
    .doc-wrap {
      width: 100%;
      max-width: none;
      margin: 0;
      padding: clamp(1rem, 1.8vw, 1.75rem) clamp(1rem, 4vw, 3.5rem) clamp(2.5rem, 6vw, 5rem);
    }
    .doc-article {
      width: 100%;
      max-width: none;
      background: transparent;
      border: none;
      border-radius: 0;
      padding: 0;
      box-shadow: none;
    }
    .doc-article h1 {
      font-size: clamp(1.55rem, 3.2vw, 2.25rem);
      margin: 0 0 1rem;
      letter-spacing: -0.03em;
      line-height: 1.15;
      background: linear-gradient(105deg, #fdf4ff 0%, #a5f3fc 42%, #c4b5fd 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }
    .doc-article h2 {
      font-size: clamp(1.05rem, 1.9vw, 1.35rem);
      margin: 2.35rem 0 0.85rem;
      padding: 0.5rem 0 0.6rem 1rem;
      border-left: 4px solid #a78bfa;
      border-bottom: 1px solid var(--border);
      color: #ede9fe;
    }
    .doc-article h2:first-of-type { margin-top: 0; }
    .doc-article h2:nth-of-type(4n+2) {
      border-left-color: #2dd4bf;
      color: #ccfbf1;
    }
    .doc-article h2:nth-of-type(4n+3) {
      border-left-color: #fbbf24;
      color: #fef9c3;
    }
    .doc-article h2:nth-of-type(4n+4) {
      border-left-color: #fb7185;
      color: #ffe4e6;
    }
    .doc-article h3 {
      font-size: clamp(0.98rem, 1.4vw, 1.08rem);
      margin: 1.45rem 0 0.55rem;
      color: #7ee8d8;
    }
    .doc-article p {
      margin: 0.65rem 0;
      color: #dcd7ec;
      font-size: clamp(0.88rem, 1.2vw, 0.98rem);
    }
    .doc-article a { color: #7dd3fc; font-weight: 500; }
    .doc-article a:hover { color: #f0abfc; }
    .doc-article hr {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.45), transparent);
      margin: 2.25rem 0;
    }
    .doc-article ul, .doc-article ol {
      margin: 0.6rem 0 0.6rem 1.25rem;
      color: #dcd7ec;
      font-size: clamp(0.88rem, 1.2vw, 0.98rem);
    }
    .doc-article li { margin: 0.35rem 0; }
    .doc-article strong { color: #fff; }
    .doc-article code {
      font-size: 0.88em;
      padding: 0.12rem 0.4rem;
      border-radius: 6px;
      background: rgba(30, 27, 55, 0.95);
      border: 1px solid rgba(94, 234, 212, 0.22);
      color: #a5f3fc;
    }
    .doc-article pre:not(.mermaid) {
      overflow-x: auto;
      padding: 1rem 1.15rem;
      border-radius: 12px;
      background: rgba(12, 10, 22, 0.92);
      border: 1px solid rgba(167, 139, 250, 0.28);
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .doc-article pre code { background: none; border: none; padding: 0; font-size: inherit; color: #d8d4e8; }
    .doc-article table {
      width: 100%;
      border-collapse: collapse;
      font-size: clamp(0.8rem, 1.1vw, 0.9rem);
      margin: 1.15rem 0;
    }
    .doc-article th, .doc-article td {
      border: 1px solid var(--border);
      padding: 0.55rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }
    .doc-article thead th {
      background: linear-gradient(180deg, rgba(139, 92, 246, 0.38) 0%, rgba(45, 212, 191, 0.16) 100%);
      color: #faf5ff;
      font-weight: 650;
    }
    .doc-article tbody tr:nth-child(even) td { background: rgba(255, 255, 255, 0.025); }
    .doc-article td { color: #e8e4f6; }
    .doc-article blockquote {
      margin: 0.9rem 0;
      padding: 0.65rem 1rem;
      border-left: 4px solid #f0abfc;
      background: rgba(139, 92, 246, 0.1);
      color: #c4bcd9;
      font-size: 0.92rem;
      border-radius: 0 10px 10px 0;
    }
    .mermaid {
      margin: 1.5rem 0;
      padding: 1.15rem 1.25rem;
      border-radius: 14px;
      background: rgba(18, 16, 35, 0.78);
      border: 1px solid rgba(45, 212, 191, 0.24);
      overflow-x: auto;
    }
    .md-missing {
      color: #fde047;
      font-size: 0.9rem;
      margin-bottom: 1rem;
    }
    .md-fallback {
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.78rem;
      line-height: 1.45;
      color: #dcd7ec;
      padding: 1rem;
      border-radius: 12px;
      background: rgba(12, 10, 22, 0.92);
      border: 1px solid rgba(167, 139, 250, 0.28);
      overflow-x: auto;
    }
    .doc-article .kw-risk {
      color: #fecaca;
      font-weight: 700;
      background: rgba(248, 113, 113, 0.2);
      border: 1px solid rgba(248, 113, 113, 0.45);
      padding: 0.08rem 0.28rem;
      border-radius: 4px;
      white-space: nowrap;
    }
    .doc-article .kw-cadence {
      color: #fef08a;
      font-weight: 650;
      background: rgba(250, 204, 21, 0.12);
      border: 1px solid rgba(250, 204, 21, 0.38);
      padding: 0.08rem 0.28rem;
      border-radius: 4px;
      white-space: nowrap;
    }
  </style>
</head>
<body>
  <header class="doc-top">
    <a href="/">← Dashboard</a>
    <span>docs/ARCHITECTURE.md · live preview</span>
  </header>
  <div class="doc-wrap">
    <article class="doc-article">
__CONTENT__
    </article>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
    mermaid.run({ querySelector: ".mermaid" });
  </script>
</body>
</html>"""


def architecture_html_page() -> str:
    """
    Serve docs/ARCHITECTURE.html when present (pre-built native HTML + Mermaid).

    Fallback: render docs/ARCHITECTURE.md with the markdown package into the same shell.

    Expected input: ARCHITECTURE_HTML and/or ARCHITECTURE_MD on disk.

    Output: UTF-8 HTML document string.
    """
    if ARCHITECTURE_HTML.is_file():
        return ARCHITECTURE_HTML.read_text(encoding="utf-8")
    if not ARCHITECTURE_MD.is_file():
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Architecture</title></head>"
            "<body style='font-family:system-ui;padding:2rem;background:#0f172a;color:#e2e8f0;'>"
            "<p>Missing <code>docs/ARCHITECTURE.html</code> and <code>docs/ARCHITECTURE.md</code>. "
            "Run <code>python3 scripts/build_architecture_html.py</code> after <code>pip install markdown</code>.</p>"
            "<p><a href='/' style='color:#38bdf8'>← Dashboard</a></p></body></html>"
        )
    md_text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    if markdown_lib is None:
        inner = (
            '<p class="md-missing">Install <code>markdown</code> or run <code>python3 scripts/build_architecture_html.py</code> '
            "to create <code>docs/ARCHITECTURE.html</code>.</p>"
            '<pre class="md-fallback">' + html.escape(md_text) + "</pre>"
        )
    else:
        inner = markdown_lib.markdown(
            md_text,
            extensions=["fenced_code", "tables", "extra"],
        )
        inner = _promote_mermaid_blocks(inner)
    return ARCHITECTURE_PAGE_SHELL.replace("__CONTENT__", inner)


@app.get("/")
def get_dashboard() -> HTMLResponse:
    """
    Mandatory route: dashboard UI that calls GET /api/dashboard for data.

    Expected input: none.

    Output: HTMLResponse containing the stakeholder dashboard single-page shell.
    """
    return HTMLResponse(content=dashboard_html(), status_code=200)


@app.get("/architecture")
def get_architecture_doc() -> HTMLResponse:
    """
    Serve docs/ARCHITECTURE.html (preferred) or render docs/ARCHITECTURE.md at runtime.

    Expected input: none (reads ARCHITECTURE_HTML or ARCHITECTURE_MD from disk).

    Output: HTMLResponse with Mermaid diagrams when diagrams are present.
    """
    return HTMLResponse(content=architecture_html_page(), status_code=200)


@app.get("/api/dashboard")
def get_dashboard_json() -> dict[str, Any]:
    """
    Mandatory route: aggregated meetings and stakeholder summaries for the UI.

    Expected input: none (reads CLASSIFICATION_CSV from disk).

    Output: JSON-serializable dict for the frontend (rows, stakeholder stats, totals).
    """
    return build_dashboard_payload()


@app.get("/api/conversation/{conversation_id}")
def get_conversation_detail(conversation_id: str) -> dict[str, Any]:
    """
    Load participant metadata, join/leave timeline, saved summary, transcript text from disk.
    """
    cid = validate_conversation_id(conversation_id)
    return load_conversation_detail(cid)


@app.post("/api/conversation/{conversation_id}/llm-summarize")
def post_conversation_llm_summary(conversation_id: str) -> dict[str, Any]:
    """
    Full-transcript LLM summary (uses ModelSingleton / Groq via llm_model_config).
    """
    cid = validate_conversation_id(conversation_id)
    bundle = load_conversation_detail(cid)
    md = summarize_transcript_via_llm(conversation_id=cid, detail=bundle)
    return {"conversation_id": cid, "markdown": md}
