"""
FastAPI dashboard for Quorum.

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

app = FastAPI(title="Quorum", docs_url=None, redoc_url=None)


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
  <title>Quorum — Meeting desk</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,520;9..144,650&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --paper: #f3ecdf;
      --sheet: #faf6ee;
      --ink: #1c1612;
      --muted: #5c5148;
      --hint: #7a6e63;
      --rule: #d8ccba;
      --copper: #8d3b24;
      --moss: #2c4638;
      --danger: #8f2d2d;
      --danger-bg: #f8ebe6;
      --danger-border: #e4c2b6;
      --amber: #8a5a12;
      --accent: #8d3b24;
      --accent-bright: #8d3b24;
      --text: #1c1612;
      --border: #d8ccba;
      --border-strong: #c4b49f;
      --radius-lg: 0;
      --radius-md: 0;
      --radius-sm: 0;
      --font: "Figtree", "Segoe UI", sans-serif;
      --display: "Fraunces", Georgia, serif;
      --font-mono: "IBM Plex Mono", ui-monospace, monospace;
    }
    *, *::before, *::after { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
    body.ti-app {
      margin: 0;
      font-family: var(--font);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.35), transparent 220px),
        var(--paper);
      color: var(--ink);
      line-height: 1.5;
      min-height: 100vh;
    }
    ::selection { background: #ead4bf; color: var(--ink); }
    .skip-link {
      position: absolute; left: 1rem; top: -3rem; background: var(--ink); color: var(--sheet);
      padding: 0.4rem 0.7rem; z-index: 50; font-weight: 650;
    }
    .skip-link:focus { top: 0.6rem; }
    .ti-shell { min-height: 100vh; }
    header.ti-header {
      position: sticky; top: 0; z-index: 30;
      background: rgba(243, 236, 223, 0.94);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--ink);
    }
    .ti-header-inner {
      max-width: 1240px;
      margin: 0 auto;
      padding: 0.7rem 1.25rem 0.85rem;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 1rem 1.5rem;
      align-items: end;
    }
    .brand { display: flex; align-items: baseline; gap: 0.75rem; }
    .brand-mark {
      font-family: var(--display);
      font-weight: 650;
      font-size: 0.95rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--copper);
    }
    header.ti-header h1 {
      margin: 0;
      font-family: var(--display);
      font-weight: 560;
      font-size: clamp(2rem, 4vw, 2.7rem);
      letter-spacing: -0.03em;
      line-height: 0.9;
    }
    header.ti-header p {
      margin: 0.35rem 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      max-width: 28rem;
    }
    .mast-copy { min-width: 0; }
    .side-nav, .ti-header-actions {
      display: flex; align-items: center; gap: 0.4rem; justify-content: flex-end;
    }
    .side-link, .arch-doc-link {
      color: var(--ink);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 650;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.35rem 0.15rem;
      border-bottom: 1px solid transparent;
    }
    .side-link.is-active, .arch-doc-link:hover, .side-link:hover { border-bottom-color: var(--copper); color: var(--copper); }
    .ti-header-tag {
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--hint);
      white-space: nowrap;
    }
    main.ti-main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 1.25rem 1.25rem 2.5rem;
    }
    .ti-footer {
      max-width: 1240px;
      margin: 0 auto;
      padding: 0.8rem 1.25rem 1.4rem;
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
      border-top: 1px solid var(--ink);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--hint);
    }
    #status.empty.status-card {
      max-width: 26rem;
      margin: 3rem auto;
      padding: 1.5rem 0;
      text-align: center;
      color: var(--muted);
      border-top: 1px solid var(--ink);
      border-bottom: 1px solid var(--ink);
      background: transparent;
    }
    .ti-app-shell { display: flex; flex-direction: column; gap: 0.25rem; }
    .glass-panel {
      background: transparent;
      border: none;
      box-shadow: none;
      border-radius: 0;
      padding: 1rem 0 0.4rem;
    }
    .section-label h3 {
      margin: 0;
      font-family: var(--display);
      font-weight: 560;
      font-size: 1.35rem;
    }
    .section-label p { margin: 0.2rem 0 0.75rem; color: var(--muted); font-size: 0.9rem; }
    .subcat-matrix {
      display: grid;
      grid-template-columns: repeat(9, minmax(0, 1fr));
      gap: 0;
      border-top: 1px solid var(--ink);
      border-bottom: 1px solid var(--ink);
    }
    .subcat-col {
      min-width: 0;
      padding: 0.55rem 0.45rem 0.7rem;
      border-right: 1px solid var(--rule);
      background: transparent;
      border-top: none;
      border-left: none;
      border-bottom: none;
      border-radius: 0;
    }
    .subcat-col:last-child { border-right: none; }
    .subcat-col-hd {
      font-family: var(--display);
      font-size: 1.25rem;
      text-align: left;
      color: var(--copper);
      font-weight: 560;
      padding-bottom: 0.35rem;
    }
    .subcat-col.ac1 .subcat-col-hd { color: var(--moss); }
    .subcat-col.ac2 .subcat-col-hd { color: #6b4b16; }
    .subcat-col-chips { display: flex; flex-direction: column; gap: 0.35rem; }
    .subcat-chip {
      font-size: 0.68rem;
      line-height: 1.35;
      color: var(--muted);
      padding: 0;
      background: none;
      border: none;
      border-radius: 0;
    }
    code {
      font-family: var(--font-mono);
      font-size: 0.84em;
      background: #efe4d4;
      padding: 0.05rem 0.3rem;
      border-radius: 0;
      border: none;
    }
    .tabs { display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; margin: 0; padding: 0; }
    .tabs.track {
      width: 100%;
      background: transparent;
      border: none;
      border-radius: 0;
      padding: 0.2rem 0 0;
      border-bottom: 1px solid var(--ink);
    }
    .tabs button {
      border: none;
      border-bottom: 2px solid transparent;
      background: transparent;
      color: var(--muted);
      padding: 0.55rem 0.1rem 0.65rem;
      border-radius: 0;
      cursor: pointer;
      font-family: var(--font);
      font-size: 0.95rem;
      font-weight: 650;
      margin-bottom: -1px;
    }
    .tabs button:hover { color: var(--ink); }
    .tabs button.active { color: var(--ink); border-bottom-color: var(--copper); }
    .tabs button:focus-visible, .search:focus, .sort-select:focus, .btn-explore:focus-visible, .btn-llm:focus-visible, .q-chip:focus-visible {
      outline: 2px solid var(--copper);
      outline-offset: 2px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      margin: 0;
      border-top: 1px solid var(--ink);
      border-bottom: 1px solid var(--ink);
    }
    .meta-pill {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
      padding: 0.85rem 1rem;
      background: transparent;
      border: none;
      border-right: 1px solid var(--rule);
      border-radius: 0;
      box-shadow: none;
      font-family: var(--display);
      font-size: 1.55rem;
      font-weight: 560;
      letter-spacing: -0.03em;
      color: var(--ink);
      word-break: break-word;
    }
    .meta-pill:last-child { border-right: none; }
    .meta-pill strong {
      font-family: var(--font);
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--hint);
    }
    .board-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem 1.5rem;
      align-items: flex-end;
      justify-content: space-between;
      padding-bottom: 0.35rem;
    }
    .board-toolbar label {
      display: block;
      margin-bottom: 0.3rem;
      font-size: 0.68rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--hint);
      font-weight: 700;
    }
    .toolbar-group { min-width: 0; }
    .search, .sort-select {
      font-family: var(--font);
      font-size: 0.95rem;
      color: var(--ink);
      background: transparent;
      border: none;
      border-bottom: 1px solid var(--ink);
      border-radius: 0;
      padding: 0.45rem 0.1rem;
      width: 100%;
    }
    .search { min-width: min(100%, 280px); }
    .sort-select { min-width: 240px; cursor: pointer; background: var(--paper); }
    .search::placeholder { color: #a89886; }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
      gap: 1.25rem;
      padding-top: 0.4rem;
    }
    .cat-card {
      background: var(--sheet);
      border: 1px solid var(--ink);
      border-radius: 0;
      box-shadow: 7px 7px 0 #e6d8c6;
      display: flex;
      flex-direction: column;
      min-height: 260px;
      overflow: hidden;
    }
    .cat-card:hover { transform: translate(-2px, -2px); box-shadow: 9px 9px 0 #e6d8c6; }
    .cat-card.ac0, .cat-card.ac1, .cat-card.ac2 { border-top: 1px solid var(--ink); }
    .cat-card-head { padding: 0.9rem 1rem 0.75rem; border-bottom: 1px solid var(--rule); }
    .cat-card-head-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.6rem; }
    .letter-mark {
      font-family: var(--display);
      font-size: 2.6rem;
      font-weight: 520;
      line-height: 0.85;
      letter-spacing: -0.04em;
      color: var(--copper);
      background: none;
      width: auto;
      height: auto;
    }
    .cat-card.ac1 .letter-mark { color: var(--moss); }
    .cat-card.ac2 .letter-mark { color: #6b4b16; }
    .count-pill {
      font-size: 0.72rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      border: none;
      background: none;
      padding: 0.2rem 0;
      font-weight: 650;
    }
    .cat-title { margin: 0.55rem 0 0; font-size: 1rem; font-weight: 650; line-height: 1.35; }
    .sub-list {
      margin: 0;
      padding: 0.7rem 1rem;
      list-style: none;
      font-size: 0.8rem;
      color: var(--muted);
      border-bottom: 1px solid var(--rule);
      background: transparent;
    }
    .sub-list li { padding: 0.12rem 0 0.12rem 0.8rem; position: relative; }
    .sub-list li::before {
      content: "";
      position: absolute;
      left: 0; top: 0.55em;
      width: 5px; height: 1px;
      background: var(--copper);
      border-radius: 0;
      box-shadow: none;
    }
    .meetings-box h4 {
      margin: 0;
      padding: 0.65rem 1rem 0.25rem;
      font-size: 0.68rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--hint);
    }
    .meetings-scroll { max-height: 460px; overflow-y: auto; padding: 0 0.65rem 0.75rem; }
    .meeting {
      margin: 0.45rem 0.2rem;
      padding: 0.7rem 0.15rem 0.75rem;
      border: none;
      border-bottom: 1px solid var(--rule);
      border-radius: 0;
      background: transparent;
    }
    .meeting:hover { background: rgba(141, 59, 36, 0.04); }
    .meeting.meeting--alert { background: var(--danger-bg); border-bottom-color: var(--danger-border); padding-left: 0.45rem; padding-right: 0.45rem; }
    .signal-row { display: flex; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 0.35rem; }
    .tag, .badge {
      font-size: 0.66rem;
      font-weight: 750;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.12rem 0.35rem;
      border-radius: 0;
    }
    .tag-red { color: var(--danger); background: transparent; border: 1px solid var(--danger); }
    .meeting-title { font-family: var(--display); font-weight: 560; font-size: 1.05rem; line-height: 1.3; margin-bottom: 0.3rem; }
    .meeting-meta { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; color: var(--hint); font-size: 0.78rem; }
    .badge.rules { color: var(--moss); border: 1px solid var(--moss); background: transparent; }
    .badge.llm { color: var(--amber); border: 1px solid var(--amber); background: transparent; }
    .meeting-details {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.22rem 0.75rem;
      margin-top: 0.45rem;
      font-size: 0.78rem;
    }
    .meeting-details dt { color: var(--hint); font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; }
    .meeting-details dd { margin: 0; color: var(--ink); word-break: break-word; }
    .empty { text-align: center; color: var(--muted); }
    .empty-mini {
      margin: 0.6rem 0.2rem;
      padding: 1rem 0;
      text-align: center;
      color: var(--hint);
      border-top: 1px dashed var(--rule);
      border-bottom: 1px dashed var(--rule);
      font-size: 0.86rem;
    }
    .meet-actions { margin-top: 0.55rem; }
    .btn-explore, .btn-llm, .q-chip { font-family: inherit; cursor: pointer; }
    .btn-explore {
      background: var(--ink);
      color: var(--sheet);
      border: 1px solid var(--ink);
      border-radius: 0;
      padding: 0.42rem 0.7rem;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .btn-explore:hover { background: var(--copper); border-color: var(--copper); }
    .btn-explore:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-explore-active { background: var(--moss); border-color: var(--moss); }
    .meet-flyout {
      margin-top: 0.7rem;
      padding: 0.8rem 0.15rem 0.2rem;
      border-top: 1px solid var(--ink);
      background: transparent;
      border-radius: 0;
      box-shadow: none;
    }
    .fly-hint { margin: 0 0 0.6rem; color: var(--muted); font-size: 0.8rem; }
    .fly-hint.is-loading { color: var(--copper); }
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem 0.7rem; margin-bottom: 0.7rem; }
    .q-chip {
      background: none;
      border: none;
      border-bottom: 1px solid transparent;
      border-radius: 0;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 0.2rem 0;
    }
    .q-chip.on, .q-chip:hover { color: var(--ink); border-bottom-color: var(--copper); }
    .fly-section { display: none; }
    .fly-section.vis { display: block; }
    .fly-section h5 {
      margin: 0 0 0.45rem;
      font-size: 0.68rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--hint);
    }
    .kv-grid {
      display: grid;
      grid-template-columns: minmax(8rem, 32%) 1fr;
      gap: 0.35rem 0.8rem;
      font-size: 0.86rem;
      margin-bottom: 0.6rem;
    }
    .kv-key { color: var(--hint); font-weight: 700; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
    .kv-val { color: var(--ink); word-break: break-word; }
    .party-list-wrap { grid-column: 1 / -1; }
    .name-tag {
      display: inline-block;
      margin: 0.2rem 0.4rem 0.2rem 0;
      padding: 0.08rem 0;
      border-bottom: 1px solid var(--rule);
      color: var(--ink);
      font-size: 0.82rem;
      background: none;
      border-radius: 0;
    }
    .join-block {
      display: grid;
      grid-template-columns: minmax(7rem, 30%) 1fr;
      gap: 0.25rem 0.7rem;
      padding: 0.45rem 0;
      border-bottom: 1px solid var(--rule);
      background: none;
      border-left: none;
      border-right: none;
      border-top: none;
      border-radius: 0;
    }
    .jn-name { font-family: var(--display); color: var(--moss); font-weight: 560; }
    .jn-lbl { color: var(--hint); font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase; margin-right: 0.3rem; }
    .jn-sub { margin-right: 0.8rem; }
    .summary-prose {
      margin: 0 0 0.75rem;
      padding: 0.2rem 0 0.2rem 0.8rem;
      border-left: 2px solid var(--moss);
      background: none;
      color: #24362c;
      border-radius: 0;
    }
    .transcript-stream {
      max-height: 380px;
      overflow: auto;
      border-top: 1px solid var(--rule);
      padding-top: 0.5rem;
      background: none;
    }
    .transcript-legend { display: flex; flex-wrap: wrap; gap: 0.35rem 0.6rem; margin-bottom: 0.55rem; font-size: 0.75rem; color: var(--muted); }
    .legend-chip { display: inline-flex; align-items: center; gap: 0.3rem; }
    .legend-swatch { width: 8px; height: 8px; border-radius: 0; }
    .t-turn { margin: 0 0 0.55rem; padding: 0.35rem 0 0.35rem 0.65rem; border-left: 2px solid var(--rule); background: none !important; border-radius: 0; }
    .t-turn-is-host { border-left-color: var(--amber); }
    .t-sp0 { border-left-color: #8d3b24; }
    .t-sp1 { border-left-color: #2c4638; }
    .t-sp2 { border-left-color: #6b4b16; }
    .t-sp3 { border-left-color: #3f4d73; }
    .t-sp4 { border-left-color: #7a3e57; }
    .t-sp5 { border-left-color: #3d5c4a; }
    .t-sp6 { border-left-color: #8a5a12; }
    .t-sp7 { border-left-color: #6e2e2e; }
    .t-sp0 .legend-swatch { background: #8d3b24; }
    .t-sp1 .legend-swatch { background: #2c4638; }
    .t-sp2 .legend-swatch { background: #6b4b16; }
    .t-sp3 .legend-swatch { background: #3f4d73; }
    .t-sp4 .legend-swatch { background: #7a3e57; }
    .t-sp5 .legend-swatch { background: #3d5c4a; }
    .t-sp6 .legend-swatch { background: #8a5a12; }
    .t-sp7 .legend-swatch { background: #6e2e2e; }
    .t-speaker-line { display: flex; gap: 0.4rem; align-items: baseline; margin-bottom: 0.15rem; }
    .t-role-badge { font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--hint); font-weight: 750; }
    .t-speaker-name { font-family: var(--display); font-weight: 560; color: var(--ink); }
    .t-turn-text { color: #2a241e; white-space: pre-wrap; word-break: break-word; }
    .btn-llm {
      margin-top: 0.4rem;
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--ink);
      border-radius: 0;
      padding: 0.42rem 0.7rem;
      font-size: 0.75rem;
      font-weight: 750;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .btn-llm:hover:not(:disabled) { background: var(--ink); color: var(--sheet); }
    .btn-llm:disabled { opacity: 0.45; cursor: not-allowed; }
    .llm-out, .trans-plain {
      margin-top: 0.55rem;
      padding: 0.7rem;
      background: #fffdf8;
      border: 1px solid var(--rule);
      color: var(--ink);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      white-space: pre-wrap;
      max-height: 340px;
      overflow: auto;
      border-radius: 0;
    }
    .fly-err { color: var(--danger); font-weight: 700; }
    .footnote { margin: 0.8rem 0 0; color: var(--muted); font-size: 0.85rem; font-style: italic; font-family: var(--display); }
    @media (max-width: 1100px) {
      .subcat-matrix { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .subcat-col { border-bottom: 1px solid var(--rule); }
      .meta { grid-template-columns: 1fr 1fr; }
      .meta-pill { border-bottom: 1px solid var(--rule); }
    }
    @media (max-width: 760px) {
      .ti-header-inner { grid-template-columns: 1fr; align-items: start; }
      .side-nav, .ti-header-actions { justify-content: flex-start; }
      header.ti-header h1 { font-size: 2.1rem; }
      .subcat-matrix, .meta { grid-template-columns: 1fr 1fr; }
      .kv-grid, .join-block { grid-template-columns: 1fr; }
      .sort-select, .search { min-width: 0; }
      .tabs.track { flex-wrap: nowrap; overflow-x: auto; }
      .tabs button { white-space: nowrap; }
    }

  </style>
</head>
<body class="ti-app">
  <a class="skip-link" href="#main">Skip to content</a>
  <div class="ti-shell">
    <header class="ti-header">
      <div class="ti-header-inner">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">Desk</span>
          <h1>Quorum</h1>
        </div>
        <div class="mast-copy">
          <p>A reading room for classified meetings. Same records, sorted for each audience.</p>
        </div>
        <nav class="side-nav" aria-label="Primary">
          <a class="side-link is-active" href="/" aria-current="page">Desk</a>
          <a href="/architecture" class="side-link arch-doc-link" title="Classification rules, LLM usage, data fields">Architecture</a>
          <span class="ti-header-tag" title="Unified meeting taxonomy">A–I</span>
        </nav>
      </div>
    </header>
    <main class="ti-main" id="main">
      <div id="status" class="empty status-card" role="status" aria-live="polite">Opening the desk…</div>
      <div id="app" class="ti-app-shell" style="display:none;">
        <section class="glass-panel glass-panel--dense glass-panel--subcats" aria-label="All taxonomy sub-themes A1 through I3 in columns A to I">
          <div class="section-label">
            <h3>The nine letters</h3>
            <p>Sub-themes sit under each letter. A meeting is filed to the letter only.</p>
          </div>
          <div id="subcat-matrix"></div>
        </section>
        <section class="glass-panel glass-panel--dense section-kpis">
          <div class="meta" id="global-meta"></div>
        </section>
        <section class="glass-panel glass-panel--dense">
          <nav class="tabs track" aria-label="Stakeholder views" id="tabs"></nav>
        </section>
        <section class="glass-panel">
          <div class="board-toolbar">
            <div class="toolbar-group" style="flex:1;">
              <label for="filter">Find a meeting</label>
              <input type="search" class="search" id="filter" placeholder="Title, conversation ID, or category…" autocomplete="off" />
            </div>
            <div class="toolbar-group">
              <label for="sort-date">Order by date</label>
              <select class="sort-select" id="sort-date" aria-label="Sort meetings by date">
                <option value="asc">Oldest first</option>
                <option value="desc">Newest first</option>
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
    <footer class="ti-footer">
      <span>Quorum</span>
      <span>Rules first · brief on demand</span>
    </footer>
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
        '<p class="fly-hint is-loading">Reading speaker-meta.json, events.json, meeting-info.json, summary.json, transcript.json…</p>';
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
  <title>Quorum — Architecture</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,520;9..144,650&display=swap" rel="stylesheet" />
  <style>
    :root {
      --text: #1c1612;
      --muted: #6d6258;
      --accent: #8d3b24;
      --accent-2: #2c4638;
      --border: #d8ccba;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: "Figtree", "Segoe UI", sans-serif;
      background: #f3ecdf;
      color: var(--text);
      line-height: 1.65;
      min-height: 100vh;
    }
    .doc-top {
      position: sticky; top: 0; z-index: 20;
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.75rem;
      padding: 0.85rem clamp(1rem, 3vw, 2rem);
      background: #f3ecdf;
      color: var(--text);
      border-bottom: 1px solid #1c1612;
    }
    .doc-top a { color: #1c1612; text-decoration: none; font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: 1.35rem; }
    .doc-top a:hover { color: var(--accent); }
    .doc-top span { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
    .doc-wrap { width: min(860px, 100%); margin: 0 auto; padding: 1.5rem 1.15rem 3rem; }
    .doc-article { background: transparent; border: none; padding: 0; box-shadow: none; }
    .doc-article h1 { font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: clamp(1.8rem, 4vw, 2.5rem); letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 0.8rem; color: #1c1612; }
    .doc-article h2 { font-family: Fraunces, Georgia, serif; font-weight: 560; font-size: 1.35rem; margin: 2rem 0 0.6rem; padding-bottom: 0.35rem; border-bottom: 1px solid #1c1612; border-left: none; color: #1c1612; }
    .doc-article h3 { font-size: 1.02rem; margin: 1.3rem 0 0.35rem; color: var(--accent); }
    .doc-article h4 { margin: 1rem 0 0.3rem; color: var(--accent-2); }
    .doc-article p, .doc-article li { color: #3a322c; }
    .doc-article a { color: var(--accent); font-weight: 650; }
    .doc-article hr { border: none; border-top: 1px solid var(--border); margin: 1.6rem 0; }
    .doc-article strong { color: #1c1612; }
    .doc-article code { font-size: 0.86em; background: #efe4d4; padding: 0.05rem 0.3rem; border: none; color: #1c1612; }
    .doc-article blockquote { margin: 0.8rem 0; padding: 0.2rem 0 0.2rem 0.8rem; border-left: 2px solid var(--accent); background: none; color: #3a322c; }
    .doc-article table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 1rem 0; }
    .doc-article th, .doc-article td { border: 1px solid var(--border); padding: 0.5rem 0.6rem; text-align: left; vertical-align: top; }
    .doc-article thead th { background: #efe4d4; color: #1c1612; }
    .doc-article tbody tr:nth-child(even) td { background: rgba(255,255,255,0.35); }
    .doc-article td { color: #3a322c; }
    .doc-article pre:not(.mermaid) { overflow-x: auto; padding: 0.9rem 1rem; background: #1c1612; color: #f6efe4; border-radius: 0; font-size: 0.8rem; }
    .doc-article pre code { background: none; color: inherit; padding: 0; }
    .mermaid { margin: 1rem 0; padding: 0.8rem; background: #faf6ee; border: 1px solid var(--border); overflow-x: auto; }
    .kw-risk, .doc-article .kw-risk { color: #8f2d2d; font-weight: 750; border-bottom: 1px solid #8f2d2d; background: none; }
    .kw-cadence, .doc-article .kw-cadence { color: #8a5a12; font-weight: 700; border-bottom: 1px solid #8a5a12; background: none; }
    .md-missing { color: #8d3b24; }
    .md-fallback { white-space: pre-wrap; background: #1c1612; color: #f6efe4; padding: 1rem; overflow-x: auto; }
    .doc-footer { margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid #1c1612; color: var(--muted); font-size: 0.82rem; }
    @media (max-width: 720px) {
      .doc-article table { display: block; overflow-x: auto; }
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
    mermaid.initialize({ startOnLoad: false, theme: "neutral", securityLevel: "loose" });
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
