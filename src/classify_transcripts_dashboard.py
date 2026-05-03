"""
Single-file pipeline: rules-first taxonomy (categories A-I), optional LLM fallback,
stakeholder dashboards, CSV export, and a Plotly HTML bar-chart report.

Run from project root:
  python3 src/classify_transcripts_dashboard.py

Requirements: meeting-info.json and summary.json in each conversation folder under dataset/.
Uses llm_model_config.ModelSingleton only when no rule assigns a category.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Ensure sibling import for llm_model_config
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from agents import Agent, ModelSettings, Runner  # noqa: E402

from llm_call_logger import append_llm_call_log  # noqa: E402
from llm_model_config import ModelSingleton  # noqa: E402

try:
    import plotly.graph_objects as go  # noqa: E402
except ImportError:
    go = None  # type: ignore[misc, assignment]

HTML_ESCAPE = str.maketrans(
    {"&": "&amp;", '"': "&quot;", "<": "&lt;", ">": "&gt;"}
)


def _esc(s: object) -> str:
    return str(s).translate(HTML_ESCAPE)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
def get_project_paths() -> tuple[Path, Path, Path]:
    """
    Resolve dataset root (conversations), project root, and output folder.

    Expected input: none (uses script location).

    Output: (dataset_root, project_root, output_dir) Paths.
    """
    project_root = _SCRIPT_DIR.parent
    dataset_root = project_root / "dataset"
    output_dir = dataset_root / "classification_output"
    return dataset_root, project_root, output_dir


# -----------------------------------------------------------------------------
# Taxonomy labels (mirror of src/plan.yaml categories)
# -----------------------------------------------------------------------------
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

PRIMARY_STAKEHOLDER_BY_CATEGORY: dict[str, str] = {
    "A": "support_leaders",
    "B": "support_leaders",
    "C": "support_leaders",
    "D": "sales_managers",
    "E": "sales_managers",
    "F": "sales_managers",
    "G": "product_managers",
    "H": "engineering_leads",
    "I": "engineering_leads",
}

STAKEHOLDER_DASHBOARDS: dict[str, dict[str, Any]] = {
    "support_leaders": {
        "label": "Support Operations",
        "categories": ["A", "B", "C"],
    },
    "sales_managers": {
        "label": "Revenue & Growth",
        "categories": ["D", "E", "F"],
    },
    "product_managers": {
        "label": "Product Strategy",
        "categories": ["G", "E", "F"],
    },
    "engineering_leads": {
        "label": "Engineering Execution",
        "categories": ["H", "B", "I"],
    },
}


# ========= START OF is_external_call =========
def is_external_call(emails: Iterable[str]) -> bool:
    """
    Detect if any participant email is outside the Aegis tenant.

    Expected input: iterable of email strings from meeting-info allEmails.

    Output: True if any address is not under @aegiscloud.com (case-insensitive).

    Raises: ValueError if the iterable is empty.
    """
    email_list = [e.strip().lower() for e in emails if e]
    if not email_list:
        raise ValueError("No emails supplied for participant check.")
    for e in email_list:
        if not e.endswith("@aegiscloud.com"):
            return True
    return False


# ========= END OF is_external_call =========


# ========= START OF is_internal_call =========
def is_internal_call(emails: Iterable[str]) -> bool:
    """
    Detect if every participant maps to internal Aegis staff.

    Expected input: iterable of email strings from meeting-info allEmails.

    Output: True if non-empty list and each email ends with @aegiscloud.com.

    Raises: ValueError if the iterable is empty.
    """
    email_list = [e.strip().lower() for e in emails if e]
    if not email_list:
        raise ValueError("No emails supplied for participant check.")
    for e in email_list:
        if not e.endswith("@aegiscloud.com"):
            return False
    return True


# ========= END OF is_internal_call =========


# ========= START OF parse_meeting_day =========
def parse_meeting_day(start_time: str) -> Optional[date]:
    """
    Read calendar day from ISO startTime for window rules.

    Expected input: meeting-info.json startTime string (ISO8601).

    Output: date object or None if parsing fails.
    """
    if not start_time:
        return None
    try:
        return datetime.strptime(start_time[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ========= END OF parse_meeting_day =========


# ========= START OF get_sentiment =========
def get_sentiment(summary: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """
    Normalize sentiment numeric score and categorical label.

    Expected input: summary dict with optional sentimentScore, overallSentiment.

    Output: (sentimentScore or None, overallSentiment or None).

    Gives: Stable tuple for dashboards and downstream CSV formatting.
    """
    score_raw = summary.get("sentimentScore")
    sentiment = summary.get("overallSentiment")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (ValueError, TypeError):
        score = None
    if sentiment is None:
        return score, None
    return score, str(sentiment)


# ========= END OF get_sentiment =========


# ========= START OF merge_text =========
def merge_text(summary: Mapping[str, Any]) -> tuple[str, str]:
    """
    Build searchable strings from summary body and topics.

    Expected input: summary dict containing summary text and topics list.

    Output: (merged_lower_text, merged_raw_text for topics joined).

    Used by keyword rules across categories.
    """
    body = str(summary.get("summary") or "")
    topics = summary.get("topics") or []
    raw_topics = " ".join(str(t) for t in topics if t is not None)
    merged = body + " " + raw_topics
    return merged.lower(), raw_topics.lower()


# ========= END OF merge_text =========


# ========= START OF format_duration =========
def format_duration(meeting_info: Mapping[str, Any]) -> tuple[Optional[float], str]:
    """
    Read numeric duration in minutes plus display string.

    Expected input: meeting-info dict with optional duration float (minutes).

    Output: (float or None if missing/unparsable, string for CSV display).
    """
    raw = meeting_info.get("duration")
    try:
        val = float(raw)
        return val, str(val)
    except (ValueError, TypeError):
        return None, ""


# ========= END OF format_duration =========


# ========= START OF load_conversation_bundle =========
def load_conversation_bundle(conv_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load deterministic inputs for categorization rules.

    Expected input: path to conversation folder holding JSON assets.

    Output: (meeting_info dict, summary dict).

    Gives: Throws FileNotFoundError if required files absent.
    """
    mi_path = conv_dir / "meeting-info.json"
    su_path = conv_dir / "summary.json"
    with open(mi_path, encoding="utf-8") as f:
        meeting_info = json.load(f)
    with open(su_path, encoding="utf-8") as f:
        summary = json.load(f)
    return meeting_info, summary


# ========= END OF load_conversation_bundle =========


# ========= START OF discover_conversation_dirs =========
def discover_conversation_dirs(dataset_root: Path) -> list[Path]:
    """
    Find each conversation subdirectory that resembles a packaged meeting bundle.

    Expected input: dataset root Path.

    Output: sorted Paths where meeting-info.json exists (one folder per call).

    Excludes stray JSON manifests at dataset root itself.
    """
    out = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "meeting-info.json").is_file():
            out.append(child)
    return out


# ========= END OF discover_conversation_dirs =========


# -------- Category rules return True when the meeting clearly fits that lane.


# ========= START OF is_outage_internal =========
def is_outage_internal(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Internal war-room style incident response (category B before customer fallout).

    Expected input: meeting_info with title, allEmails, startTime; summary with text/topics.

    Output: True when call is internal and title or dated summary matches outage response.

    Notes: Category B is checked first in the priority router.
    """
    try:
        internal = is_internal_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not internal:
        return False
    title = str(meeting_info.get("title") or "").lower()
    markers = [
        "incident:",
        "outage",
        "war room",
        "root cause",
        "escalation bridge",
    ]
    if any(m in title for m in markers):
        return True
    day = parse_meeting_day(str(meeting_info.get("startTime") or ""))
    text_lower, _ = merge_text(summary)
    if day and date(2026, 3, 10) <= day <= date(2026, 3, 18):
        if "outage" in text_lower or "pipeline" in text_lower:
            return True
    return False


# ========= END OF is_outage_internal =========


# ========= START OF is_outage_fallout =========
def is_outage_fallout(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    External customers reacting negatively to reliability or churn pressure (C).

    Expected input: meeting_info allEmails; summary sentiment and narrative.

    Output: True when external plus outage language plus negative tone bucket.
    """
    try:
        external = is_external_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not external:
        return False
    text_lower, _ = merge_text(summary)
    fallout_terms = [
        "outage",
        "downtime",
        "sla",
        "lost visibility",
        "blind",
        "competitor",
        "churn",
        "renewal at risk",
        "sentinel",
    ]
    if not any(t in text_lower for t in fallout_terms):
        return False
    _, overall = get_sentiment(summary)
    if overall is None:
        return False
    neg = overall.lower()
    return neg in ("negative", "very-negative", "mixed-negative")


# ========= END OF is_outage_fallout =========


# ========= START OF is_comply_v2_launch =========
def is_comply_v2_launch(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Comply v2 launch or GTM motion (E).

    Expected input: meeting title and summary topics/body.

    Output: True when Comply v2 appears with launch vocabulary.
    """
    title = str(meeting_info.get("title") or "").lower()
    text_lower, _ = merge_text(summary)
    if "comply v2" not in title and "comply v2" not in text_lower:
        return False
    launch_terms = [
        "launch",
        "ga deployment",
        "preview",
        "demo",
        "early access",
        "onboarding",
        "kickoff",
        "readiness",
    ]
    return any(t in title for t in launch_terms) or any(t in text_lower for t in launch_terms)


# ========= END OF is_comply_v2_launch =========


# ========= START OF is_renewal_call =========
def is_renewal_call(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Contract or commercial renewal discussion (D).

    Expected input: meeting_info title and external roster; summary topics.

    Output: True for external calls with renewal or pricing language.
    """
    try:
        external = is_external_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not external:
        return False
    title = str(meeting_info.get("title") or "").lower()
    text_lower, topics_lower = merge_text(summary)
    renewal_title = any(
        k in title
        for k in [
            "renewal",
            "contract",
            "annual review",
            "pricing",
            "multi-year",
            "commercial",
        ]
    )
    renewal_body = "renewal" in text_lower or "pricing" in text_lower or "contract" in text_lower
    renewal_topics = "renewal" in topics_lower or "pricing" in topics_lower
    return renewal_title or renewal_topics or renewal_body


# ========= END OF is_renewal_call =========


# ========= START OF is_competitive_analysis =========
def is_competitive_analysis(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Competitive evaluation or win-loss intelligence (F).

    Expected input: meeting title and summary body.

    Output: True when explicit competitive framing or named rivals with eval language.
    """
    title = str(meeting_info.get("title") or "").lower()
    text_lower, _ = merge_text(summary)
    if any(
        k in title
        for k in [
            "competitive",
            "win/loss",
            "win-loss",
            "vendor comparison",
        ]
    ):
        return True
    rivals = [
        "sentinelshield",
        "vaultedge",
        "cybernova",
        "fortiguard pro",
        "fortiguardpro",
    ]
    if any(r in text_lower for r in rivals):
        if any(s in text_lower for s in ["evaluating", "comparison", "competing", "versus", "vs "]):
            return True
    return False


# ========= END OF is_competitive_analysis =========


# ========= START OF is_support_case =========
def is_support_case(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Ticketed technical support or troubleshooting (A).

    Expected input: meeting title plus summary keywords.

    Output: True when Support Case number exists or external troubleshooting cues fire.
    """
    title = str(meeting_info.get("title") or "").lower()
    if "support case #" in title:
        return True
    try:
        external = is_external_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not external:
        return False
    text_lower, _ = merge_text(summary)
    signals = [
        "troubleshoot",
        "bug",
        "error",
        "failure",
        "timeout",
        "connector",
        "sync failure",
        "latency",
        "not working",
        "backup",
        "integration",
    ]
    return any(s in text_lower for s in signals)


# ========= END OF is_support_case =========


# ========= START OF is_product_feedback =========
def is_product_feedback(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Customer sharing roadmap or feature pain (G).

    Expected input: external roster, title, topics.

    Output: True when feedback vocabulary appears on an external call.
    """
    try:
        external = is_external_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not external:
        return False
    title = str(meeting_info.get("title") or "").lower()
    text_lower, topics_lower = merge_text(summary)
    title_hits = any(
        k in title
        for k in [
            "product feedback",
            "feature request",
            "roadmap review",
            "roadmap",
        ]
    )
    topic_hits = any(
        k in topics_lower
        for k in [
            "feature request",
            "feature gap",
            "product feedback",
            "product gap",
        ]
    )
    body_hits = any(
        k in text_lower
        for k in [
            "feature gap",
            "pain point",
            "missing functionality",
        ]
    )
    return title_hits or topic_hits or body_hits


# ========= END OF is_product_feedback =========


# ========= START OF is_compliance_audit =========
def is_compliance_audit(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Audit prep or certification evidence work (I).

    Expected input: meeting title and summary topics.

    Output: True for SOC2/ISO/HIPAA style language unless title is a product demo.
    """
    title = str(meeting_info.get("title") or "").lower()
    if any(
        demo in title
        for demo in [
            "demo",
            "preview",
            "early access",
        ]
    ):
        has_audit = any(
            k in title
            for k in [
                "audit",
                "soc 2",
                "soc2",
                "iso 27001",
                "hipaa",
                "pci dss",
                "compliance review",
                "type ii",
                "type 2",
            ]
        )
        if not has_audit:
            return False
    frameworks = [
        "iso 27001",
        "soc 2",
        "soc2",
        "hipaa",
        "pci dss",
        "audit preparation",
        "compliance review",
        "type ii",
        "type 2",
    ]
    if any(fw in title for fw in frameworks):
        return True
    text_lower, topics_blob = merge_text(summary)
    audit_topics = ["audit preparation", "audit readiness", "evidence collection"]
    if any(at in topics_blob for at in audit_topics):
        return True
    if "soc 2" in text_lower or "iso 27001" in text_lower:
        return True
    return False


# ========= END OF is_compliance_audit =========


# ========= START OF is_engineering_planning =========
def is_engineering_planning(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    """
    Engineering cadence or internal planning (H).

    Expected input: internal-only roster and title keywords.

    Output: True for sprint/retro language while excluding incident titles.
    """
    try:
        internal = is_internal_call(meeting_info.get("allEmails") or [])
    except ValueError:
        return False
    if not internal:
        return False
    title = str(meeting_info.get("title") or "").lower()
    outage_block = any(
        k in title
        for k in [
            "outage",
            "incident",
            "war room",
            "root cause",
        ]
    )
    if outage_block:
        return False
    planning = any(
        k in title
        for k in [
            "sprint",
            "standup",
            "retro",
            "retrospective",
            "roadmap",
            "planning",
            "sync",
            "all hands",
        ]
    )
    if not planning:
        return False
    text_lower, _ = merge_text(summary)
    planning_topics = [
        "sprint planning",
        "sprint retrospective",
        "roadmap planning",
    ]
    if any(p in text_lower for p in planning_topics):
        return True
    return True  # Title already matched cadence vocabulary


# ========= END OF is_engineering_planning =========


# ========= START OF classify_by_rules =========
def classify_by_rules(meeting_info: Mapping[str, Any], summary: Mapping[str, Any]) -> Optional[str]:
    """
    Priority-ordered rule pass returning a single category letter or None.

    Expected input: meeting_info and summary dicts (loaded JSON).

    Output: one of A-I when a rule matches; None if every rule returns False.

    Order matches the stakeholder brief: outage first, then Comply v2, commercial,
    customer operations, internal engineering last.
    """
    if is_outage_internal(meeting_info, summary):
        return "B"
    if is_outage_fallout(meeting_info, summary):
        return "C"
    if is_comply_v2_launch(meeting_info, summary):
        return "E"
    if is_renewal_call(meeting_info, summary):
        return "D"
    if is_competitive_analysis(meeting_info, summary):
        return "F"
    if is_support_case(meeting_info, summary):
        return "A"
    if is_product_feedback(meeting_info, summary):
        return "G"
    if is_compliance_audit(meeting_info, summary):
        return "I"
    if is_engineering_planning(meeting_info, summary):
        return "H"
    return None


# ========= END OF classify_by_rules =========


# ========= START OF parse_llm_letter =========
def parse_llm_letter(text: str) -> Optional[str]:
    """
    Extract first valid category letter from model output.

    Expected input: raw model string (may include whitespace or punctuation).

    Output: uppercase letter A-I if found, else None.
    """
    if not text:
        return None
    m = re.search(r"\b([A-Ia-i])\b", text)
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in CATEGORY_LABELS else None


# ========= END OF parse_llm_letter =========


# ========= START OF llm_classify =========
def llm_classify(
    meeting_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    conversation_id: str | None = None,
) -> str:
    """
    Ask the configured chat model to pick exactly one taxonomy letter.

    Expected input: meeting_info (title, allEmails, startTime) and summary dict.

    Output: category letter A-I; falls back to 'H' if the model is unavailable.

    Uses ModelSingleton from llm_model_config with temperature 0 for stability.
    """
    model = ModelSingleton.get_instance()
    if model is None:
        print("[llm_classify] Model unavailable; defaulting to H.")
        return "H"
    title = str(meeting_info.get("title") or "")
    body = str(summary.get("summary") or "")
    topics = summary.get("topics") or []
    topics_text = ", ".join(str(t) for t in topics)
    _, sentiment = get_sentiment(summary)
    try:
        ext_flag = is_external_call(meeting_info.get("allEmails") or [])
    except ValueError:
        ext_flag = False
    lines = [
        "You are a classifier. Choose ONE category letter (A through I) for this meeting.",
        "Reply with only that single letter and nothing else.",
        "",
        "A - Technical Support & Troubleshooting",
        "B - Platform Outage & Incident Response (internal war rooms)",
        "C - Customer Outage Fallout & Churn Risk",
        "D - Contract Renewal & Commercial Negotiation",
        "E - Comply v2 Launch & Go-to-Market",
        "F - Competitive Threat & Win/Loss Analysis",
        "G - Customer Product Feedback & Feature Requests",
        "H - Engineering Planning & Execution",
        "I - Compliance & Audit Preparation",
        "",
        f"Title: {title}",
        f"Summary: {body}",
        f"Topics: {topics_text}",
        f"Sentiment: {sentiment}",
        f"External call: {ext_flag}",
    ]
    user_prompt = "\n".join(lines)
    instr = "You classify meetings into exactly one letter A-I. Output only the letter."
    agent = Agent(
        name="taxonomy_classifier",
        instructions=instr,
        model=model,
        model_settings=ModelSettings(temperature=0.0),
    )

    extras: dict[str, Any] = {"temperature": 0.0, "purpose": "taxonomy_letter"}
    if conversation_id:
        extras["conversation_id"] = conversation_id
    model_id = os.getenv("model_name")
    if model_id:
        extras["model_name"] = model_id

    result = None
    try:
        result = Runner.run_sync(agent, user_prompt)
    except Exception as exc:
        append_llm_call_log(
            call_site="llm_classify",
            agent_name="taxonomy_classifier",
            agent_instructions=instr,
            user_input=user_prompt,
            response=None,
            error=str(exc),
            extras=extras,
        )
        raise
    raw_out = str(getattr(result, "final_output", "") or "")
    letter = parse_llm_letter(raw_out)
    extras_log = dict(extras)
    if letter is None:
        extras_log["parse_failed"] = True
        extras_log["normalized_letter"] = "H"
        append_llm_call_log(
            call_site="llm_classify",
            agent_name="taxonomy_classifier",
            agent_instructions=instr,
            user_input=user_prompt,
            response=raw_out,
            error="Could not parse A-I letter from model output; defaulted to H",
            extras=extras_log,
        )
        print("[llm_classify] Could not parse letter; defaulting to H.")
        return "H"

    extras_log["parsed_category_letter"] = letter
    append_llm_call_log(
        call_site="llm_classify",
        agent_name="taxonomy_classifier",
        agent_instructions=instr,
        user_input=user_prompt,
        response=raw_out,
        error=None,
        extras=extras_log,
    )
    return letter


# ========= END OF llm_classify =========


# ========= START OF classify_conversation =========
def classify_conversation(
    meeting_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    conversation_id: str | None = None,
) -> tuple[str, str]:
    """
    Full classification path: deterministic rules, then LLM if still ambiguous.

    Expected input: meeting_info and summary dicts for one conversation.

    Output: (category_letter, source_tag) where source_tag is 'rules' or 'llm'.
    """
    ruled = classify_by_rules(meeting_info, summary)
    if ruled:
        return ruled, "rules"
    letter = llm_classify(meeting_info, summary, conversation_id=conversation_id)
    return letter, "llm"


# ========= END OF classify_conversation =========


# ========= START OF build_row_record =========
def build_row_record(
    conversation_id: str,
    meeting_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    category: str,
    source: str,
) -> dict[str, Any]:
    """
    Normalize one conversation into a flat dict for CSV and dashboards.

    Expected input: ids, meeting_info, summary, category letter, rules/llm source.

    Output: dict with conversation_id, title, category, labels, sentiment, duration, dates.
    """
    score, _ = get_sentiment(summary)
    duration_val, _ = format_duration(meeting_info)
    start = str(meeting_info.get("startTime") or "")
    start_day = start[:10] if start else ""
    title = str(meeting_info.get("title") or "")
    stakeholder = PRIMARY_STAKEHOLDER_BY_CATEGORY.get(category, "unknown")
    return {
        "conversation_id": conversation_id,
        "title": title,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "stakeholder_primary": stakeholder,
        "classification_source": source,
        "sentiment_score": score if score is not None else "",
        "duration_min": duration_val if duration_val is not None else "",
        "start_date": start_day,
    }


# ========= END OF build_row_record =========


# ========= START OF build_support_dashboard =========
def build_support_dashboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Curated Support Leaders view (categories A, B, C).

    Expected input: list of row dicts from build_row_record.

    Output: dict with label, allowed categories, counts, detail rows, optional avg sentiment for C.
    """
    allowed = {"A", "B", "C"}
    filtered = [r for r in rows if r["category"] in allowed]
    counts: dict[str, int] = defaultdict(int)
    for r in filtered:
        counts[r["category"]] += 1
    c_scores = []
    for r in filtered:
        if r["category"] == "C" and r["sentiment_score"] != "":
            try:
                c_scores.append(float(r["sentiment_score"]))
            except (ValueError, TypeError):
                continue
    avg_c = sum(c_scores) / len(c_scores) if c_scores else None
    return {
        "stakeholder_key": "support_leaders",
        "dashboard_label": STAKEHOLDER_DASHBOARDS["support_leaders"]["label"],
        "categories": sorted(allowed),
        "counts": dict(counts),
        "rows": filtered,
        "avg_sentiment_category_c": avg_c,
    }


# ========= END OF build_support_dashboard =========


# ========= START OF build_sales_dashboard =========
def build_sales_dashboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Curated Sales Managers view (D, E, F).

    Expected input: unified row list.

    Output: dict with counts and matching rows for revenue-facing categories.
    """
    allowed = {"D", "E", "F"}
    filtered = [r for r in rows if r["category"] in allowed]
    counts: dict[str, int] = defaultdict(int)
    for r in filtered:
        counts[r["category"]] += 1
    return {
        "stakeholder_key": "sales_managers",
        "dashboard_label": STAKEHOLDER_DASHBOARDS["sales_managers"]["label"],
        "categories": sorted(allowed),
        "counts": dict(counts),
        "rows": filtered,
    }


# ========= END OF build_sales_dashboard =========


# ========= START OF build_product_dashboard =========
def build_product_dashboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Curated Product Managers view (G, E, F).

    Expected input: unified row list.

    Output: dict ready for console/HTML blocks focused on product strategy.
    """
    allowed = {"G", "E", "F"}
    filtered = [r for r in rows if r["category"] in allowed]
    counts: dict[str, int] = defaultdict(int)
    for r in filtered:
        counts[r["category"]] += 1
    return {
        "stakeholder_key": "product_managers",
        "dashboard_label": STAKEHOLDER_DASHBOARDS["product_managers"]["label"],
        "categories": sorted(allowed),
        "counts": dict(counts),
        "rows": filtered,
    }


# ========= END OF build_product_dashboard =========


# ========= START OF build_engineering_dashboard =========
def build_engineering_dashboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Curated Engineering Leads view (H, B, I).

    Expected input: unified row list.

    Output: dict covering execution, shared outage, and compliance themes.
    """
    allowed = {"H", "B", "I"}
    filtered = [r for r in rows if r["category"] in allowed]
    counts: dict[str, int] = defaultdict(int)
    for r in filtered:
        counts[r["category"]] += 1
    return {
        "stakeholder_key": "engineering_leads",
        "dashboard_label": STAKEHOLDER_DASHBOARDS["engineering_leads"]["label"],
        "categories": sorted(allowed),
        "counts": dict(counts),
        "rows": filtered,
    }


# ========= END OF build_engineering_dashboard =========


# ========= START OF print_dashboard_summary =========
def print_dashboard_summary(payload: dict[str, Any]) -> None:
    """
    Emit a human-readable mini report for one stakeholder dict.

    Expected input: dict returned by any build_*_dashboard function.

    Output: printed lines only (no return value).
    """
    print("")
    print("=" * 72)
    print(payload["dashboard_label"])
    print("Categories in view:", ", ".join(payload["categories"]))
    print("Counts by letter:", payload["counts"])
    if payload.get("avg_sentiment_category_c") is not None:
        print(
            "Average sentiment (category C only):",
            round(payload["avg_sentiment_category_c"], 2),
        )
    print("Meetings in this view:", len(payload["rows"]))


# ========= END OF print_dashboard_summary =========


# ========= START OF write_master_csv =========
def write_master_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """
    Persist the combined classification table for spreadsheet analysis.

    Expected input: list of build_row_record dicts; output Path for CSV.

    Output: writes UTF-8 CSV with one row per conversation.

    Columns align with the stakeholder reporting brief.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conversation_id",
        "title",
        "category",
        "category_label",
        "stakeholder_primary",
        "classification_source",
        "sentiment_score",
        "duration_min",
        "start_date",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ========= END OF write_master_csv =========


# ========= START OF write_stakeholder_slice_csv =========
def write_stakeholder_slice_csv(payload: dict[str, Any], path: Path) -> None:
    """
    Save a filtered CSV for a single stakeholder dashboard.

    Expected input: dashboard dict with rows; destination path.

    Output: CSV limited to that stakeholder's category lens.
    """
    write_master_csv(payload["rows"], path)


# ========= END OF write_stakeholder_slice_csv =========


# ========= START OF write_plotly_html_report =========
def write_plotly_html_report(
    dashboards: list[dict[str, Any]],
    path: Path,
) -> None:
    """
    Build a simple multi-panel HTML bar chart for each stakeholder view.

    Expected input: list of dashboard dicts; output Path for HTML.

    Output: standalone HTML file, or prints a skip message if Plotly missing.

    Linearizes counts per category for quick visual comparison.
    """
    if go is None:
        print("[write_plotly_html_report] plotly not installed; skipping HTML chart.")
        return
    fig = go.Figure()
    for payload in dashboards:
        letters = payload["categories"]
        counts = [payload["counts"].get(letter, 0) for letter in letters]
        labels = [f"{letter} – {CATEGORY_LABELS[letter]}" for letter in letters]
        fig.add_trace(
            go.Bar(
                name=payload["dashboard_label"],
                x=labels,
                y=counts,
                text=counts,
                textposition="auto",
            )
        )
    fig.update_layout(
        title="Transcript Intelligence – stakeholder category counts",
        barmode="group",
        legend_title="Dashboard",
        xaxis_tickangle=-25,
        height=560,
        margin=dict(l=40, r=40, t=70, b=140),
        template="plotly_white",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print("Wrote HTML dashboard:", path)


# ========= END OF write_plotly_html_report =========


# ========= START OF write_simple_html_dashboard =========
def write_simple_html_dashboard(
    dashboards: list[dict[str, Any]],
    path: Path,
) -> None:
    """
    Write a self-contained HTML dashboard with CSS bar charts (no Plotly).

    Expected input: same dashboard dicts as the Plotly exporter.

    Output: static HTML file readable in any browser.

    Used when plotly is unavailable or for zero-dependency deployment.
    """
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>Transcript Intelligence – stakeholder views</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;background:#f6f7fb;}",
        "h1{font-size:1.35rem;color:#1a1a2e;}",
        ".panel{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:28px;",
        "box-shadow:0 4px 24px rgba(20,30,60,.08);}",
        ".panel h2{margin:0 0 16px;font-size:1.1rem;color:#2d3a4a;}",
        ".bar-wrap{display:flex;align-items:flex-end;height:200px;gap:12px;",
        "padding:8px 0;border-bottom:1px solid #e8ebf2;}",
        ".bar-col{flex:1;display:flex;flex-direction:column;align-items:center;}",
        ".bar{background:linear-gradient(180deg,#4f7cff,#2c4fd4);width:100%;",
        "border-radius:6px 6px 2px 2px;min-height:4px;transition:height .3s;}",
        ".bar.alt{background:linear-gradient(180deg,#00b894,#00916e);}",
        ".bar.alt2{background:linear-gradient(180deg,#fdcb6e,#e17055);}",
        ".lbl{font-size:.72rem;color:#555;margin-top:10px;text-align:center;line-height:1.35;}",
        ".cnt{font-weight:600;font-size:.85rem;color:#1a1a2e;margin-bottom:6px;}",
        "</style></head><body>",
        "<h1>Transcript Intelligence – stakeholder category counts</h1>",
        "<p style='color:#666;max-width:720px;'>Each panel shows one stakeholder dashboard. ",
        "Bars are scaled to the highest count in that panel.</p>",
    ]
    bar_classes = ["bar", "bar alt", "bar alt2"]
    for payload in dashboards:
        max_count = 1
        for letter in payload["categories"]:
            max_count = max(max_count, int(payload["counts"].get(letter, 0)))
        parts.append("<div class='panel'>")
        parts.append(f"<h2>{_esc(payload['dashboard_label'])}</h2>")
        parts.append("<div class='bar-wrap'>")
        for j, letter in enumerate(payload["categories"]):
            cnt = int(payload["counts"].get(letter, 0))
            pct = max(8, int(100 * cnt / max_count)) if max_count else 8
            cls = bar_classes[j % len(bar_classes)]
            label = f"{letter} – {CATEGORY_LABELS[letter]}"
            parts.append("<div class='bar-col'>")
            parts.append(f"<div class='cnt'>{cnt}</div>")
            parts.append(f"<div class='{cls}' style='height:{pct}px;width:100%;'></div>")
            parts.append(f"<div class='lbl'>{_esc(label)}</div>")
            parts.append("</div>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    print("Wrote simple HTML dashboard:", path)


# ========= END OF write_simple_html_dashboard =========


# ========= START OF main =========
def main() -> None:
    """
    Load every conversation bundle, classify, emit console dashboards, CSV, and HTML.

    Expected input: dataset folders under PROJECT/dataset/*/meeting-info.json.

    Output: prints summaries and writes classification_output artifacts.

    No CLI flags yet: edit DEFAULT_* constants above if paths must change.
    """
    dataset_root, _project_root, output_dir = get_project_paths()
    conv_dirs = discover_conversation_dirs(dataset_root)
    if not conv_dirs:
        print("No conversation bundles found under", dataset_root)
        return
    rows: list[dict[str, Any]] = []
    llm_needed = 0
    rules_hits = 0
    for folder in conv_dirs:
        cid = folder.name
        meeting_info, summary = load_conversation_bundle(folder)
        cat, src = classify_conversation(meeting_info, summary, conversation_id=cid)
        if src == "llm":
            llm_needed += 1
        else:
            rules_hits += 1
        rows.append(build_row_record(cid, meeting_info, summary, cat, src))
    dashboards = [
        build_support_dashboard(rows),
        build_sales_dashboard(rows),
        build_product_dashboard(rows),
        build_engineering_dashboard(rows),
    ]
    print("Processed conversations:", len(rows))
    print("Rule-based classifications:", rules_hits)
    print("LLM classifications:", llm_needed)
    for dash in dashboards:
        print_dashboard_summary(dash)
    master_csv = output_dir / "all_classified.csv"
    write_master_csv(rows, master_csv)
    print("")
    print("Wrote combined CSV:", master_csv)
    for dash in dashboards:
        slug = dash["stakeholder_key"]
        slice_path = output_dir / f"{slug}_view.csv"
        write_stakeholder_slice_csv(dash, slice_path)
        print("Wrote stakeholder CSV:", slice_path)
    html_plotly = output_dir / "stakeholder_bar_charts_plotly.html"
    write_plotly_html_report(dashboards, html_plotly)
    html_simple = output_dir / "stakeholder_dashboard.html"
    write_simple_html_dashboard(dashboards, html_simple)


# ========= END OF main =========


if __name__ == "__main__":
    main()
