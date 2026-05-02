# Transcript Intelligence

A repository for meeting transcript data, summaries, speaker metadata, and related analysis artifacts—used for transcript intelligence tooling, modeling, or evaluation.

## What's in this repo

| Path | Description |
|------|-------------|
| `dataset/` | One folder per conversation (named by conversation ID). Each folder holds structured JSON for that meeting. |
| `dataset/unified_records.json` | Consolidated index: one entry per conversation with titles, timing, summaries, sentiment/topics (when present), and paths to per-meeting files. |
| `Analysis/transcript_intelligence_design.docx` | Design notes for the transcript intelligence initiative. |

## Per-meeting files

Inside each conversation folder under `dataset/<conversation_id>/` you typically find:

- `meeting-info.json` — identifiers, organizer, invitees, schedule, duration
- `transcript.json` — full conversation transcript
- `summary.json` — meeting summary
- `speakers.json` / `speaker-meta.json` — speaker lists and metadata
- `events.json` — structured events extracted from the meeting (when available)

Paths in `unified_records.json` are relative to `dataset/` (for example `01KQ…/summary.json`).

## Local setup (Python)

A virtual environment is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Dependencies can be added to a `requirements.txt` as the project grows; `.venv/` is ignored by `.gitignore` and should not be committed.

## License

If you intend to publish this repository, add a license file and update this section accordingly.
