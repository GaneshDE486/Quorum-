# Transcript Intelligence

Meeting bundles under `dataset/<conversation_id>/` are classified into nine categories (**A–I**) with **rules-first** batch logic and optional **Groq** LLM fallback. A **FastAPI** app serves stakeholder dashboards, per-meeting drill-down, and optional **on-demand** full-transcript summarisation.

---

## How to run (quick start)

All commands assume the **repository root** as the current directory.

### 1. Python environment

Use Python 3.10 or newer (3.11+ recommended).

```bash
cd "/path/to/Transcript Intelligence"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Groq (for LLM features)

Create a `.env` file in the project root (same folder as `requirements.txt`). The app loads it automatically via `python-dotenv`.

| Variable | Required for | Notes |
|----------|----------------|--------|
| `GROQ_API_KEY` | LLM classification fallback, transcript summarisation in the UI | From [Groq Console](https://console.groq.com/) |
| `model_name` | Same | Example: `llama-3.3-70b-versatile` (use a chat model id your key supports) |
| `GROQ_BASE_URL` | Optional | Defaults to Groq’s OpenAI-compatible endpoint |
| `TI_LLM_LOG_FILE` | Optional | Override path for append-only LLM logs (default: `dataset/llm_output/llm_calls.jsonl`) |

Meetings that match **rules** only do **not** call the model. If `GROQ_API_KEY` / `model_name` are missing, rule-only rows still work; anything that needs the LLM will fail or fall back according to the code paths you trigger.

### 3. (Optional) Regenerate classification CSVs

If you add or change conversation folders, or you want to refresh labels and stakeholder slices:

```bash
python3 src/classify_transcripts_dashboard.py
```

**Inputs:** each conversation folder must include at least `meeting-info.json` and `summary.json`.

**Outputs (under `dataset/classification_output/`):**

- `all_classified.csv` — master table the API reads
- `*_view.csv` — per–stakeholder-tab slices
- `stakeholder_bar_charts_plotly.html`, `stakeholder_dashboard.html` — static reports

### 4. Start the web dashboard

```bash
uvicorn src.fastapi_dashboard:app --reload --host 127.0.0.1 --port 8000
```

Then open:

| URL | Purpose |
|-----|---------|
| [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Main stakeholder dashboard (reads `all_classified.csv`) |
| [http://127.0.0.1:8000/architecture](http://127.0.0.1:8000/architecture) | Architecture & classification doc (`docs/ARCHITECTURE.html` when built) |

**Architecture HTML:** after editing `docs/ARCHITECTURE.md`, regenerate the styled page:

```bash
python3 scripts/build_architecture_html.py
```

---

## Other utilities

| Command | Purpose |
|---------|---------|
| `python3 src/build_unified_records.py` | Rebuild `dataset/unified_records.json` from each folder’s `meeting-info.json` + `summary.json` |
| `python3 src/extract_unified_subset.py` | Slim export helper (see script header for usage) |

---

## Repository layout

| Path | Role |
|------|------|
| `dataset/<id>/` | Per-meeting JSON (`meeting-info.json`, `summary.json`, `transcript.json`, speakers/events metadata, …) |
| `dataset/classification_output/` | CSV + HTML outputs from `classify_transcripts_dashboard.py` |
| `dataset/llm_output/llm_calls.jsonl` | Optional append-only log of LLM calls |
| `src/fastapi_dashboard.py` | FastAPI app (`uvicorn src.fastapi_dashboard:app …`) |
| `src/classify_transcripts_dashboard.py` | Batch rules + LLM fallback + CSV export |
| `src/llm_model_config.py` | Groq client / `ModelSingleton` |
| `docs/ARCHITECTURE.md` | Source for the architecture page; build to `ARCHITECTURE.html` with the script above |
| `docs/ARCHITECTURE.html` | Pre-built doc (preferred by `/architecture`) |
| `src/plan.yaml` | Human-readable taxonomy / plan mirror |

Design notes also live under `Analysis/` (e.g. Word doc).

---

## License

Add a `LICENSE` file and update this section if you publish the repository.
