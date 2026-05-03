#!/usr/bin/env python3
"""
Extract a slim slice from dataset/unified_records.json.

Default: first N records with fields
  record, title, summary, overallSentiment, sentimentScore, topics

Run from repo root:

  python src/extract_unified_subset.py
  python src/extract_unified_subset.py --limit 50 --output dataset/subset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields requested (six keys; unify with unified_records.json)
FIELD_KEYS: tuple[str, ...] = (
    "record",
    "title",
    "summary",
    "overallSentiment",
    "sentimentScore",
    "topics",
)


def pick_fields(record: dict) -> dict[str, object]:
    return {key: record[key] for key in FIELD_KEYS}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    default_input = root / "dataset" / "unified_records.json"
    default_output = root / "dataset" / "unified_records_subset_100.json"

    parser = argparse.ArgumentParser(
        description="Extract record, title, summary, sentiment, and topics from unified_records.json."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"path to unified_records.json (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"JSON output path (default: {default_output})",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=100,
        help="number of records from the start of the array (default: 100)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr)
        sys.exit(1)

    with args.input.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    missing = []
    for i, rec in enumerate(records[: args.limit]):
        for key in FIELD_KEYS:
            if key not in rec:
                missing.append((i, key))
    if missing:
        first_idx, key = missing[0]
        print(
            f"missing field {key!r} in record index {first_idx}",
            file=sys.stderr,
        )
        sys.exit(1)

    subset = [pick_fields(rec) for rec in records[: args.limit]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {len(subset)} records to {args.output}")


if __name__ == "__main__":
    main()
