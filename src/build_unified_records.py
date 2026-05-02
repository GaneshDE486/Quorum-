#!/usr/bin/env python3
"""
Build dataset/unified_records.json from each conversation folder's
meeting-info.json and summary.json. Paths are relative to dataset/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dataset = root / "dataset"
    if not dataset.is_dir():
        print("dataset folder not found", file=sys.stderr)
        sys.exit(1)

    dirs = sorted(
        p for p in dataset.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    records = []
    for i, folder in enumerate(dirs, start=1):
        cid = folder.name
        with (folder / "meeting-info.json").open(encoding="utf-8") as f:
            info = json.load(f)
        with (folder / "summary.json").open(encoding="utf-8") as f:
            summary = json.load(f)

        mid = info.get("meetingId", cid)
        if mid != cid:
            print(
                f"warning: folder {cid} meetingId {mid} mismatch; using folder name as conversation_id",
                file=sys.stderr,
            )

        records.append(
            {
                "record": i,
                "conversation_id": cid,
                "title": info["title"],
                "organizerEmail": info["organizerEmail"],
                "allEmails": info["allEmails"],
                "duration_min": info["duration"],
                "startTime": info["startTime"],
                "endTime": info["endTime"],
                "summary": summary["summary"],
                "overallSentiment": summary["overallSentiment"],
                "sentimentScore": summary["sentimentScore"],
                "topics": summary["topics"],
                "meeting_summary_path": f"{cid}/summary.json",
                "total_conversation_path": f"{cid}/transcript.json",
                "speakers_path": f"{cid}/speaker-meta.json",
                "meeting_info_path": f"{cid}/meeting-info.json",
                "speaker_info_path": f"{cid}/speakers.json",
                "event_path": f"{cid}/events.json",
            }
        )

    out_path = dataset / "unified_records.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
