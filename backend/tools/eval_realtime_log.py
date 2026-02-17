#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.words.metrics import compute_fp_per_minute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate realtime words log (jsonl)")
    parser.add_argument("--log", required=True, help="Path to words runtime log jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[eval_realtime_log] log file not found: {log_path}")
        return 1

    events: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

    if not events:
        print("[eval_realtime_log] no events")
        return 1

    events = sorted(events, key=lambda x: int(x.get("timestamp_ms", 0)))
    start_ms = int(events[0].get("timestamp_ms", 0))
    end_ms = int(events[-1].get("timestamp_ms", 0))
    duration_sec = max(1e-6, (end_ms - start_ms) / 1000.0)

    states = {}
    commit_words = []
    latencies = []
    for event in events:
        state = str(event.get("state", "UNKNOWN")).upper()
        states[state] = states.get(state, 0) + 1
        if state == "COMMIT":
            commit_words.append(str(event.get("word", "")))
        latency = event.get("latency_ms")
        if latency is not None:
            try:
                latencies.append(float(latency))
            except Exception:
                pass

    fp_min = compute_fp_per_minute(events)
    commit_rate = len(commit_words) / max(1e-6, duration_sec / 60.0)

    print("[eval_realtime_log] summary")
    print(f"  events={len(events)}")
    print(f"  duration_sec={duration_sec:.2f}")
    print(f"  states={states}")
    print(f"  commits={len(commit_words)}")
    print(f"  commit_rate_per_min={commit_rate:.3f}")
    print(f"  fp_per_min={fp_min:.3f}")

    if latencies:
        arr = np.asarray(latencies, dtype=np.float32)
        print(f"  infer_latency_avg_ms={float(arr.mean()):.3f}")
        print(f"  infer_latency_p95_ms={float(np.percentile(arr, 95)):.3f}")
    else:
        print("  infer_latency_avg_ms=n/a")
        print("  infer_latency_p95_ms=n/a")

    top_words = {}
    for word in commit_words:
        top_words[word] = top_words.get(word, 0) + 1
    top_words_sorted = sorted(top_words.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print(f"  top_commit_words={top_words_sorted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
