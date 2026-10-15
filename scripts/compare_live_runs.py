#!/usr/bin/env python
"""Compare live-eval runs step by step (for decisions such as making the router the default).

Reviewers group steps differently ("S8-S13" in one review, "S8" in another), so a FAIL count
is not comparable across runs. This maps every reviewer finding to the script steps it covers
and prints, per step, each run's verdicts, plus execution status, model calls by model, slow
turns, and deterministic checker failures.

    .venv/bin/python scripts/compare_live_runs.py profiles/live-eval-...-A profiles/live-eval-...-B
"""
import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"S(\d+)(?:\s*[–-]\s*S?(\d+))?")
RANK = {"FAIL": 0, "BLOCKED": 1, "MIXED": 2, "NOT OBSERVED": 3, "PASS": 4}


def steps_of(check: str) -> list:
    """Script steps a finding covers, from names like 'S8–S13 — Move 24' or 'S14a'."""
    steps = []
    for start, end in STEP_RE.findall(check.split("—")[0].split(":")[0]):
        lo = int(start)
        steps.extend(range(lo, int(end) + 1) if end else [lo])
    return sorted(set(steps))


def summarise(run_dir: Path, slow_s: float) -> dict:
    review = json.loads((run_dir / "review.json").read_text()) if (run_dir / "review.json").exists() else None
    execution = json.loads((run_dir / "execution.json").read_text())
    per_step = collections.defaultdict(list)
    whole = []
    for finding in (review or {}).get("findings", []):
        steps = steps_of(finding["check"])
        for s in steps:
            per_step[s].append(finding["verdict"])
        if not steps:
            whole.append((finding["verdict"], finding["check"]))
    calls = collections.Counter()
    slow = []
    stamp = lambda s: dt.datetime.fromisoformat(s)  # noqa: E731
    current = None
    for line in (run_dir / "events.jsonl").open():
        event = json.loads(line)
        if event["kind"] == "model_request" and isinstance(event["data"].get("payload"), dict):
            calls[(event["data"]["payload"].get("_url") or {}).get("model", "?").split("/")[-1]] += 1
        elif event["kind"] == "student":
            current = (stamp(event["timestamp"]), event["data"]["text"][:40])
        elif event["kind"] in ("coach_reply", "turn_error", "session_interrupt") and current:
            seconds = (stamp(event["timestamp"]) - current[0]).total_seconds()
            if seconds > slow_s:
                slow.append(f"{seconds:.0f}s '{current[1]}'")
            current = None
    checks = json.loads((run_dir / "checks.json").read_text()) if (run_dir / "checks.json").exists() else {}
    check_fails = sum(1 for c in checks.get("checks", []) if c.get("verdict") == "FAIL")
    return {"run": run_dir.name.split("-")[-1], "status": execution.get("execution_status"),
            "last_step": execution.get("last_sent_step"), "per_step": per_step, "whole": whole,
            "calls": dict(calls), "slow": slow, "check_fails": check_fails, "reviewed": review is not None}


def worst(verdicts: list) -> str:
    return min(verdicts, key=lambda v: RANK.get(v, 9)) if verdicts else "—"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--slow", type=float, default=60.0, help="report turns slower than this (s)")
    args = parser.parse_args(argv)
    runs = [summarise(r, args.slow) for r in args.runs]
    print("| | " + " | ".join(r["run"] for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for label, key in (("Execution", "status"), ("Last step", "last_step"), ("Model calls", "calls"),
                       ("Checker FAILs", "check_fails")):
        print(f"| {label} | " + " | ".join(str(r[key]) for r in runs) + " |")
    print(f"| Turns > {args.slow:.0f}s | " + " | ".join("; ".join(r["slow"]) or "none" for r in runs) + " |")
    steps = sorted({s for r in runs for s in r["per_step"]})
    print("\n| Step | " + " | ".join(r["run"] for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for s in steps:
        print(f"| S{s} | " + " | ".join(worst(r["per_step"].get(s, [])) for r in runs) + " |")
    for r in runs:
        for verdict, check in r["whole"]:
            print(f"- {r['run']} whole-session: {verdict} — {check}")
    if not all(r["reviewed"] for r in runs):
        print("\nWARNING: a run has no review.json; its verdict columns are empty.", file=sys.stderr)


if __name__ == "__main__":
    main()
