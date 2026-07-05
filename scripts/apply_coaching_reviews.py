#!/usr/bin/env python
"""Promote completed human coaching reviews into the corpus manifest.

The proposal packet remains the review record. This script only copies explicit
human decisions; it never derives labels from engine-generated evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


_DECISION_FIELDS = (
    "agenda_must_include",
    "agenda_must_exclude",
    "diagnosis_by_ply",
    "key_reason_tokens_by_ply",
    "together_plies",
    "separate_plies",
    "slide_count",
)


def _require_review_metadata(decision: dict, fixture_id: str) -> tuple[str, str]:
    reviewer = str(decision.get("reviewed_by") or "").strip()
    reviewed_on = str(decision.get("reviewed_on") or "").strip()
    if not reviewer:
        raise ValueError(f"{fixture_id}: human_decision.reviewed_by is required")
    try:
        date.fromisoformat(reviewed_on)
    except ValueError as exc:
        raise ValueError(
            f"{fixture_id}: human_decision.reviewed_on must be YYYY-MM-DD"
        ) from exc
    return reviewer, reviewed_on


def _approved_expected(packet: dict, fixture: dict) -> dict:
    fixture_id = fixture["id"]
    review = packet.get("review") or {}
    decision = packet.get("human_decision") or {}
    if review.get("student_color") != fixture.get("student_color"):
        raise ValueError(f"{fixture_id}: proposal student_color does not match manifest")
    if review.get("eval_source") != "pgn":
        raise ValueError(f"{fixture_id}: proposal eval_source must be 'pgn'")
    if "slide_count" not in decision or decision["slide_count"] is None:
        raise ValueError(f"{fixture_id}: human_decision.slide_count must be explicit")
    if not isinstance(decision["slide_count"], int) or decision["slide_count"] < 0:
        raise ValueError(f"{fixture_id}: human_decision.slide_count must be a non-negative integer")
    missing = [field for field in _DECISION_FIELDS if field not in decision]
    if missing:
        raise ValueError(f"{fixture_id}: human_decision lacks fields: {', '.join(missing)}")

    for field in ("agenda_must_include", "agenda_must_exclude"):
        if not isinstance(decision[field], list) or not all(
            isinstance(ply, int) and ply > 0 for ply in decision[field]
        ):
            raise ValueError(f"{fixture_id}: {field} must contain positive integer plies")
    for field in ("diagnosis_by_ply", "key_reason_tokens_by_ply"):
        if not isinstance(decision[field], dict):
            raise ValueError(f"{fixture_id}: {field} must be an object")
    for field in ("together_plies", "separate_plies"):
        if not isinstance(decision[field], list) or not all(
            isinstance(pair, list) and len(pair) == 2
            and all(isinstance(ply, int) and ply > 0 for ply in pair)
            for pair in decision[field]
        ):
            raise ValueError(f"{fixture_id}: {field} must contain two-ply arrays")

    expected = {
        "student_color": fixture["student_color"],
        "eval_source": "pgn",
    }
    expected.update({field: decision[field] for field in _DECISION_FIELDS})
    return expected


def apply_review_decisions(
    manifest_path: str | Path,
    proposal_dir: str | Path,
    fixture_ids: set[str] | None = None,
) -> list[str]:
    """Apply explicitly approved/rejected proposal decisions and return their IDs."""
    manifest_path = Path(manifest_path)
    proposal_dir = Path(proposal_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = {fixture["id"]: fixture for fixture in manifest.get("fixtures", [])}
    requested = fixture_ids or set(fixtures)
    unknown = requested - set(fixtures)
    if unknown:
        raise ValueError(f"unknown fixture ids: {', '.join(sorted(unknown))}")

    applied = []
    for fixture_id in sorted(requested):
        fixture = fixtures[fixture_id]
        proposal_path = proposal_dir / f"{fixture_id}.proposal.json"
        if not proposal_path.exists():
            raise ValueError(f"{fixture_id}: proposal file not found: {proposal_path}")
        packet = json.loads(proposal_path.read_text(encoding="utf-8"))
        if packet.get("fixture_id") != fixture_id:
            raise ValueError(f"{fixture_id}: proposal fixture_id mismatch")
        if packet.get("source_sha256") != fixture.get("sha256"):
            raise ValueError(f"{fixture_id}: proposal source hash does not match manifest")

        decision = packet.get("human_decision") or {}
        approved = decision.get("approve_fixture")
        if approved is None:
            continue
        if not isinstance(approved, bool):
            raise ValueError(f"{fixture_id}: approve_fixture must be true, false, or null")
        reviewer, reviewed_on = _require_review_metadata(decision, fixture_id)
        fixture["review_status"] = "approved" if approved else "rejected"
        fixture["reviewed_by"] = reviewer
        fixture["reviewed_on"] = reviewed_on
        fixture["expected"] = _approved_expected(packet, fixture) if approved else {}
        applied.append(fixture_id)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return applied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy explicit human proposal decisions into the coaching manifest."
    )
    parser.add_argument("manifest", help="Path to tests/fixtures/coaching/manifest.json")
    parser.add_argument("proposal_dir", help="Directory containing *.proposal.json files")
    parser.add_argument("--fixture", action="append", dest="fixture_ids",
                        help="Apply only this fixture id (repeatable)")
    args = parser.parse_args(argv)
    try:
        applied = apply_review_decisions(
            args.manifest,
            args.proposal_dir,
            set(args.fixture_ids) if args.fixture_ids else None,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Applied {len(applied)} human review(s): {', '.join(applied) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
