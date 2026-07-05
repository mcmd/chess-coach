#!/usr/bin/env python
"""Generate engine evidence packets for human approval of the P0-5b corpus.

The output is deliberately marked PROPOSAL ONLY. This script never changes the
manifest's human review status and therefore cannot make the corpus gate pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from scripts.eval_coaching import review_proposal_packet, validate_corpus_manifest


def _load_runtime_config():
    from app.config import load_config

    saved = sys.argv
    try:
        # load_config owns the application's CLI flags; corpus flags belong here.
        sys.argv = [saved[0]]
        return load_config()
    finally:
        sys.argv = saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate real-engine proposal packets for P0-5b human review.")
    parser.add_argument("manifest", help="Path to the curated corpus manifest.json")
    parser.add_argument("output_dir", help="Directory for one proposal JSON per fixture")
    parser.add_argument("--agenda-size", type=int, default=8,
                        help="Candidate moments to deep-enrich for human selection (default: 8)")
    parser.add_argument("--max-fixtures", type=int, default=0,
                        help="Generate only the first N fixtures (0 = all)")
    parser.add_argument("--fixture", action="append", dest="fixture_ids",
                        help="Generate only this fixture id (repeatable)")
    parser.add_argument("--without-maia", action="store_true",
                        help="Skip Maia evidence; Stockfish and PGN evals still run")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    report = validate_corpus_manifest(manifest_path)
    if report["source_errors"]:
        print("Corpus source validation failed:", file=sys.stderr)
        for error in report["source_errors"]:
            print(f"  - {error}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    if args.fixture_ids:
        requested = set(args.fixture_ids)
        fixtures = [fixture for fixture in fixtures if fixture["id"] in requested]
        missing = requested - {fixture["id"] for fixture in fixtures}
        if missing:
            print(f"Unknown fixture ids: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    if args.max_fixtures:
        fixtures = fixtures[:args.max_fixtures]

    load_dotenv()
    config = _load_runtime_config()
    config.use_online_databases = False

    from app.engines import EngineManager
    from app.review import review_game

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engines = EngineManager(config)
    engines.start()
    try:
        for index, fixture in enumerate(fixtures, 1):
            fixture_id = fixture["id"]
            print(f"[{index}/{len(fixtures)}] reviewing {fixture_id} ({fixture['student_color']})")
            config.player_name = fixture["student_name"]
            pgn_path = manifest_path.parent / fixture["pgn"]
            review = review_game(
                pgn_path.read_text(encoding="utf-8"), engines, config,
                agenda_size=args.agenda_size,
                with_maia=not args.without_maia,
                enrich=True,
            )
            packet = {
                "fixture_id": fixture_id,
                "generated_on": date.today().isoformat(),
                "review_status": "proposal_only_pending_human",
                "purposes": fixture["purposes"],
                "source_sha256": fixture["sha256"],
                "generation_policy": {
                    "stockfish": {str(key): str(value) for key, value in
                                  (getattr(engines.stockfish, "id", {}) or {}).items()},
                    "review_eval_time": config.review_eval_time,
                    "agenda_size": args.agenda_size,
                    "agenda_best_depth": config.agenda_best_depth,
                    "agenda_best_time": config.agenda_best_time,
                    "agenda_punishment_depth": config.agenda_punishment_depth,
                    "agenda_punishment_time": config.agenda_punishment_time,
                    "maia_enabled": not args.without_maia,
                    "maia_deep_model": config.maia3_deep_model if not args.without_maia else None,
                    "online_enrichment": False,
                },
                "review": review_proposal_packet(review),
                "human_decision": {
                    "approve_fixture": None,
                    "reviewed_by": None,
                    "reviewed_on": None,
                    "agenda_must_include": [],
                    "agenda_must_exclude": [],
                    "diagnosis_by_ply": {},
                    "key_reason_tokens_by_ply": {},
                    "together_plies": [],
                    "separate_plies": [],
                    "slide_count": None,
                    "notes": "",
                },
            }
            target = output_dir / f"{fixture_id}.proposal.json"
            target.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    finally:
        engines.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
