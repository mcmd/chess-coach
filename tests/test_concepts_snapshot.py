"""Snapshot regression test for the concept extractor.

The committed baseline (concepts_snapshot.json) pins the full feature output
for a set of representative positions. Regenerate it intentionally after a
deliberate behavior change with:

    python -m tests.test_concepts_snapshot --save
"""
import json
import os

import chess
import pytest

from app.concepts import extract_board_features

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # Sicilian
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",  # Italian
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Complex
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",  # Simple Endgame
    "2r3k1/1p3p1p/p2p2p1/3P4/P1P1R3/1P6/5PPP/6K1 b - - 0 24",  # Rook endgame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",  # Four Knights
]

SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "concepts_snapshot.json")


def _current_snapshots() -> dict:
    return {fen: json.loads(json.dumps(extract_board_features(chess.Board(fen)))) for fen in FENS}


@pytest.mark.skipif(not os.path.exists(SNAPSHOT_FILE),
                    reason="baseline missing — run: python -m tests.test_concepts_snapshot --save")
def test_concept_features_match_snapshot():
    with open(SNAPSHOT_FILE, "r") as f:
        baseline = json.load(f)
    current = _current_snapshots()
    for fen in FENS:
        assert fen in baseline, f"FEN missing from baseline: {fen}"
        assert current[fen] == baseline[fen], f"Feature regression for FEN: {fen}"


def _save():
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(_current_snapshots(), f, indent=4)
    print(f"Saved baseline snapshots to {SNAPSHOT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save current output as baseline")
    args = parser.parse_args()
    if args.save:
        _save()
    else:
        parser.print_help()
