import json
from datetime import date, timedelta

import chess

from app.drills import (popularity_branch, record_drill_outcome,
                        record_drill_presented, retrievability,
                        reverify_drill_position,
                        select_dossier_drill)
from app.memory import load_dossier
from tests.conftest import FakeEngines


def _after(*sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def test_drill_selector_prioritizes_recurring_pattern_and_varies_instance():
    fen_seen = chess.STARTING_FEN
    fen_fresh = _after("e4")
    dossier = {
        "recent_positions": [
            {"fen": fen_seen, "error_category": "ignored_threat"},
            {"fen": fen_fresh, "error_category": "ignored_threat"},
            {"fen": _after("d4"), "error_category": "hanging_piece"},
        ],
        "recurring_patterns": [
            {"id": "ignored_threat:opening", "title": "Ignored threat", "status": "active",
             "evidence_count": 4},
        ],
        "drill_history": [{"fen": fen_seen, "error_category": "ignored_threat"}],
        "error_frequencies": {"hanging_piece": 10, "ignored_threat": 2},
    }

    drill = select_dossier_drill(dossier)

    assert drill["fen"] == fen_fresh
    assert drill["error_category"] == "ignored_threat"
    assert drill["pattern_id"] == "ignored_threat:opening"
    assert drill["source"] == "recurring_pattern"


def test_drill_selector_uses_batch_drill_sets_without_recent_positions():
    fen = _after("e4")
    dossier = {
        "recent_positions": [],
        "drill_sets": [{
            "pattern_id": "ignored_threat:opening",
            "title": "Ignored threat",
            "status": "active",
            "positions": [{
                "fen": fen,
                "error_category": "ignored_threat",
                "pattern_id": "ignored_threat:opening",
                "pattern_title": "Ignored threat",
                "source": "batch_drill_set",
                "drill_type": "defend",
                "drill_goal": "Find the defensive move.",
            }],
        }],
        "recurring_patterns": [
            {"id": "ignored_threat:opening", "title": "Ignored threat", "status": "active",
             "evidence_count": 4},
        ],
    }

    drill = select_dossier_drill(dossier)

    assert drill["fen"] == fen
    assert drill["pattern_id"] == "ignored_threat:opening"
    assert drill["source"] == "batch_drill_set"
    assert drill["drill_type"] == "defend"
    assert drill["drill_goal"] == "Find the defensive move."


def test_drill_selector_prioritizes_slipping_pattern():
    dossier = {
        "recent_positions": [
            {"fen": chess.STARTING_FEN, "error_category": "ignored_threat"},
            {"fen": _after("e4"), "error_category": "king_safety"},
        ],
        "recurring_patterns": [
            {"id": "ignored_threat:opening", "title": "Ignored threat", "status": "active",
             "evidence_count": 10},
            {"id": "king_safety:opening", "title": "King safety", "status": "slipping",
             "evidence_count": 2},
        ],
    }

    drill = select_dossier_drill(dossier)

    assert drill["error_category"] == "king_safety"


def test_drill_selector_prefers_due_theme_over_fresh_theme():
    today = date.today()
    dossier = {
        "recent_positions": [
            {"fen": chess.STARTING_FEN, "error_category": "ignored_threat"},
            {"fen": _after("e4"), "error_category": "king_safety"},
        ],
        "recurring_patterns": [
            {"id": "ignored_threat:opening", "title": "Ignored threat", "status": "active",
             "evidence_count": 4},
            {"id": "king_safety:opening", "title": "King safety", "status": "active",
             "evidence_count": 4},
        ],
        "drill_state": {
            "ignored_threat": {"last_reviewed": today.isoformat(), "stability_days": 10.0},
            "king_safety": {"last_reviewed": (today - timedelta(days=10)).isoformat(),
                            "stability_days": 1.0},
        },
    }

    drill = select_dossier_drill(dossier)

    assert drill["error_category"] == "king_safety"
    assert drill["retrievability"] < 0.01


def test_drill_selector_skips_invalid_stale_fens():
    dossier = {
        "recent_positions": [
            {"fen": "not a fen", "error_category": "ignored_threat"},
            {"fen": chess.STARTING_FEN, "error_category": "hanging_piece"},
        ],
        "error_frequencies": {"ignored_threat": 10, "hanging_piece": 1},
    }

    drill = select_dossier_drill(dossier)

    assert drill["fen"] == chess.STARTING_FEN
    assert drill["error_category"] == "hanging_piece"


def test_drill_selector_skips_positions_rejected_by_verifier():
    dossier = {
        "recent_positions": [
            {"fen": chess.STARTING_FEN, "error_category": "ignored_threat"},
            {"fen": _after("e4"), "error_category": "hanging_piece"},
        ],
        "error_frequencies": {"ignored_threat": 10, "hanging_piece": 1},
    }

    drill = select_dossier_drill(
        dossier,
        verifier=lambda pos: None if pos["error_category"] == "ignored_threat"
        else {"engine_verified": True},
    )

    assert drill["error_category"] == "hanging_piece"
    assert drill["verification"] == {"engine_verified": True}


def test_reverify_drill_position_records_local_engine_metadata():
    engines = FakeEngines(evals={chess.STARTING_FEN: 42}, bests={chess.STARTING_FEN: "e2e4"})

    verification = reverify_drill_position({"fen": chess.STARTING_FEN}, engines)

    assert verification["engine_verified"] is True
    assert verification["source"] == "local_quick_eval"
    assert verification["eval_cp_white"] == 42
    assert verification["best_move_san"] == "e4"
    assert "not narrate" in verification["usage"]


def test_popularity_branch_uses_probability_not_severity():
    assert "common" in popularity_branch(0.62)
    assert "fairly common" in popularity_branch(0.35)
    assert popularity_branch(0.12) == ""
    assert popularity_branch(None) == ""


def test_record_drill_presented_updates_history_and_state(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    drill = {
        "fen": chess.STARTING_FEN,
        "error_category": "king_safety",
        "pattern_id": "king_safety:middlegame",
        "drill_type": "defend",
        "source": "recurring_pattern",
    }

    record_drill_presented(dossier_file, drill)

    dossier = load_dossier(dossier_file)
    assert dossier["drill_history"][0]["fen"] == chess.STARTING_FEN
    assert dossier["drill_history"][0]["pattern_id"] == "king_safety:middlegame"
    assert dossier["drill_history"][0]["drill_type"] == "defend"
    assert dossier["drill_state"]["king_safety"]["seen_count"] == 1
    assert dossier["drill_state"]["king_safety"]["last_fen"] == chess.STARTING_FEN


def test_record_drill_outcome_updates_memory_state_after_success(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    drill = {
        "fen": chess.STARTING_FEN,
        "error_category": "king_safety",
        "pattern_id": "king_safety:middlegame",
        "drill_type": "defend",
        "source": "recurring_pattern",
    }
    record_drill_presented(dossier_file, drill)

    entry = record_drill_outcome(dossier_file, drill, {
        "verdict": "best",
        "reveal_allowed": True,
        "attempts_so_far": 1,
        "student_move": "e4",
        "win_pct_lost_vs_best": 0.0,
    })

    dossier = load_dossier(dossier_file)
    state = dossier["drill_state"]["king_safety"]
    assert entry["success"] is True
    assert dossier["drill_outcomes"][0]["verdict"] == "best"
    assert dossier["drill_outcomes"][0]["drill_type"] == "defend"
    assert dossier["drill_history"][0]["outcome"] == "success"
    assert state["success_count"] == 1
    assert state["stability_days"] == 2.0
    assert state["retrievability"] == 1.0


def test_record_drill_outcome_updates_memory_state_after_lapse(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    drill = {
        "fen": chess.STARTING_FEN,
        "error_category": "ignored_threat",
        "source": "recent_position",
    }
    record_drill_presented(dossier_file, drill)

    entry = record_drill_outcome(dossier_file, drill, {
        "verdict": "wrong",
        "reveal_allowed": True,
        "attempts_so_far": 2,
        "student_move": "Nf3",
        "win_pct_lost_vs_best": 20.0,
    })

    dossier = load_dossier(dossier_file)
    state = dossier["drill_state"]["ignored_threat"]
    assert entry["success"] is False
    assert dossier["drill_history"][0]["outcome"] == "lapse"
    assert state["lapse_count"] == 1
    assert state["difficulty"] > 5.0
    assert state["stability_days"] == 0.5
    assert state["retrievability"] == 0.0


def test_retrievability_decays_from_last_review_date():
    today = date(2026, 7, 6)
    assert retrievability({"last_reviewed": "2026-07-06", "stability_days": 10}, today=today) == 1.0
    assert retrievability({"last_reviewed": "2026-06-26", "stability_days": 1}, today=today) < 0.001


def test_old_dossier_loads_with_drill_defaults(tmp_path):
    dossier_file = tmp_path / "dossier.json"
    dossier_file.write_text(json.dumps({"total_blunders_analyzed": 0}), encoding="utf-8")

    dossier = load_dossier(str(dossier_file))

    assert dossier["drill_history"] == []
    assert dossier["drill_outcomes"] == []
    assert dossier["drill_state"] == {}
    assert dossier["drill_sets"] == []
