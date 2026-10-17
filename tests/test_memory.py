import json
from types import SimpleNamespace

from app.drills import record_drill_outcome, record_drill_presented
from app.memory import (drill_practice_summary, load_dossier, record_self_assessment,
                        record_session_takeaway, update_dossier,
                        update_dossier_with_batch, save_history)


def test_dossier_aggregates_taxonomy_categories(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    diagnostics = [
        {"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "Count defenders.", "evidence_version": 1},
        {"fen": "f2", "error_category": "hanging_piece", "preventative_habit": "Count defenders.", "evidence_version": 1},
        {"fen": "f3", "error_category": "ignored_threat", "preventative_habit": "Ask what changed.", "evidence_version": 1},
    ]
    update_dossier(diagnostics, dossier_file)
    d = load_dossier(dossier_file)
    assert d["total_blunders_analyzed"] == 3
    assert d["error_frequencies"]["hanging_piece"] == 2
    assert d["error_frequencies"]["ignored_threat"] == 1
    assert len(d["recent_positions"]) == 3
    assert d["recent_positions"][0]["fen"] == "f3"  # most recent first

    # Aggregation must survive a second session (the F2 fix).
    update_dossier([{"fen": "f4", "error_category": "hanging_piece", "preventative_habit": "x",
                     "evidence_version": 1}], dossier_file)
    d = load_dossier(dossier_file)
    assert d["error_frequencies"]["hanging_piece"] == 3


def test_recent_positions_capped_at_20(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    diagnostics = [{"fen": f"f{i}", "error_category": "missed_tactic", "preventative_habit": "h"} for i in range(30)]
    update_dossier(diagnostics, dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["recent_positions"]) == 20


def test_load_dossier_missing_file_returns_defaults(tmp_path):
    d = load_dossier(str(tmp_path / "nope.json"))
    assert d["total_blunders_analyzed"] == 0
    assert d["recent_positions"] == []


class _FakeChat:
    def __init__(self, messages):
        self._messages = messages

    def get_history(self):
        return self._messages


def _msg(role, text):
    return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])


def test_save_history_strips_think_blocks(tmp_path):
    history_file = str(tmp_path / "hist.json")
    chat = _FakeChat([
        _msg("user", "hello"),
        _msg("model", "<think>secret scratchpad</think>The d4 pawn is undefended."),
        _msg("model", "<think>only scratchpad</think>"),
    ])
    save_history(chat, history_file)
    data = json.load(open(history_file))
    assert len(data) == 2  # think-only message dropped entirely
    assert data[1]["parts"][0]["text"] == "The d4 pawn is undefended."
    assert "secret" not in json.dumps(data)


def test_save_history_caps_length(tmp_path):
    history_file = str(tmp_path / "hist.json")
    chat = _FakeChat([_msg("user", f"msg {i}") for i in range(200)])
    save_history(chat, history_file)
    data = json.load(open(history_file))
    assert len(data) == 80
    assert data[-1]["parts"][0]["text"] == "msg 199"


def test_typicality_trend_recorded_and_capped(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    update_dossier([
        {"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "h",
         "error_typicality": "typical_for_level"},
        {"fen": "f2", "error_category": "king_safety", "preventative_habit": "h",
         "error_typicality": "atypical_for_level"},
        {"fen": "f3", "error_category": "missed_tactic", "preventative_habit": "h"},
    ], dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["typicality_history"]) == 1
    entry = d["typicality_history"][0]
    assert entry["typical_errors"] == 1
    assert entry["atypical_errors"] == 1
    assert entry["total_diagnosed"] == 3
    assert d["recent_positions"][2]["error_typicality"] == "typical_for_level"

    # A second session appends a second entry.
    update_dossier([{"fen": "f4", "error_category": "hanging_piece", "preventative_habit": "h",
                     "error_typicality": "atypical_for_level"}], dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["typicality_history"]) == 2


def test_dossier_context_shows_typicality_trend(tmp_path):
    from app.prompts import get_dossier_context
    dossier_file = str(tmp_path / "dossier.json")
    update_dossier([{"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "h",
                     "error_typicality": "typical_for_level", "evidence_version": 1}], dossier_file)
    context = get_dossier_context(dossier_file)
    assert "Error-typicality trend" in context
    assert "1 peer-common" in context


def test_session_takeaway_recorded_and_added_to_context(tmp_path):
    from app.prompts import get_dossier_context
    dossier_file = str(tmp_path / "dossier.json")
    record_session_takeaway(dossier_file, {
        "game_id": "g1",
        "theme_id": "ignored_threat:opening",
        "text": "Session takeaway: a single ignored threat moment in the opening: 1. e4.",
        "source": "review_theme.safe_summary",
    })
    record_session_takeaway(dossier_file, {
        "game_id": "g1",
        "theme_id": "ignored_threat:opening",
        "text": "Session takeaway: a single ignored threat moment in the opening: 1. e4.",
        "source": "review_theme.safe_summary",
    })

    dossier = load_dossier(dossier_file)
    assert len(dossier["session_takeaways"]) == 1
    assert dossier["session_takeaways"][0]["theme_id"] == "ignored_threat:opening"
    context = get_dossier_context(dossier_file)
    assert "Recent closing takeaways" in context
    assert "ignored threat moment" in context


def test_self_assessment_history_recorded_and_added_to_context(tmp_path):
    from app.prompts import get_dossier_context
    dossier_file = str(tmp_path / "dossier.json")
    record_self_assessment(dossier_file, {
        "game_id": "g1",
        "result": "exact_top_agenda_match",
        "top_ply": 30,
        "top_move_label": "15. Re1",
        "matched_ply": 30,
        "matched_move_label": "15. Re1",
        "student_text": "I think 15. Re1 was the turning point.",
    })
    record_self_assessment(dossier_file, {
        "game_id": "g2",
        "result": "near_miss",
        "top_ply": 40,
        "top_move_label": "20. Rxe8",
        "matched_ply": 38,
        "matched_move_label": "19. h3",
        "student_text": "Maybe 19. h3.",
    })

    dossier = load_dossier(dossier_file)
    assert len(dossier["self_assessment_history"]) == 2
    assert dossier["self_assessment_history"][0]["game_id"] == "g2"
    context = get_dossier_context(dossier_file)
    assert "Self-assessment trend" in context
    assert "1/2 exact top-agenda matches" in context
    assert "1 near misses" in context


def test_drill_practice_attaches_to_batch_pattern_and_context(tmp_path):
    from app.prompts import get_dossier_context

    dossier_file = str(tmp_path / "dossier.json")
    drill = {
        "fen": "fen1",
        "error_category": "ignored_threat",
        "pattern_id": "ignored_threat:opening",
        "source": "recurring_pattern",
    }
    record_drill_presented(dossier_file, drill)
    record_drill_outcome(dossier_file, drill, {
        "verdict": "best",
        "success": True,
        "reveal_allowed": True,
        "attempts_so_far": 1,
    })
    update_dossier_with_batch(dossier_file, {
        "batch_id": "b1",
        "games": 5,
        "patterns": [{
            "id": "ignored_threat:opening",
            "title": "Ignored threat",
            "count": 3,
            "games_seen": 2,
            "habit": "Ask what the opponent threatened.",
            "rate": 0.6,
            "representative_positions": [],
        }],
    })

    dossier = load_dossier(dossier_file)
    pattern = dossier["recurring_patterns"][0]
    assert pattern["drill_practice"]["attempts"] == 1
    assert pattern["drill_practice"]["successes"] == 1
    summaries = drill_practice_summary(dossier)
    assert summaries[0]["title"] == "Ignored threat"
    context = get_dossier_context(dossier_file)
    assert "practice: 1/1 recent solved" in context
    assert "Drill practice trend" in context
    assert "practice evidence only" in context


def test_drill_only_dossier_context_is_not_empty(tmp_path):
    from app.prompts import get_dossier_context

    dossier_file = str(tmp_path / "dossier.json")
    drill = {
        "fen": "fen1",
        "error_category": "king_safety",
        "source": "recent_position",
    }
    record_drill_presented(dossier_file, drill)
    record_drill_outcome(dossier_file, drill, {
        "verdict": "wrong",
        "reveal_allowed": True,
        "attempts_so_far": 2,
    })

    context = get_dossier_context(dossier_file)

    assert "No historical data available" not in context
    assert "king safety practice: 0/1 recent solved" in context
    assert "last drill was a lapse" in context


# --- Quality-SOL §7.3 schema migration and §7.4 slice E supersession ---

# 2dtDqQgR before 31. Kh1, which repeated the position a third time (see tests/test_verify_chess.py).
FEN_BEFORE_31 = "1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31"
# A move whose draw by rule is visible from the FEN alone (the 100th quiet half-move). An old entry for
# 2dtDqQgR's 31. Kh1 stores only its FEN, which cannot show the repetition, so it stays legacy_unverified.
DRAWN_FEN = "8/8/8/4k3/8/8/4K3/7R w - - 99 120"


def _fen_after(*sans):
    import chess
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def _schema_1_dossier():
    """Shaped like a dossier written before schema_version existed, including a diagnosis of a move that
    reached a draw by rule (as 31. Kh1 was wrongly logged ignored_threat)."""
    fen_a, fen_b = _fen_after("e4", "e5", "Nf3"), _fen_after("d4")
    return {
        "total_blunders_analyzed": 4,
        "error_frequencies": {"ignored_threat": 3, "hanging_piece": 1},
        "recent_habits": ["Ask what the opponent threatens."],
        "recent_positions": [
            {"fen": DRAWN_FEN, "played": "Rh2", "error_category": "ignored_threat",
             "error_typicality": "typical_for_level"},
            {"fen": fen_a, "error_category": "ignored_threat"},
            {"fen": fen_b, "error_category": "hanging_piece", "maia_played_prob": 0.4},
        ],
        "drill_history": [{"date": "2026-10-01", "fen": fen_a, "error_category": "ignored_threat"}],
        "drill_outcomes": [{"date": "2026-10-01", "fen": fen_a, "error_category": "ignored_threat",
                            "success": True, "verdict": "best"}],
        "drill_state": {"ignored_threat": {"seen_count": 1, "last_reviewed": "2026-10-01"}},
        "recurring_patterns": [{
            "id": "ignored_threat:middlegame", "title": "Ignored threat", "status": "active",
            "evidence_count": 3, "games_seen": 2,
            "representative_positions": [{"fen": DRAWN_FEN, "played": "Rh2"}, {"fen": fen_a}],
        }],
        "drill_sets": [{
            "pattern_id": "ignored_threat:middlegame", "title": "Ignored threat", "status": "active",
            "positions": [{"fen": DRAWN_FEN, "error_category": "ignored_threat", "played": "Rh2"},
                          {"fen": fen_a, "error_category": "ignored_threat"}],
        }],
        "typicality_history": [{"date": "2026-10-01", "typical_errors": 1, "atypical_errors": 0,
                                "total_diagnosed": 1}],
        "a_field_from_elsewhere": {"keep": True},
    }


def test_old_dossier_migrates_once_keeping_history_and_superseding_the_drawn_move(tmp_path):
    from app.drills import select_dossier_drill
    from app.prompts import get_dossier_context
    path = tmp_path / "dossier.json"
    old = _schema_1_dossier()
    path.write_text(json.dumps(old), encoding="utf-8")

    d = load_dossier(str(path))

    assert d["schema_version"] == 2
    assert json.loads((tmp_path / "dossier.json.v1.bak").read_text(encoding="utf-8")) == old
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    for key in ("recent_habits", "drill_history", "drill_outcomes", "drill_state", "typicality_history",
                "a_field_from_elsewhere"):
        assert d[key] == old[key]
    assert [p["fen"] for p in d["recent_positions"]] == [p["fen"] for p in old["recent_positions"]]
    assert all(p["evidence_status"] == "legacy_unverified" for p in d["recent_positions"])
    drawn = d["recent_positions"][0]
    assert drawn["superseded"]["draw_rule"] == "fifty_move_rule"
    assert "superseded" not in d["recent_positions"][1]
    # Old totals are kept as recorded and labelled legacy; none become verified counts.
    assert d["total_blunders_analyzed"] == 0 and d["error_frequencies"] == {}
    legacy = d["legacy_totals"]
    assert legacy["evidence_status"] == "legacy_unverified"
    assert legacy["as_recorded"] == {"total_blunders_analyzed": 4, "error_frequencies": old["error_frequencies"]}
    assert legacy["error_frequencies"] == {"ignored_threat": 2, "hanging_piece": 1}
    assert d["recurring_patterns"][0]["superseded_moments"] == 1

    for _ in range(3):
        drill = select_dossier_drill(d)
        assert drill["fen"] != DRAWN_FEN
        d["drill_history"].insert(0, {"fen": drill["fen"], "error_category": drill["error_category"]})
    context = get_dossier_context(str(path))
    assert "legacy, unverified" in context
    assert "ignored_threat (2x)" in context and "ignored_threat (3x)" not in context
    assert "Ignored threat (2 moments" in context

    # Reloading is a no-op: nothing is rewritten and the backup is not replaced.
    migrated = path.read_bytes()
    backup = (tmp_path / "dossier.json.v1.bak").read_bytes()
    load_dossier(str(path))
    assert path.read_bytes() == migrated
    assert (tmp_path / "dossier.json.v1.bak").read_bytes() == backup

    # Saving keeps fields this code does not know.
    update_dossier([{"fen": _fen_after("c4"), "error_category": "hanging_piece", "preventative_habit": "h",
                     "evidence_version": 1, "game_id": "g9", "ply": 0, "student_side": "white"}], str(path))
    d = load_dossier(str(path))
    assert d["a_field_from_elsewhere"] == {"keep": True}
    assert d["error_frequencies"] == {"hanging_piece": 1}
    assert d["legacy_totals"]["error_frequencies"] == {"ignored_threat": 2, "hanging_piece": 1}


def test_a_newer_schema_dossier_is_read_but_never_overwritten(tmp_path):
    path = tmp_path / "dossier.json"
    path.write_text(json.dumps({"schema_version": 99, "recent_positions": [], "future": [1, 2]}), encoding="utf-8")
    original = path.read_bytes()

    assert load_dossier(str(path))["future"] == [1, 2]
    assert update_dossier([{"fen": "f1", "error_category": "hanging_piece", "evidence_version": 1}],
                          str(path)) is False
    record_session_takeaway(str(path), {"game_id": "g1", "text": "Check forcing replies."})
    assert path.read_bytes() == original
    assert not (tmp_path / "dossier.json.v99.bak").exists()


def _diag(game_id, ply=20, **extra):
    return {"fen": FEN_BEFORE_31.replace("8 31", f"8 {ply}"), "error_category": "ignored_threat",
            "preventative_habit": "Ask what the opponent threatens.", "evidence_version": 1,
            "game_id": game_id, "ply": ply, "student_side": "white", **extra}


def test_teaching_the_same_moment_twice_is_idempotent_but_another_game_counts(tmp_path):
    path = str(tmp_path / "dossier.json")
    update_dossier([_diag("g1")], path)
    first = load_dossier(path)
    update_dossier([_diag("g1")], path)
    again = load_dossier(path)
    assert again["total_blunders_analyzed"] == 1
    assert len(again["diagnosis_events"]) == 1
    assert again["typicality_history"] == first["typicality_history"]
    assert again["recent_positions"] == first["recent_positions"]

    update_dossier([_diag("g2")], path)
    assert load_dossier(path)["error_frequencies"] == {"ignored_threat": 2}


def test_a_diagnosis_of_a_drawn_move_is_kept_superseded_and_not_counted(tmp_path):
    path = str(tmp_path / "dossier.json")
    update_dossier([_diag("g1", ply=60, draw_rule="threefold_repetition"), _diag("g1", ply=40)], path)
    d = load_dossier(path)
    drawn = next(e for e in d["diagnosis_events"] if e["ply"] == 60)
    assert drawn["superseded"]["draw_rule"] == "threefold_repetition"
    assert d["error_frequencies"] == {"ignored_threat": 1}
    assert [p["ply"] for p in d["recent_positions"]] == [40]


def test_dossier_context_calls_categories_provisional_lesson_labels(tmp_path):
    from app.prompts import get_dossier_context
    path = str(tmp_path / "dossier.json")
    update_dossier([_diag("g1", cause_basis="unknown", student_cause=None)], path)
    context = get_dossier_context(path)
    assert "Provisional lesson labels" in context and "not established weaknesses" in context
    assert "ignored_threat (1x)" in context
    assert "the student said" not in context

    update_dossier([_diag("g2", cause_basis="student_statement", student_cause="I didn't see the queen check",
                          student_evidence={"turn_id": 4, "quote": "I didn't see the queen check"})], path)
    assert 'ignored_threat: the student said "I didn\'t see the queen check"' in get_dossier_context(path)
