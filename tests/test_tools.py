import chess

from app.memory import load_lesson_metadata, save_lesson_metadata
from app.state import SessionState, TimelineManager
from app.tools import CoachTools, ERROR_TAXONOMY
from tests.conftest import FakeEngines
from tests.test_review import PGN as REVIEW_PGN, _scripted_engines

START = chess.STARTING_FEN


def make_tools(stub_config, engines=None):
    state = SessionState()
    timeline = TimelineManager(state)
    tools = CoachTools(state, timeline, engines or FakeEngines(), stub_config)
    return tools, state


# --- evaluate_student_answer: the Socratic state machine ---

def _answer_engines():
    # Best move e4 at +300; d4 close behind; Nf3 much worse.
    return FakeEngines(multipv={START: [("e2e4", 300), ("d2d4", 250), ("g1f3", 0)]})


def test_wrong_answer_withholds_best_move(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    result = tools.evaluate_student_answer(START, "Nf3")
    assert result["verdict"] == "wrong"
    assert result["attempts_so_far"] == 1
    assert result["reveal_allowed"] is False
    assert "best_move" not in result
    assert "note" in result


def test_second_failure_unlocks_reveal(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    tools.evaluate_student_answer(START, "Nf3")
    result = tools.evaluate_student_answer(START, "Nf3")
    assert result["attempts_so_far"] == 2
    assert result["reveal_allowed"] is True
    assert result["best_move"] == "e4"


def test_correct_answer_reveals_immediately(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    result = tools.evaluate_student_answer(START, "e4")
    assert result["verdict"] == "best"
    assert result["reveal_allowed"] is True
    assert result["best_move"] == "e4"


def test_near_miss_gets_graded_feedback(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    result = tools.evaluate_student_answer(START, "d4")
    assert result["verdict"] in ("excellent_alternative", "reasonable_but_imprecise")
    assert result["rank_among_engine_top5"] == 2


def test_illegal_answer_counts_as_attempt(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    result = tools.evaluate_student_answer(START, "Ke5")
    assert result["verdict"] == "illegal"
    assert result["attempts_so_far"] == 1


# --- submit_diagnosis: closed taxonomy (F2) ---

def test_diagnosis_rejects_free_text_category(stub_config):
    tools, state = make_tools(stub_config)
    result = tools.submit_diagnosis("fen1", "left the knight en prise lol", "x", "y")
    assert "error" in result
    assert result["allowed_categories"] == ERROR_TAXONOMY
    assert state.diagnostics == []


def test_diagnosis_accepts_taxonomy_and_dedupes(stub_config):
    tools, state = make_tools(stub_config)
    r1 = tools.submit_diagnosis("fen1", "hanging_piece", "Left the d4 knight undefended.", "Check defenders before every move.")
    assert r1["success"]
    assert state.diagnostics[0]["error_category"] == "hanging_piece"
    r2 = tools.submit_diagnosis("fen1", "missed_tactic", "dup", "dup")
    assert "already logged" in r2["message"]
    assert len(state.diagnostics) == 1


def test_diagnosis_rejects_mismatched_backend_recommendation_without_override(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    result = tools.submit_diagnosis(fen, wrong, "x", "y")
    assert "error" in result
    assert result["recommended_diagnosis"] == rec
    assert state.diagnostics == []


def test_diagnosis_allows_mismatched_backend_recommendation_with_override(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    result = tools.submit_diagnosis(fen, wrong, "x", "y", override_reason="Student explained a different root cause.")
    assert result["success"] is True
    assert "warning" in result
    assert state.diagnostics[0]["recommended_diagnosis"] == rec


# --- tablebase gate ---

def test_tablebase_refuses_middlegame_without_network(stub_config):
    tools, _ = make_tools(stub_config)
    result = tools.query_endgame_tablebase(START)
    assert "error" in result
    assert "7 pieces" in result["error"]


# --- analyze_position: the coaching packet ---

def test_analyze_position_returns_packet(stub_config):
    tools, state = make_tools(stub_config)
    result = tools.analyze_position(START)
    for key in ("top_stockfish_lines", "tactical_motifs_in_best_line",
                "threats_if_side_to_move_passes", "board_token_before_moves",
                "mathematically_legal_moves", "positional_and_safety_features"):
        assert key in result, key
    assert result["board_token_before_moves"] in state.board_visuals
    assert result["positional_and_safety_features"]["game_phase"] == "opening"


def test_analyze_position_rejects_illegal_hypothetical(stub_config):
    tools, _ = make_tools(stub_config)
    result = tools.analyze_position(START, moves_to_apply=["e4", "Ke7"])
    assert "error" in result
    assert "legal_moves_there" in result


# --- sparring: backend picks the move (F8) ---

def test_sparring_reply_requires_spar_mode(stub_config):
    tools, _ = make_tools(stub_config)
    assert "error" in tools.play_sparring_reply()


def test_sparring_reply_plays_maia_move(stub_config):
    engines = FakeEngines(maia={START: {"e2e4": 1.0}})
    tools, state = make_tools(stub_config, engines)
    state.sparring_mode = True
    result = tools.play_sparring_reply()
    assert result["coach_played"] == "e4"
    assert state.board.move_stack  # board actually advanced
    assert "Maia" in result["selected_by"]


def test_execute_coach_move_logic_tutor_still_guards(stub_config):
    tools, state = make_tools(stub_config)
    state.sparring_mode = True
    result = tools.execute_coach_move("Ke5")
    assert "Logic Tutor Intervention" in result["error"]
    assert state.board.fen() == START


def test_withheld_answer_includes_verified_hint_facts(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    result = tools.evaluate_student_answer(START, "Nf3")
    assert result["reveal_allowed"] is False
    facts = result["hint_facts"]
    assert facts["moving_piece"] == "pawn"          # best move is e4
    assert facts["is_a_capture"] is False
    assert facts["gives_check"] is False
    assert "best_move" not in result                # still withheld


# --- lesson resume/restart boundary ---

class _FakeResponse:
    status_code = 200
    text = REVIEW_PGN


def test_known_lichess_game_requires_resume_choice(stub_config):
    save_lesson_metadata(stub_config.lesson_file, {
        "last_game_id": "2dtDqQgR",
        "last_agenda_index": 0,
        "lesson_status": "awaiting_answer",
    })
    tools, state = make_tools(stub_config, _scripted_engines())
    result = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD")
    assert result["resume_choice_required"] is True
    assert result["game_id"] == "2dtDqQgR"
    assert state.review is None


def test_restart_known_lichess_game_loads_from_start(stub_config, monkeypatch):
    save_lesson_metadata(stub_config.lesson_file, {
        "last_game_id": "2dtDqQgR",
        "last_agenda_index": 0,
        "lesson_status": "awaiting_answer",
    })
    monkeypatch.setattr("app.tools.requests.get", lambda *args, **kwargs: _FakeResponse())
    tools, state = make_tools(stub_config, _scripted_engines())
    result = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD", mode="restart")
    assert "GAME REVIEW" in result["review_digest"]
    assert state.review is not None
    assert state.move_index == 0
    meta = load_lesson_metadata(stub_config.lesson_file)
    assert meta["last_game_id"] == "2dtDqQgR"
    assert meta["lesson_status"] == "fresh_review"


def test_continue_known_lichess_game_jumps_to_saved_agenda(stub_config, monkeypatch):
    save_lesson_metadata(stub_config.lesson_file, {
        "last_game_id": "2dtDqQgR",
        "last_agenda_index": 0,
        "lesson_status": "diagnosis_logged",
    })
    monkeypatch.setattr("app.tools.requests.get", lambda *args, **kwargs: _FakeResponse())
    tools, state = make_tools(stub_config, _scripted_engines())
    result = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD", mode="continue")
    assert result["resume"]["moment_number"] == 1
    assert result["resume"]["move_label"] == "3. Qxf7+"
    assert state.move_index == state.review.agenda_plies[0]
    meta = load_lesson_metadata(stub_config.lesson_file)
    assert meta["lesson_status"] == "awaiting_answer"


def test_explain_pawn_structure_evolution_requires_loaded_review(stub_config):
    tools, _ = make_tools(stub_config)
    result = tools.explain_pawn_structure_evolution()
    assert "error" in result


def test_explain_pawn_structure_evolution_returns_guardrails(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    result = tools.explain_pawn_structure_evolution("white")
    assert result["color"] == "white"
    assert "language_guardrails" in result
    assert "safe_language" in result
    assert "avoid_language" in result
    assert any("open file" in rule for rule in result["language_guardrails"])
