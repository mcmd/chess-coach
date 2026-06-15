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
    assert result["student_move_label"] == "1. e4"
    assert result["best_move_label"] == "1. e4"


def test_agenda_answer_returns_numbered_best_move_label(stub_config):
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    engines = FakeEngines(multipv={fen: [("c3e4", 300), ("e2e4", 0)]})
    tools, state = make_tools(stub_config, engines)
    from app.review import GameReview, MoveRecord
    state.review = GameReview(headers={}, records=[
        MoveRecord(
            ply=38, number=20, color="white", san="Nxb5", uci="c3b5",
            fen_before=fen, fen_after=fen, eval_before_cp=300, eval_after_cp=0,
            drop_pct=17.0, quality="inaccuracy", best_move_san="Ne4",
            explanation_primitives={"best_move": "Ne4", "best_move_label": "20. Ne4"},
        )
    ], student_color="white", agenda_plies=[0])
    result = tools.evaluate_student_answer(fen, "Ne4")
    assert result["best_move"] == "Ne4"
    assert result["best_move_label"] == "20. Ne4"
    assert result["moment_move_label"] == "20. Nxb5"


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


def test_candidate_list_requires_commitment_before_grading(stub_config):
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    engines = FakeEngines(multipv={fen: [("c3e4", 300), ("e2e4", 0)]})
    tools, _ = make_tools(stub_config, engines)
    result = tools.evaluate_student_answer(fen, "I could do Qe4, or Ne4?")
    assert result["verdict"] == "needs_commitment"
    assert result["candidate_moves"] == ["Qe4", "Ne4"]
    assert result["reveal_allowed"] is False


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


def test_diagnosis_rejects_boilerplate_override_when_mismatched(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    result = tools.submit_diagnosis(fen, wrong, "x", "y", override_reason="Matches backend recommendation")
    assert "error" in result
    assert state.diagnostics == []


# --- submit_diagnosis: explanation grounding ---

M17_FEN = "r4rk1/p4ppp/n1pq4/1p2p3/1P1PP1n1/P1N2N2/4QPPP/R4RK1 w - - 0 17"


def _m17_tools(stub_config):
    """The live-session moment where the model fabricated 'cxb4' in a diagnosis."""
    from app.review import GameReview, MoveRecord
    board = chess.Board(M17_FEN)
    board.push_san("d5")
    record = MoveRecord(
        ply=32, number=17, color="white", san="d5", uci="d4d5",
        fen_before=M17_FEN, fen_after=board.fen(), eval_before_cp=50,
        eval_after_cp=-30, drop_pct=10.4, quality="inaccuracy",
        best_move_san="Rad1", is_capture=False,
        best_line=["Rad1", "Rfe8"], punishment_line=["cxd5", "exd5", "Nxd5"],
    )
    tools, state = make_tools(stub_config)
    state.review = GameReview(headers={}, records=[record],
                              student_color="white", agenda_plies=[0])
    return tools, state, record


def test_diagnosis_rejects_fabricated_move_in_explanation(stub_config):
    tools, state, record = _m17_tools(stub_config)
    result = tools.submit_diagnosis(
        record.fen_before, "missed_fork",
        "The student played d5, which allowed Black to play cxb4, forking the a3 pawn and the c3 knight.",
        "Name the urgent priority first.")
    assert "error" in result
    assert "cxb4" in result["error"]
    assert result["verified_punishment_line"] == ["cxd5", "exd5", "Nxd5"]
    assert state.diagnostics == []


def test_diagnosis_accepts_explanation_built_from_verified_lines(stub_config):
    tools, state, record = _m17_tools(stub_config)
    result = tools.submit_diagnosis(
        record.fen_before, "calculation_error",
        "The student played d5, allowing cxd5; Rad1 instead adds a defender to the d4 pawn.",
        "Count defenders before pushing.")
    assert result["success"] is True
    assert "cxd5" in state.diagnostics[0]["error_explanation"]


def test_diagnosis_second_fabrication_logs_backend_explanation(stub_config):
    tools, state, record = _m17_tools(stub_config)
    bad = "It allowed Black to play cxb4, forking the knight."
    assert "error" in tools.submit_diagnosis(record.fen_before, "missed_fork", bad, "habit")
    result = tools.submit_diagnosis(record.fen_before, "missed_fork", bad, "habit")
    assert result["success"] is True
    logged = state.diagnostics[0]["error_explanation"]
    assert "cxb4" not in logged
    assert "cxd5" in logged


def test_weak_diagnosis_text_is_replaced_with_backend_sentence(stub_config):
    tools, state, record = _m17_tools(stub_config)
    record.explanation_primitives = {
        "diagnosis_sentence": "Played 17. d5 before improving the pieces; 17. Rad1 supported the center first."
    }
    result = tools.submit_diagnosis(
        record.fen_before, "calculation_error",
        "The student allowed Black to launch a devastating attack starting with",
        "Name the urgent priority first.")
    assert result["success"] is True
    logged = state.diagnostics[0]["error_explanation"]
    assert logged == record.explanation_primitives["diagnosis_sentence"]
    assert not logged.endswith("with")


def test_diagnosis_allows_naming_the_opponents_prevented_threat(stub_config):
    from app.review import GameReview, MoveRecord
    fen = "6k1/8/8/8/8/Q6q/6P1/6K1 w - - 0 1"
    board = chess.Board(fen)
    board.push_san("Qxh3")
    record = MoveRecord(
        ply=0, number=1, color="white", san="Qxh3", uci="a3h3",
        fen_before=fen, fen_after=board.fen(), eval_before_cp=0,
        eval_after_cp=900, drop_pct=0.0, quality="good",
        best_move_san="Qxh3", is_capture=True)
    tools, state = make_tools(stub_config)
    state.review = GameReview(headers={}, records=[record],
                              student_color="white", agenda_plies=[0])
    # Qxg2# is legal for neither side after the move — it is the threat the
    # move PREVENTED, only visible with the turn flipped in the before-position.
    result = tools.submit_diagnosis(
        fen, "ignored_threat",
        "Black threatened Qxg2#, and Qxh3 removed the attacking queen.",
        "Scan checks first.")
    assert result["success"] is True


# --- _valid_diagnosis_override: evidence, not phrasing ---

def test_override_reason_arguing_against_recommendation_is_valid():
    from app.tools import _valid_diagnosis_override
    reason = ("This was not primarily a bad trade: the student ignored the forcing "
              "check Qxg4+, which they admitted they never considered.")
    assert _valid_diagnosis_override(reason, "bad_trade", "ignored_threat") is True


def test_override_reason_mentioning_override_is_valid():
    from app.tools import _valid_diagnosis_override
    reason = ("Overriding the backend recommendation: the student never looked "
              "at the opponent's forcing check during the discussion.")
    assert _valid_diagnosis_override(reason, "bad_trade", "ignored_threat") is True


def test_override_reason_pure_category_talk_is_invalid():
    from app.tools import _valid_diagnosis_override
    assert _valid_diagnosis_override(
        "override to ignored_threat from bad_trade", "bad_trade", "ignored_threat") is False


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


def test_analyze_position_exposes_wdl_risk_profile(stub_config):
    # maia3-style WDL (with draw/loss) → analyze_position should surface a queryable
    # risk verdict so the coach can answer "how risky is this / drawing chances?".
    engines = FakeEngines(maia={START: {"e2e4": 1.0}}, maia_wdl={START: (0.2, 0.6, 0.2)})
    tools, _ = make_tools(stub_config, engines)
    summary = tools.analyze_position(START)["human_tendency_at_student_elo"]
    assert summary["risk_profile"] == "drawish"
    assert summary["win_draw_loss_for_side_to_move"] == {"win": 0.2, "draw": 0.6, "loss": 0.2}


def test_analyze_position_no_wdl_for_maia2_style(stub_config):
    # maia2-style prediction (no draw/loss) → no risk fields, no crash.
    engines = FakeEngines(maia={START: {"e2e4": 1.0}})
    tools, _ = make_tools(stub_config, engines)
    summary = tools.analyze_position(START)["human_tendency_at_student_elo"]
    assert "risk_profile" not in summary
    assert "win_probability_for_side_to_move" in summary


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


# --- LESSON STATE FIXES: completion + no overwrites outside agenda moments ---

def test_final_diagnosis_marks_lesson_complete(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    from app.memory import load_lesson_metadata
    tools, state = make_tools(stub_config, _scripted_engines())
    from app.review import review_game
    state.review = review_game(RPGN, _scripted_engines(), stub_config)
    state.current_game_id = "testgame1"
    record = state.review.agenda[-1]  # the ONLY agenda moment here
    result = tools.submit_diagnosis(record.fen_before, record.recommended_diagnosis, "x", "y")
    assert result.get("success")
    assert state.lesson_metadata["lesson_status"] == "lesson_complete"
    stored = load_lesson_metadata(stub_config.lesson_file)
    assert stored["lesson_status"] == "lesson_complete"
    assert not tools._has_resume_metadata("testgame1")  # nothing to resume into


def test_drill_grading_does_not_touch_lesson_state(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    state.lesson_metadata = {"last_game_id": "g1", "last_agenda_index": 2,
                             "lesson_status": "awaiting_diagnosis"}
    # Grading a position that is NOT an agenda moment (no review loaded at all).
    tools.evaluate_student_answer(START, "e4")
    assert state.lesson_metadata["lesson_status"] == "awaiting_diagnosis"
    assert state.lesson_metadata["last_agenda_index"] == 2


# --- THREAT-ANSWER GRADING ---

THREAT_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3"


def _threat_engines():
    return FakeEngines(threats={THREAT_FEN: [
        {"threat": "mate in 1 if you do nothing", "threat_line": "Qh4#"},
    ]})


def test_correct_threat_identified(stub_config):
    tools, _ = make_tools(stub_config, _threat_engines())
    result = tools.evaluate_threat_answer(THREAT_FEN, "Qh4")
    assert result["verdict"] == "correct_threat"
    assert result["threat_line"] == "Qh4#"
    assert result["reveal_allowed"] is True


def test_wrong_threat_withheld_then_revealed(stub_config):
    tools, _ = make_tools(stub_config, _threat_engines())
    first = tools.evaluate_threat_answer(THREAT_FEN, "a5")
    assert first["verdict"] == "not_a_real_threat"
    assert first["reveal_allowed"] is False
    assert "actual_threats" not in first
    second = tools.evaluate_threat_answer(THREAT_FEN, "Nc6")
    assert second["attempts_so_far"] == 2
    assert second["reveal_allowed"] is True
    assert second["actual_threats"][0]["threat_line"] == "Qh4#"


def test_illegal_threat_counts_attempt(stub_config):
    tools, _ = make_tools(stub_config, _threat_engines())
    result = tools.evaluate_threat_answer(THREAT_FEN, "Qh5")  # illegal for Black here
    assert result["verdict"] == "not_a_legal_opponent_move"
    assert result["attempts_so_far"] == 1


def test_no_threat_position_says_so(stub_config):
    tools, _ = make_tools(stub_config, FakeEngines())
    result = tools.evaluate_threat_answer(START, "e5")
    assert result["verdict"] == "no_significant_threat_exists"
    assert result["reveal_allowed"] is True


# --- predict_human_move retired ---

def test_predict_human_move_is_retired(stub_config):
    tools, _ = make_tools(stub_config)
    assert not hasattr(tools, "predict_human_move")
    names = [t.__name__ for t in tools.get_tool_list()]
    assert "predict_human_move" not in names
    assert "evaluate_threat_answer" in names


def test_wrong_threat_includes_verified_comparison(stub_config):
    tools, _ = make_tools(stub_config, _threat_engines())
    result = tools.evaluate_threat_answer(THREAT_FEN, "a5")
    comparison = result["comparison"]
    assert comparison["engine_main_threat"] == "mate in 1 if you do nothing"
    assert comparison["engine_main_threat_line"] == "Qh4#"
    assert comparison["student_named_move_is_legal"] is True
    assert comparison["your_named_threat_gain_for_opponent"] == "+0.00"
    assert "not the verified main threat" in comparison["secondary_threat_guidance"]
    assert "NEVER invent a geometric reason" in comparison["note"]
