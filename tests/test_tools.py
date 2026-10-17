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
    # Slice C (26. f3 gives up the edge but stays level): "wrong" is not "losing".
    assert "does not lose" in result["outcome"]


def test_a_worse_move_that_leaves_the_student_losing_says_so(stub_config):
    tools, _ = make_tools(stub_config, FakeEngines(multipv={START: [("e2e4", 300), ("g1f3", -400)]}))
    result = tools.evaluate_student_answer(START, "Nf3")
    assert result["outcome"] == "You are worse after it."


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


def test_dossier_drill_answer_records_outcome(stub_config):
    from app.drills import record_drill_presented
    from app.memory import load_dossier

    tools, state = make_tools(stub_config, _answer_engines())
    drill = {
        "fen": START,
        "error_category": "ignored_threat",
        "pattern_id": "ignored_threat:opening",
        "source": "recurring_pattern",
    }
    record_drill_presented(stub_config.dossier_file, drill)
    state.active_dossier_drill = dict(drill)

    result = tools.evaluate_student_answer(START, "e4")

    assert result["drill_outcome_recorded"]["success"] is True
    assert state.active_dossier_drill == {}
    dossier = load_dossier(stub_config.dossier_file)
    assert dossier["drill_outcomes"][0]["error_category"] == "ignored_threat"
    assert dossier["drill_state"]["ignored_threat"]["success_count"] == 1


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


def test_submit_diagnosis_records_teaching_redundancy_ledger(stub_config):
    from app.review import GameReview, MoveRecord, ReviewTheme

    tools, state = make_tools(stub_config)
    rec = MoveRecord(ply=0, number=1, color="white", san="e4", uci="e2e4",
                     fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                     eval_before_cp=0, eval_after_cp=-100, drop_pct=12.0,
                     quality="inaccuracy", recommended_diagnosis="king_safety",
                     lesson_objective="protect_the_king")
    state.review = GameReview(headers={}, records=[rec], student_color="white",
                              agenda_plies=[0], themes=[
                                  ReviewTheme(
                                      id="king_safety:opening",
                                      label="king safety",
                                      plies=[0],
                                      confidence="low",
                                      safe_summary="king safety issue",
                                  )
                              ])

    result = tools.submit_diagnosis(
        chess.STARTING_FEN,
        "king_safety",
        "White weakened the king position.",
        "Check king safety before moving.",
    )

    assert result["success"]
    assert state.teaching_ledger["explained_moments"][0]["move_label"] == "1. e4"
    assert state.teaching_ledger["explained_moments"][0]["theme_ids"] == ["king_safety:opening"]
    assert {c["concept"] for c in state.teaching_ledger["explained_concepts"]} >= {
        "king_safety", "protect_the_king"
    }


def test_diagnosis_rejects_mismatched_backend_recommendation_without_override(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    tools.evaluate_explanation(fen, "not the key reason")
    tools.evaluate_explanation(fen, "still not the key reason")
    result = tools.submit_diagnosis(fen, wrong, "x", "y")
    assert "error" in result
    assert result["recommended_diagnosis"] == rec


def test_submit_diagnosis_rejects_played_bad_move_as_needed(stub_config):
    from app.review import GameReview, MoveRecord

    fen = "1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31"
    tools, state = make_tools(stub_config)
    rec = MoveRecord(
        ply=60, number=31, color="white", san="Kh1", uci="g1h1",
        fen_before=fen, fen_after=fen, eval_before_cp=0, eval_after_cp=-600,
        drop_pct=33.7, quality="blunder", best_move_san="Kh2",
        recommended_diagnosis="ignored_threat",
    )
    rec.explanation_primitives = {"diagnosis_sentence": "Played 31. Kh1; 31. Kh2 was the verified improvement."}
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    state.socratic_attempts[f"explain-complete:{fen}"] = True

    result = tools.submit_diagnosis(
        fen,
        "ignored_threat",
        "Played 31. Kh1 without stopping Qh3+; 31. Kh1 was needed to meet that forcing threat.",
        "Name the urgent priority before choosing a natural move.",
    )

    assert "error" in result
    assert "contradicts the verified agenda" in result["error"]
    assert result["played_move"] == "Kh1"
    assert result["best_move"] == "Kh2"
    assert state.diagnostics == []
    assert state.diagnostics == []


def test_diagnosis_allows_mismatched_backend_recommendation_with_override(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    tools.evaluate_explanation(fen, "not the key reason")
    tools.evaluate_explanation(fen, "still not the key reason")
    result = tools.submit_diagnosis(fen, wrong, "x", "y", override_reason="Student explained a different root cause.")
    assert result["success"] is True
    assert "warning" in result
    diag = state.diagnostics[0]
    # Quality-SOL §7.1: override prose is not the student's statement.
    assert diag["recommended_category"] == rec and diag["category_basis"] == "rule_based"
    assert diag["cause_basis"] == "unknown" and diag["student_cause"] is None


def test_a_logged_diagnosis_carries_its_evidence_and_no_inferred_cause(stub_config):
    # Quality-SOL §7.1: mechanism, assessment and the student's own words, from what the backend graded.
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN, source_game_id="g1")
    record = state.review.agenda[0]
    for text in ("not the key reason", "still not the key reason"):
        state.latest_user_text = text
        tools.evaluate_explanation(record.fen_before, text)

    assert tools.submit_diagnosis(record.fen_before, record.recommended_diagnosis, "x", "y")["success"] is True

    diag = state.diagnostics[0]
    assert (diag["game_id"], diag["ply"], diag["student_side"]) == ("g1", record.ply, record.color)
    assert diag["mechanism"]["kind"] and f"agenda_ply:{record.ply}" in diag["mechanism"]["evidence_ids"]
    assert diag["recommended_category"] == record.recommended_diagnosis
    assert diag["category_basis"] == "rule_based"
    assert diag["student_cause"] is None and diag["cause_basis"] == "unknown"
    assert diag["student_evidence"]["quote"] == "still not the key reason"
    assert diag["assessment"] == "revealed" and diag["independent_success"] is False
    assert diag["evidence_version"] == 1


def test_an_override_backed_by_the_students_own_words_records_their_statement(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())
    tools._review_and_load(REVIEW_PGN)
    fen = state.review.agenda[0].fen_before
    rec = state.review.agenda[0].recommended_diagnosis
    wrong = next(c for c in ERROR_TAXONOMY if c != rec)
    tools.evaluate_explanation(fen, "not the key reason")
    tools.evaluate_explanation(fen, "still not the key reason")
    state.latest_user_text = "Honestly I just didn't notice my queen could be taken."

    invented = tools.submit_diagnosis(fen, wrong, "x", "y", override_reason="Student explained a different root cause.",
                                      student_quote="I never look at what the opponent threatens")
    assert "student_quote_warning" in invented
    assert state.diagnostics[0]["cause_basis"] == "unknown"

    state.diagnostics.clear()
    tools.submit_diagnosis(fen, wrong, "x", "y", override_reason="Student explained a different root cause.",
                           student_quote="I just didn't notice my queen could be taken")
    diag = state.diagnostics[0]
    assert diag["category_basis"] == "student_statement" and diag["cause_basis"] == "student_statement"
    assert diag["student_cause"] == "I just didn't notice my queen could be taken"
    assert diag["student_evidence"]["quote"] == "I just didn't notice my queen could be taken"


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


def test_analyze_position_no_wdl(stub_config):
    # a prediction with no draw/loss → no risk fields, no crash.
    engines = FakeEngines(maia={START: {"e2e4": 1.0}})
    tools, _ = make_tools(stub_config, engines)
    summary = tools.analyze_position(START)["human_tendency_at_student_elo"]
    assert "risk_profile" not in summary
    assert "win_probability_for_side_to_move" in summary


def test_human_tendency_summary_says_whether_most_players_agree(stub_config):
    # Router run t0zauzig, S5: the data was Qh6 31.6%, Qa6 12.6%, Qg6 11.6% and the coach said
    # "most humans at this level try Qh6 or Qg6". The backend now writes the sentence to quote.
    split = FakeEngines(maia={START: {"e2e4": 0.32, "d2d4": 0.13, "g1f3": 0.12}})
    summary = make_tools(stub_config, split)[0].analyze_position(START)["human_tendency_at_student_elo"]
    assert summary["in_words"] == ("No single move is chosen by most players rated about 1150 playing White: "
                                   "e4 32%, d4 13%, Nf3 12%.")
    clear = FakeEngines(maia={START: {"e2e4": 0.73, "d2d4": 0.07}})
    summary = make_tools(stub_config, clear)[0].analyze_position(START)["human_tendency_at_student_elo"]
    assert summary["in_words"] == "Most players rated about 1150 playing White choose e4 (73%)."


def test_analyze_position_reports_the_digest_risk_for_an_agenda_position(stub_config):
    # Run sce7o0q6 S27: the digest called the move-31 position must-hold (Maia 10/53/37 at the
    # game-time rating) but analyze_position on the same FEN said drawish (17/63/19 at 1200), and
    # the coach told the student the opposite of the digest. One position, one set of numbers.
    from app.review import GameReview, MoveRecord
    from tests.test_review import _GAME_2DTD_RECORDS
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=0, human_win_prob=0.10, human_draw_prob=0.53,
                     human_loss_prob=0.37, maia_elo=1142, **_GAME_2DTD_RECORDS[31])
    engines = FakeEngines(maia={rec.fen_before: {"g1h1": 0.83, "g1h2": 0.17}},
                          maia_wdl={rec.fen_before: (0.17, 0.63, 0.19)})
    tools, state = make_tools(stub_config, engines)
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    human = tools.analyze_position(rec.fen_before)["human_tendency_at_student_elo"]
    assert human["win_draw_loss_for_side_to_move"] == {"win": 0.10, "draw": 0.53, "loss": 0.37}
    assert human["risk_profile"] == "must-hold" and "1142" in human["wdl_source"]


def test_human_move_sentence_names_the_position_and_side_after_applied_moves(stub_config):
    # Run sce7o0q6 S2: a Maia share for the position after 20. Kh1 was told as if it were for
    # the position before it. Run 9wthn48e: Black's 89% recapture after 20. Nxb5 was told as the
    # share of players choosing 20. Nxb5.
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    engines = FakeEngines(maia={after_e4.fen(): {"e7e5": 0.6, "c7c5": 0.3}})
    human = make_tools(stub_config, engines)[0].analyze_position(START, moves_to_apply=["e4"])
    assert human["human_tendency_at_student_elo"]["in_words"] == (
        "After e4, Black to move: Most players rated about 1150 playing Black choose e5 (60%).")


def test_analyze_position_says_whether_moving_a_piece_is_forced(stub_config):
    # Run kjofd3ug S6: "moving the queen is absolutely forced", asserted with nothing behind it.
    board = chess.Board("r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 3 20")
    board.push_san("Ne4")
    fen = board.fen()
    engines = FakeEngines(multipv={fen: [("d6d7", -460), ("d6h6", -480), ("d6g6", -500)]})
    engines.restricted[fen] = ("c7e6", -900)       # best reply that leaves the queen: 4.4 pawns worse
    forced = make_tools(stub_config, engines)[0].analyze_position(fen)["forced_check"]
    assert "queen on d6 is practically forced" in forced and "Ne6" in forced and "4.4 pawns" in forced
    engines.restricted[fen] = ("c7e6", -500)       # within half a pawn
    choice = make_tools(stub_config, engines)[0].analyze_position(fen)["forced_check"]
    assert "preference, not forced" in choice


def test_submit_diagnosis_accepts_allows_forcing_sequence(stub_config):
    from app.tools import ERROR_TAXONOMY
    tools, state = make_tools(stub_config)
    assert "allows_forcing_sequence" in ERROR_TAXONOMY
    result = tools.submit_diagnosis(START, "allows_forcing_sequence",
                                    "f4 opened the diagonal and allowed ...Qb6+ and ...Qxb2.",
                                    "Check what forcing replies a move gives the opponent.")
    assert result.get("success") is True
    assert tools.submit_diagnosis(START, "allows_forcing_sequences", "x y z w v u", "a b c d e f").get("error")


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


def test_sparring_reply_offers_post_mortem_when_game_ends(stub_config):
    board = chess.Board()
    for san in ("f3", "e5", "g4"):
        board.push_san(san)
    engines = FakeEngines(maia={board.fen(): {"d8h4": 1.0}})
    tools, state = make_tools(stub_config, engines)
    state.sparring_mode = True
    state.board = board

    result = tools.play_sparring_reply()

    assert result["coach_played"] == "Qh4#"
    assert result["game_over"] is True
    assert result["result"] == "0-1"
    assert "/review_spar" in result["post_mortem"]


def test_explain_move_returns_verified_facts(stub_config):
    tools, _ = make_tools(stub_config)
    facts = tools.explain_move(START, "Nf3")
    assert facts["legal"] is True
    assert facts["moving_piece"] == "knight"
    assert facts["is_capture"] is False and facts["gives_check"] is False
    assert facts["moved_piece_hangs"] is False


def test_explain_move_rejects_illegal(stub_config):
    tools, _ = make_tools(stub_config)
    facts = tools.explain_move(START, "Qd5")  # queen can't reach d5 from the start
    assert facts["legal"] is False and "error" in facts


def test_explain_move_flags_hanging_piece(stub_config):
    tools, _ = make_tools(stub_config)
    # White Qd1 plays Qd5, walking onto a square the e6 pawn attacks, undefended.
    fen = "4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1"
    facts = tools.explain_move(fen, "Qd5")
    assert facts["legal"] is True and facts["moved_piece_hangs"] is True


def _review_with_reasons(reasons):
    from app.review import GameReview, MoveRecord
    rec = MoveRecord(ply=0, number=1, color="white", san="Re1", uci="f1e1",
                     fen_before=START, fen_after=START, eval_before_cp=0, eval_after_cp=-300,
                     drop_pct=20.0, quality="mistake")
    rec.explanation_primitives = {"reasons": reasons}
    return GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])


_TACTIC_REASONS = [
    {"text": "Re1 does not address Qxh2#; the opponent gets a forcing move on h2",
     "key": True, "squares": ["h2"], "keywords": ["mate"]},
    {"text": "won a knight on b5", "key": False, "squares": ["b5"], "keywords": ["material", "won"]},
]


def test_evaluate_explanation_probes_when_key_reason_missing(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    result = tools.evaluate_explanation(START, "it pushes the queen away and gains tempo")
    assert result["gradable"] is True
    assert result["key_reason_named"] is False
    assert result["reveal_allowed"] is False
    assert "missing_key_reason" not in result
    assert "key_reason" not in result
    assert result["probe_question"] == "After the best move, what forcing threat on h2 is being stopped?"
    assert result["probe_facts"]["target_squares"] == ["h2"]
    assert result["probe_facts"]["focus"] == "forcing_threat"
    assert "probe_question" in result["guidance"]


def test_evaluate_explanation_passes_when_key_reason_named(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    result = tools.evaluate_explanation(START, "it blocks e4 and stops the mate on h2")
    assert result["key_reason_named"] is True
    assert "submit_diagnosis" in result["guidance"]
    assert "missing_key_reason" not in result


def test_evaluate_explanation_stops_probing_after_two_misses(stub_config):
    # Backend-enforced probe budget: after two misses, tell the coach to teach it.
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    r1 = tools.evaluate_explanation(START, "it gains tempo")
    r2 = tools.evaluate_explanation(START, "still just tempo")
    assert r1["probes_so_far"] == 1 and r1["reveal_allowed"] is False
    assert "key_reason" not in r1
    assert r2["probes_so_far"] == 2 and r2.get("reveal_allowed") is True
    assert "teach it directly" in r2["guidance"]


def test_evaluate_explanation_provenance_rejects_stale_student_text(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    state.latest_user_text = "that makes sense, let's go back to your agenda"

    result = tools.evaluate_explanation(START, "not sure")

    assert result["verdict"] == "stale_or_fabricated_student_input"
    assert state.socratic_attempts == {}
    assert state.grading_integrity_events
    assert state.grading_integrity_events[0]["tool"] == "evaluate_explanation"


def test_evaluate_student_answer_provenance_rejects_fabricated_answer(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    state.latest_user_text = "What is the best move here?"

    result = tools.evaluate_student_answer(START, "e4")

    assert result["verdict"] == "stale_or_fabricated_student_input"
    assert state.socratic_attempts == {}
    assert state.grading_integrity_events[0]["tool"] == "evaluate_student_answer"


def test_evaluate_threat_answer_provenance_allows_current_student_text(stub_config):
    tools, state = make_tools(stub_config, _threat_engines())
    state.latest_user_text = "Black is threatening Qh4, I think."

    result = tools.evaluate_threat_answer(THREAT_FEN, "Qh4")

    assert result["verdict"] == "correct_threat"
    assert state.grading_integrity_events == []


def test_evaluate_threat_answer_provenance_rejects_fabricated_answer(stub_config):
    tools, state = make_tools(stub_config, _threat_engines())
    state.latest_user_text = "What is Black threatening?"

    result = tools.evaluate_threat_answer(THREAT_FEN, "Qh4")

    assert result["verdict"] == "stale_or_fabricated_student_input"
    assert state.socratic_attempts == {}
    assert state.grading_integrity_events[0]["tool"] == "evaluate_threat_answer"


def test_threat_described_in_words_is_graded_not_rejected(stub_config):
    # Recorded 2026-10-12 ("black is threatening queen to g6"): the model's Qg6 was rejected
    # as stale input, and the routing note misread the phrase as the pawn move g6.
    tools, state = make_tools(stub_config, _threat_engines())
    state.latest_user_text = "Black is threatening the queen to h4"

    result = tools.evaluate_threat_answer(THREAT_FEN, "Qh4")

    assert result["verdict"] == "correct_threat"
    assert state.grading_integrity_events == []


def test_student_move_described_in_words_is_graded_not_rejected(stub_config):
    tools, state = make_tools(stub_config, _answer_engines())
    state.latest_user_text = "I'd move my knight to f3"

    result = tools.evaluate_student_answer(START, "Nf3")

    assert result["verdict"] != "stale_or_fabricated_student_input"
    assert state.grading_integrity_events == []


def test_generic_engine_reason_is_taught_not_probed(stub_config):
    # #31, live eval 2026-10-12 move 24: the key reason is anchored on Rfe1's destination
    # (the human-labelled corpus endorses that token), but it is only the generic "engine-
    # verified improvement" sentence. Probing it ("what tactical idea around e1 changes?",
    # then "what does your rook do from e1?") asked for what the student had already said.
    from app.review import GameReview, MoveRecord, _explanation_primitives
    from tests.test_review import _GAME_2DTD_RECORDS
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=0, **_GAME_2DTD_RECORDS[24])
    rec.explanation_primitives = _explanation_primitives(
        rec, chess.Board(rec.fen_before), chess.Board(rec.fen_after))
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    answer = "it moves my rook to safety, and offers a rook trade"
    state.latest_user_text = answer

    result = tools.evaluate_explanation(rec.fen_before, answer)

    assert "probe_question" not in result
    assert result["gradable"] is False and "teach the verified line" in result["guidance"].lower()
    # Run s2hy697w S11: both reasons are true on the board (the g2 pawn attacked the f1 rook;
    # Rfe1 faces the e8 rook), but the coach answered "Yes, exactly" while the key point, the
    # forcing line Kxg2 allows, was missing. The grader credits the true parts and says so.
    assert any("out of the attack by the pawn on g2" in c for c in result["credited"])
    assert any("trade" in c and "e8" in c for c in result["credited"])
    assert "do not say 'exactly'" in result["guidance"].lower()
    logged = tools.submit_diagnosis(rec.fen_before, "ignored_threat",
                                    "Played 24. Kxg2 instead of 24. Rfe1.", "Scan forcing replies first.")
    assert logged.get("success") is True


def test_naming_only_the_move_is_not_an_explanation(stub_config):
    # Live eval 2026-10-12 move 26: the student answered just "Rd4" and the coach passed
    # it to evaluate_explanation; the square d4 inside the move matched the reason
    # ("the rook on d4 controls g4"), so the WHY step closed without any reason given.
    from app.review import GameReview, MoveRecord, _explanation_primitives
    from tests.test_review import _GAME_2DTD_RECORDS
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=0, **_GAME_2DTD_RECORDS[26])
    rec.explanation_primitives = _explanation_primitives(
        rec, chess.Board(rec.fen_before), chess.Board(rec.fen_after))
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])

    state.latest_user_text = "Rd4"
    result = tools.evaluate_explanation(rec.fen_before, "Rd4")
    assert result.get("key_reason_named") is not True
    assert f"explain-complete:{rec.fen_before}" not in state.socratic_attempts
    assert "Ask why" in result["guidance"]

    state.latest_user_text = "Rd4 so the rook on d4 covers g4"
    assert tools.evaluate_explanation(rec.fen_before, state.latest_user_text)["key_reason_named"] is True


def test_a_tactic_override_needs_the_students_own_verified_motif(stub_config):
    # Run ec0yrv9l: 26. Qxa7 was logged missed_fork over the backend's ignored_threat, citing "a skewer, pin,
    # and queen fork" from the better line; the skewer and pin were Black's and the fork won nothing.
    from app.review import GameReview, MoveRecord
    from tests.test_review import _GAME_2DTD_RECORDS
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=-446, **{**_GAME_2DTD_RECORDS[26], "best_line_motifs": [
        {"motif": "skewer", "move": "Re4", "detail": "", "side": "black", "ply": 1},
        {"motif": "queen_fork", "move": "Qc7", "detail": "", "side": "white", "ply": 4, "material_won": False}]})
    rec.recommended_diagnosis = "ignored_threat"
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    reason = ("The backend suggested ignored_threat, but the line for Rd4 involves a skewer, pin, and queen fork, "
              "making missed_fork more accurate for the tactical nature of the defense.")

    refused = tools.submit_diagnosis(rec.fen_before, "missed_fork", "Played 26. Qxa7.", "Check threats.",
                                     override_reason=reason)
    assert "needs a verified fork" in refused["error"] and not state.diagnostics
    rec.best_line_motifs[1]["material_won"] = True  # a fork that wins material is real evidence
    assert tools.submit_diagnosis(rec.fen_before, "missed_fork", "Played 26. Qxa7.", "Check threats.",
                                  override_reason=reason)["success"] is True


def test_castling_described_in_words_matches_the_submitted_move(stub_config):
    tools, state = make_tools(stub_config)
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"):
        board.push_san(san)
    state.latest_user_text = "I would castle kingside"
    assert tools._verify_latest_student_input("evaluate_student_answer", "student_move_san", "O-O", [board]) is None
    assert tools._verify_latest_student_input("evaluate_student_answer", "student_move_san", "Ke2", [board])


def test_explanation_with_typos_fixed_is_accepted_but_stale_replay_is_not(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    state.latest_user_text = "because it blocks teh advance of their e pawn and atacks the queen"
    fixed = tools.evaluate_explanation(START, "because it blocks the advance of their e pawn and attacks the queen")
    assert fixed.get("verdict") != "stale_or_fabricated_student_input"
    state.latest_user_text = "black is threatening Qxg4. I need to move Rd4"
    assert tools.evaluate_explanation(START, "not sure")["verdict"] == "stale_or_fabricated_student_input"


def test_explanation_given_earlier_in_the_same_moment_is_genuine(stub_config):
    # Run fjtvd2b8: at S4 the student gave the move with its reason; after two detour questions
    # ("let's go back to move 24") the coach graded that verbatim reason and it was rejected as
    # stale, so the integrity guard replaced the whole reply.
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    reason = "I think I should moe Ne4 to block their pawn's advancement, and to gain tempo"
    state.moment_student_texts = [reason, "Why is Ne4 better than taking on b5?"]
    state.latest_user_text = "Thanks, that's clear. Let's go back to move 24."

    assert tools.evaluate_explanation(START, reason).get("verdict") != "stale_or_fabricated_student_input"
    state.moment_student_texts = []
    assert tools.evaluate_explanation(START, reason)["verdict"] == "stale_or_fabricated_student_input"


def test_a_trimmed_explanation_is_graded_with_the_students_whole_sentence(stub_config):
    # Quality-SOL §6.1: a grader-extracted span ("it gains tempo") cost the student a probe although the
    # same sentence named the key reason, and a span can drop a qualifier the coach must see.
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    state.latest_user_text = "Hmm. It gains tempo, and it stops the mate on h2. Right?"
    result = tools.evaluate_explanation(START, "it gains tempo")
    assert result["key_reason_named"] is True and f"explain:{START}" not in state.socratic_attempts
    assert (result["graded_text"], result["graded_text_source"]) == (
        "It gains tempo, and it stops the mate on h2.", "latest_message")

    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    state.moment_student_texts = ["Honestly I'm not sure it stops the mate on h2."]
    state.latest_user_text = "Back to the lesson."
    result = tools.evaluate_explanation(START, "it stops the mate on h2")
    assert (result["graded_text"], result["graded_text_source"]) == (
        "Honestly I'm not sure it stops the mate on h2.", "earlier_moment_message")


def test_agenda_moment_is_graded_against_its_verified_best_move(stub_config):
    # Runs cgxiksj3 and t0zauzig, move 31: both king moves lose at depth, so a fresh depth-16
    # search named the played Kh1 "best_move" while grading Kh2 "best". The agenda's verified
    # best move (Kh2) is the reference; the played mistake grades by the moment's own label.
    from app.review import GameReview, MoveRecord
    from tests.test_review import _MOVE31_RUN13
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=-445, **_MOVE31_RUN13)
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])

    state.latest_user_text = "Kh2"
    best = tools.evaluate_student_answer(rec.fen_before, "Kh2")
    assert best["verdict"] == "best" and best["best_move"] == "Kh2"
    assert best["best_line"].split()[:3] == ["Kh2", "Qh5+", "Kg3"]

    state.socratic_attempts.clear()
    state.latest_user_text = "Kh1"
    assert tools.evaluate_student_answer(rec.fen_before, "Kh1")["verdict"] == "wrong"


def test_described_moves_resolve_to_the_named_piece_not_a_pawn():
    from app.notation import legal_san_mentions
    session_b = chess.Board("3r2k1/p2r1pp1/2p1q2p/Q1P1p3/1P2P3/P3BP2/7P/R3R1K1 w - - 1 23")
    session_b.push(chess.Move.null())  # the opponent's view, as threat grading uses
    assert legal_san_mentions(session_b, "black is threatening queen to g6") == ["Qg6+"]
    move4 = chess.Board("rn1qkbnr/ppp2ppp/4p3/3p1b2/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq - 0 4")
    assert legal_san_mentions(move4, "Once Black defends c7, your knight on b5 isn't doing much") == []


def test_evaluate_explanation_accepts_live_pawn_defense_paraphrase(stub_config):
    # #25: the reason must come from the real review pipeline for move 26, not a
    # hand-written reason (the earlier fixture passed while live grading failed).
    from app.review import GameReview, MoveRecord, _explanation_primitives
    from tests.test_review import _GAME_2DTD_RECORDS
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=0, **_GAME_2DTD_RECORDS[26])
    rec.explanation_primitives = _explanation_primitives(
        rec, chess.Board(rec.fen_before), chess.Board(rec.fen_after))
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    state.latest_user_text = "because that one actually defends my g pawn"

    result = tools.evaluate_explanation(rec.fen_before, "because that one actually defends my g pawn")

    assert result["key_reason_named"] is True
    assert result["reveal_allowed"] is True


def test_evaluate_explanation_accepts_ne4_tempo_for_simple_trade(stub_config):
    tools, state = make_tools(stub_config)
    from app.review import GameReview, MoveRecord
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    rec = MoveRecord(ply=38, number=20, color="white", san="Nxb5", uci="c3b5",
                     fen_before=fen, fen_after=fen, eval_before_cp=0, eval_after_cp=-100,
                     drop_pct=18.0, quality="inaccuracy", best_move_san="Ne4")
    rec.explanation_primitives = {"reasons": [
        {"text": "Ne4 attacks the queen on d6 with tempo and keeps the knight active",
         "key": True, "squares": ["e4", "d6"], "keywords": ["active", "tempo"]},
    ]}
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])

    result = tools.evaluate_explanation(
        fen, "because it blocks the advance of their e pawn, attacks the queen, and gains tempo")

    assert result["key_reason_named"] is True
    assert result["reveal_allowed"] is True
    assert "probe_question" not in result


def test_evaluate_explanation_give_up_exhausts_probe_budget_and_allows_diagnosis(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)

    result = tools.evaluate_explanation(START, "I don't know, just tell me")

    assert result["student_passed"] is True
    assert result["reveal_allowed"] is True
    assert state.socratic_attempts[f"explain:{START}"] == 2
    assert state.socratic_attempts[f"explain-complete:{START}"] is True
    submitted = tools.submit_diagnosis(
        START,
        "ignored_threat",
        "Played Re1 without stopping the mate threat on h2.",
        "Ask what forcing threat the opponent has before moving.",
    )
    assert submitted["success"] is True


def test_evaluate_explanation_spanish_give_up_exhausts_probe_budget(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)

    result = tools.evaluate_explanation(START, "No sé, dímelo")

    assert result["student_passed"] is True
    assert result["reveal_allowed"] is True
    assert state.socratic_attempts[f"explain-complete:{START}"] is True


def test_submit_diagnosis_names_remaining_explanation_attempt(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    tools.evaluate_explanation(START, "it gains tempo")

    result = tools.submit_diagnosis(START, "ignored_threat", "x", "y")

    assert "error" in result
    assert result["remaining_explanation_attempts"] == 1
    assert "1 more explanation attempt required, or the student may pass" in result["error"]


def test_evaluate_explanation_no_key_reason_is_not_gradable(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons([_TACTIC_REASONS[1]])  # only the secondary (key=False)
    result = tools.evaluate_explanation(START, "anything")
    assert result["gradable"] is False


def test_unverified_reason_is_not_confirmed_when_no_key_reason_exists(stub_config):
    # Router run 5kf7rqwj S18: "Kh2, because it avoids the perpetual" (for White, who was lost,
    # the perpetual was the draw) met the old "if the explanation is sound, affirm it" guidance,
    # and the coach affirmed it. Only verified facts may be confirmed.
    from app.review import GameReview, MoveRecord, _explanation_primitives
    from tests.test_review import _GAME_2DTD_RECORDS
    tools, state = make_tools(stub_config)
    rec = MoveRecord(eval_before_cp=0, eval_after_cp=0, deep_eval_before_cp=-834, **_GAME_2DTD_RECORDS[31])
    rec.explanation_primitives = _explanation_primitives(
        rec, chess.Board(rec.fen_before), chess.Board(rec.fen_after))
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
    state.latest_user_text = "Kh2, because it avoids the perpetual."

    result = tools.evaluate_explanation(rec.fen_before, state.latest_user_text)

    assert result["gradable"] is False and result["credited"] == []
    assert "do not confirm" in result["guidance"].lower() and "if the explanation is sound" not in result["guidance"]
    # Run sce7o0q6 S18: "do not confirm" alone left the misconception standing (the coach said only
    # "Kh2 is the best move" and wrapped up). The grader names the correction, and the backend
    # states it when the reply leaves it out.
    assert "drawing chance" in result["correction"] and "White" in result["correction"]
    from app.agent_loop import _state_unreported_feedback
    state._turn_grading_results = [{"tool": "evaluate_explanation", "args": {}, "result": result}]
    assert "drawing chance" in _state_unreported_feedback("Kh2 is the best move. That wraps up.", state)
    spoken = "Careful: for you a perpetual check would have been a draw, which is welcome here."
    assert _state_unreported_feedback(spoken, state) == spoken


def test_evaluate_explanation_unknown_fen_errors(stub_config):
    tools, state = make_tools(stub_config)
    state.review = _review_with_reasons(_TACTIC_REASONS)
    assert "error" in tools.evaluate_explanation("8/8/8/8/8/8/8/K6k w - - 0 1", "x")


def test_sparring_temperature_zero_is_argmax(stub_config):
    # P2-11c: spar_temperature=0 → always Maia's most likely move (deterministic),
    # even when a lower-prob move would sometimes win proportional sampling.
    stub_config.spar_temperature = 0.0
    engines = FakeEngines(maia={START: {"e2e4": 0.49, "d2d4": 0.51}})
    tools, state = make_tools(stub_config, engines)
    state.sparring_mode = True
    for _ in range(8):
        state.board = chess.Board()
        assert tools.play_sparring_reply()["coach_played"] == "d4"


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

def _serve_review_pgn(monkeypatch, pgn=REVIEW_PGN):
    monkeypatch.setattr("app.tools.external.lichess_game_pgn", lambda game_id: pgn)


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
    _serve_review_pgn(monkeypatch)
    tools, state = make_tools(stub_config, _scripted_engines())
    result = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD", mode="restart")
    assert "GAME REVIEW" in result["review_digest"]
    assert state.review is not None
    assert state.move_index == 0
    meta = load_lesson_metadata(stub_config.lesson_file)
    assert meta["last_game_id"] == "2dtDqQgR"
    assert meta["lesson_status"] == "fresh_review"
    assert state.teaching_ledger["self_assessment"]["status"] == "pending"
    assert "Backend self-assessment opener - P0-8" in result["review_digest"]


def test_continue_known_lichess_game_jumps_to_saved_agenda(stub_config, monkeypatch):
    save_lesson_metadata(stub_config.lesson_file, {
        "last_game_id": "2dtDqQgR",
        "last_agenda_index": 0,
        "lesson_status": "diagnosis_logged",
    })
    _serve_review_pgn(monkeypatch)
    tools, state = make_tools(stub_config, _scripted_engines())
    result = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD", mode="continue")
    assert result["resume"]["moment_number"] == 1
    assert result["resume"]["move_label"] == "3. Qxf7+"
    assert state.move_index == state.review.agenda_plies[0]
    meta = load_lesson_metadata(stub_config.lesson_file)
    assert meta["lesson_status"] == "awaiting_answer"


def test_lichess_download_failure_is_reported_without_parsing(stub_config, monkeypatch):
    tools, state = make_tools(stub_config, _scripted_engines())
    _serve_review_pgn(monkeypatch, None)
    failed = tools.fetch_lichess_game_by_id("2dtDqQgRy2OD")
    assert "did not return the game" in failed["error"] and state.review is None
    _serve_review_pgn(monkeypatch, "")
    missing = tools.fetch_lichess_game_by_id("zzzzzzzz")
    assert "not found" in missing["error"] and state.review is None


def test_downloaded_unmatched_game_requires_pov_before_digest(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines

    tools, state = make_tools(stub_config, _scripted_engines())
    pgn = RPGN.replace('[White "mcmd"]', '[White "Alice"]')

    result = tools._review_and_load(pgn, source_game_id="borrowed1")

    assert result["pov_choice_required"] is True
    assert "review_digest" not in result
    assert result["white"] == "Alice"
    assert result["default_dossier_consent"] is False
    assert state.pending_unmatched_review["source_game_id"] == "borrowed1"
    assert state.review_context["dossier_consent"] is False


def test_downloaded_own_game_includes_current_rating_offer(stub_config, monkeypatch):
    from app import external
    from tests.test_review import PGN as RPGN, _scripted_engines

    monkeypatch.setattr(external, "lichess_user_profile",
                        lambda username: {"perfs": {"rapid": {"rating": 1420}}})
    pgn = RPGN.replace('[Event "Test"]',
                       '[Event "Test"]\n[Site "https://lichess.org/abc12345"]\n[TimeControl "600+0"]')
    tools, state = make_tools(stub_config, _scripted_engines())

    result = tools._review_and_load(pgn, source_game_id="abc12345")

    assert "Backend rating enrichment - P0-4d" in result["review_digest"]
    assert "current rapid rating ~1420" in result["review_digest"]
    assert state.review_context["rating_enrichment"]["current_rating"] == 1420
    assert state.review.conversation_elo == 1150


def test_tool_loaded_fresh_own_game_includes_self_assessment_opener(stub_config):
    tools, state = make_tools(stub_config, _scripted_engines())

    result = tools._review_and_load(REVIEW_PGN, source_game_id="fresh1")

    assert state.teaching_ledger["self_assessment"]["status"] == "pending"
    assert "Backend self-assessment opener - P0-8" in result["review_digest"]
    assert "Before I show you my agenda" in result["self_assessment_opener"]


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
    tools.evaluate_explanation(record.fen_before, "not the key reason")
    tools.evaluate_explanation(record.fen_before, "still not the key reason")
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


def test_student_answer_routes_opponent_only_move_to_threat_grading(stub_config):
    from app.review import GameReview, MoveRecord
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    tools, state = make_tools(stub_config, FakeEngines(threats={fen: [
        {"threat": "opponent gains space and attacks the knight", "threat_line": "e4"},
    ]}))
    record = MoveRecord(
        ply=38, number=20, color="white", san="Nxb5", uci="c3b5",
        fen_before=fen, fen_after=fen, eval_before_cp=0, eval_after_cp=-100,
        drop_pct=18.0, quality="inaccuracy",
        initial_question="White to move. What candidate moves would you consider, and what is Black threatening?",
    )
    state.review = GameReview(headers={}, records=[record], student_color="white", agenda_plies=[0])
    state.lesson_metadata = {"last_agenda_index": 0, "lesson_status": "awaiting_answer"}

    result = tools.evaluate_student_answer(fen, "e4")

    assert result["verdict"] == "correct_threat"
    assert result["routed_from"] == "evaluate_student_answer"
    assert result["interpreted_as"] == "opponent_threat"
    assert result["threat_line"] == "e4"
    assert state.socratic_attempts.get(f"threat:{fen}") == 1
    assert state.socratic_attempts.get(fen) is None


def test_explicit_own_illegal_move_does_not_route_to_threat_grading(stub_config):
    from app.review import GameReview, MoveRecord
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    tools, state = make_tools(stub_config, FakeEngines(threats={fen: [
        {"threat": "opponent gains space and attacks the knight", "threat_line": "e4"},
    ]}))
    record = MoveRecord(
        ply=38, number=20, color="white", san="Nxb5", uci="c3b5",
        fen_before=fen, fen_after=fen, eval_before_cp=0, eval_after_cp=-100,
        drop_pct=18.0, quality="inaccuracy",
        initial_question="White to move. What candidate moves would you consider, and what is Black threatening?",
    )
    state.review = GameReview(headers={}, records=[record], student_color="white", agenda_plies=[0])
    state.lesson_metadata = {"last_agenda_index": 0, "lesson_status": "awaiting_answer"}

    result = tools.evaluate_student_answer(fen, "I would play e4")

    assert result["verdict"] == "illegal"
    assert "routed_from" not in result
    assert state.socratic_attempts.get(fen) == 1
    assert state.socratic_attempts.get(f"threat:{fen}") is None


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


def test_a_repetition_draw_is_seen_from_game_positions_and_cannot_be_logged_as_a_mistake(stub_config):
    # 2dtDqQgR: 31. Kh1 repeated the position a third time. Engine lines from a bare FEN cannot know.
    from app.review import review_game
    from tests.test_review import _Cfg
    pgn = ('[White "mcmd"]\n[Black "Opponent"]\n[Result "1/2-1/2"]\n\n'
           "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 1/2-1/2\n")
    engines = FakeEngines()
    tools, state = make_tools(stub_config, engines)
    state.review = review_game(pgn, FakeEngines(), _Cfg(), with_maia=False)
    final_fen = state.review.records[-1].fen_after
    engines.maia_data[final_fen] = {"g8f6": 1.0}            # Maia's view of play going on: "sharp"
    engines.maia_wdl[final_fen] = (0.45, 0.15, 0.40)
    before_last = state.review.records[-1].fen_before

    in_game = tools.analyze_position(before_last, moves_to_apply=["Nf3"])
    assert "threefold repetition" in in_game["draw_rule"] and "claimable" in in_game["draw_rule"]
    # Run h2z7j1nd S27: the final position itself (after the repeating move) was analysed without history.
    final = tools.analyze_position(final_fen)
    assert "threefold repetition" in final["draw_rule"]
    assert final["human_tendency_at_student_elo"]["risk_profile"] == "drawn by rule"
    logged = tools.submit_diagnosis(fen=before_last, error_category="ignored_threat",
                                    error_explanation="Played 5. Nf3.", preventative_habit="Check threats.")
    assert "not a mistake" in logged["error"] and not state.diagnostics
    state.review = None  # the same FEN, no longer a position of a reviewed game
    assert "draw_rule" not in tools.analyze_position(before_last, moves_to_apply=["Nf3"])


def test_game_positions_come_from_the_game_and_an_invented_one_is_refused(stub_config):
    # Run pr40jw2f: with 31. Kh1 off the agenda the coach had no FEN for it, made one up (two black
    # queens) and reported a 97% loss chance for a position that never occurred.
    from app.review import review_game
    from tests.test_review import _Cfg
    pgn = ('[White "mcmd"]\n[Black "Opponent"]\n[Result "1/2-1/2"]\n\n'
           "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 1/2-1/2\n")
    tools, state = make_tools(stub_config)
    state.review = review_game(pgn, FakeEngines(), _Cfg(), with_maia=False)

    final = tools.get_game_position(5, "white")
    assert final["fen_before"] == state.review.records[-1].fen_before
    assert "not a mistake" in final["draw_rule"] and "repetition" in final["game_ending"]
    assert "correction" not in final  # level if play had gone on: nothing to correct
    state.review.records[-1].eval_without_draw_rule_cp = -500  # 2dtDqQgR: White lost had play gone on
    assert "draw to welcome" in tools.get_game_position(5, "white")["correction"]
    assert "error" in tools.get_game_position(9, "white")

    invented = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 8 5"  # move 5, White to move
    refused = tools.analyze_position(invented)
    assert refused["game_fen"] == final["fen_before"] and "Never construct a FEN" in refused["error"]
    assert "error" not in tools.analyze_position(final["fen_before"])
    assert "error" not in tools.analyze_position("4k3/8/8/8/8/8/8/4K2R w K - 0 40")  # not a game move


def test_a_search_cut_short_by_its_time_limit_says_its_numbers_are_rough(stub_config):
    # Run e_adq2d7 S27: a 4-second search stopped at depth 14 gave 31. Kh2 -4.59 and the coach quoted it
    # as exact; at depth 30 it is -8.76.
    engines = FakeEngines()
    tools, _ = make_tools(stub_config, engines)
    full = tools.analyze_position(START)
    assert "search_note" not in full and full["top_stockfish_lines"][0]["depth_reached"] == 18
    engines.depth_reached[START] = 14
    cut = tools.analyze_position(START)
    assert "depth 14 of 18" in cut["search_note"] and "do not quote these numbers" in cut["search_note"]


def test_move_24_explanations_get_full_partial_or_a_concrete_probe(stub_config):
    # Slice B: "Qf4 fails to mate on e8" is the point of 24. Rfe1; "rook to safety" is true but partial.
    from app.review import GameReview, MoveRecord, _explanation_primitives
    from tests.test_review import _GAME_2DTD_RECORDS

    def grade(answer):
        tools, state = make_tools(stub_config)
        rec = MoveRecord(eval_before_cp=184, eval_after_cp=0, **_GAME_2DTD_RECORDS[24])
        rec.resource_refutation = {"resource": "Qf4", "line": ["Rxe8+", "Rxe8", "Qxe8#"],
                                   "alternatives": [["Qxe8+", "Rxe8", "Rxe8#"]], "mate_in": 2, "student_cp": 99998}
        rec.explanation_primitives = _explanation_primitives(rec, chess.Board(rec.fen_before), chess.Board(rec.fen_after))
        state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[0])
        state.latest_user_text = answer
        return tools.evaluate_explanation(rec.fen_before, answer)

    assert grade("If Black plays Qf4, I take on e8 with check and it's mate.")["key_reason_named"] is True
    assert grade("My rook and queen both hit e8, so Black can't play Qf4.")["key_reason_named"] is True
    partial = grade("it moves my rook to safety, and offers a rook trade")
    assert partial["key_reason_named"] is False and any("out of the attack" in c for c in partial["also_named"])
    assert partial["probe_question"] == "After 24. Rfe1, what happens if Black plays the same 24... Qf4? Look at e8."
