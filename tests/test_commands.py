"""Regression tests for the command layer — including the /reset timeline wipe
(F3) and the chat-text board clobber (F4) found in the 2026-06-10 review."""
import chess

from app.commands import CommandProcessor, looks_like_pgn
from app.state import SessionState, TimelineManager
from tests.conftest import FakeEngines

PGN = """[Event "Test"]
[White "mcmd"]
[Black "Opponent"]
[Result "0-1"]

1. e4 e5 2. Qh5 Nc6 3. Qxf7+ Kxf7 0-1
"""


def make_processor(stub_config, engines=None):
    state = SessionState()
    timeline = TimelineManager(state)
    proc = CommandProcessor(state, timeline, engines or FakeEngines(), stub_config)
    return proc, state, timeline


# --- PGN DETECTION GUARD (F4 regression) ---

def test_chat_mentioning_move_one_is_not_a_pgn():
    assert not looks_like_pgn("why is 1. e4 considered the best move for beginners?")


def test_prose_with_move_names_and_squares_is_not_a_pgn(capsys):
    text = ("There are only two possible moves here, Kxg2 or Rfe1. "
            "Leaving the pawn on g2 gives me shelter.")
    assert not looks_like_pgn(text)
    captured = capsys.readouterr()
    assert "illegal san" not in captured.err.lower()


def test_real_pgn_detected():
    assert looks_like_pgn(PGN)
    assert looks_like_pgn("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6")


def test_chat_about_move_one_does_not_clobber_board(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("why is 1. e4 considered the best move for beginners?")
    assert state.board.fen() == chess.STARTING_FEN
    assert state.review is None
    assert result.prompt is not None  # treated as chat
    assert "1. e4" in result.prompt


def test_commands_match_exact_first_token_not_prefix(stub_config):
    # Audit: startswith made "/review-old", "/movefoo", "/sparring" trigger commands.
    proc, _, _ = make_processor(stub_config)
    # A lookalike is NOT a command — it's chat (gets a model prompt, not handled locally).
    res = proc.process("/review-old please")
    assert res.prompt is not None and "/review-old" in res.prompt
    assert proc.process("/sparring is fun").prompt is not None
    # The real command (bare) is still handled locally (its own message, no chat prompt).
    assert proc.process("/help").message and proc.process("/help").prompt is None


# --- LOADING & REVIEW (F1: digest goes to the model) ---

def test_pasted_pgn_loads_review_and_sends_digest(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process(PGN)
    assert state.review is not None
    assert state.board.fen() == chess.STARTING_FEN  # board at the START (F7)
    assert state.move_index == 0
    assert "GAME REVIEW" in result.prompt
    assert "Annotated moves" in result.prompt


# --- /reset VCR REWIND (F3 regression) ---

def test_reset_keeps_loaded_game_and_next_still_works(stub_config):
    proc, state, _ = make_processor(stub_config)
    proc.process(PGN)
    # Step into the game, rewind, step again — must not raise.
    assert proc.process("/next").message.startswith("[green]")
    assert proc.process("/next").message.startswith("[green]")
    proc.process("/reset")
    assert state.move_index == 0
    assert len(state.loaded_moves) == 6  # game NOT forgotten
    result = proc.process("/next")
    assert "Stepped forward" in result.message


def test_clear_wipes_everything(stub_config):
    proc, state, _ = make_processor(stub_config)
    proc.process(PGN)
    result = proc.process("/clear")
    assert result.reset_chat
    assert state.review is None
    assert state.loaded_moves == []


# --- NAVIGATION ---

def test_next_attaches_review_quality_note(stub_config):
    proc, state, _ = make_processor(stub_config)
    proc.process(PGN)
    result = proc.process("/next")
    assert "[Backend review of this move:" in result.prompt


def test_next_past_end_is_handled(stub_config):
    proc, _, _ = make_processor(stub_config)
    result = proc.process("/next")
    assert "end of the loaded game" in result.message


def test_goto_jumps_to_position_before_move(stub_config):
    proc, state, _ = make_processor(stub_config)
    proc.process(PGN)
    proc.process("/goto 2")  # before White's 2nd move => after 1. e4 e5
    board = chess.Board()
    board.push_san("e4"); board.push_san("e5")
    assert state.board.fen() == board.fen()


def test_move_illegal_rejected(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("/move Ke5")
    assert "Invalid or illegal" in result.message
    assert state.board.fen() == chess.STARTING_FEN


def test_move_in_sparring_requests_backend_reply(stub_config):
    proc, state, _ = make_processor(stub_config)
    proc.process("/spar")
    result = proc.process("/move e4")
    assert state.sparring_mode
    assert "play_sparring_reply" in result.prompt


def test_help_is_local_only(stub_config):
    proc, _, _ = make_processor(stub_config)
    result = proc.process("/help")
    assert result.prompt is None
    assert "/load_pgn" in result.message


def test_p0_8_declines_full_game_annotation_in_coach_mode(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("Can you annotate every move in this game?")

    assert result.prompt is None
    assert "can't annotate every move" in result.message
    assert "critical moments one at a time" in result.message
    assert state.teaching_ledger["last_plan"]["intent"] == "decline_full_game_annotation"
    assert state.teaching_ledger["last_plan"]["decline"] is True


def test_p0_8_direct_language_density_policy_is_injected(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("Explain directly and briefly in Spanish why Qxf7 is bad.")

    assert "[Backend Teaching Plan - P0-8]" in result.prompt
    assert "intent: answer_directly" in result.prompt
    assert "language: Spanish" in result.prompt
    assert "density: brief" in result.prompt
    assert "socratic_redirect: false" in result.prompt
    assert "withhold_answer: false" in result.prompt
    assert "no_tail_redirect: true" in result.prompt
    assert state.teaching_ledger["last_plan"]["language"] == "Spanish"


def test_p0_8_phase_summary_suppresses_tail_redirect(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("Summarize the game phase by phase.")

    assert "intent: explain_phase_summary" in result.prompt
    assert "no_tail_redirect: true" in result.prompt
    assert "do not append an unrequested quiz" in result.prompt
    assert state.teaching_ledger["last_plan"]["no_tail_redirect"] is True


def test_p0_8_quiz_me_keeps_socratic_withholding(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("Quiz me on this position.")

    assert "intent: ask_socratic_question" in result.prompt
    assert "withhold_answer: true" in result.prompt
    assert "socratic_redirect: true" in result.prompt
    assert state.teaching_ledger["last_plan"]["withhold_answer"] is True


def test_p0_8_fuzzy_intents_are_routed_without_prompt_guesswork(stub_config):
    proc, state, _ = make_processor(stub_config)

    translated = proc.process("Can you translate that to Spanish?")
    assert "intent: translate_previous" in translated.prompt
    assert "language: Spanish" in translated.prompt
    assert "no_tail_redirect: true" in translated.prompt

    exercise = proc.process("Give me a practice exercise from this.")
    assert "intent: generate_exercise" in exercise.prompt
    assert "withhold_answer: true" in exercise.prompt
    assert "socratic_redirect: true" in exercise.prompt

    blunder = proc.process("Why was that move a blunder?")
    assert "intent: explain_blunder" in blunder.prompt

    strength = proc.process("What did I do well in this game?")
    assert "intent: reinforce_strength" in strength.prompt
    assert state.teaching_ledger["last_plan"]["intent"] == "reinforce_strength"


def test_p0_8_fresh_review_opens_with_self_assessment(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())

    result = proc.process(RPGN)

    assert "Backend self-assessment opener - P0-8" in result.prompt
    assert "Before I show you my agenda" in result.prompt
    assert state.teaching_ledger["self_assessment"]["status"] == "pending"


def test_p0_8_self_assessment_answer_is_deterministically_graded(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    top = state.review.agenda[0]

    result = proc.process(f"I think {top.move_label} was the big mistake because I grabbed material.")

    assert "Backend self-assessment result - P0-8" in result.prompt
    assert "exact_top_agenda_match" in result.prompt
    assert state.teaching_ledger["self_assessment"]["status"] == "graded"
    assert state.teaching_ledger["self_assessment"]["matched_ply"] == top.ply


def test_p0_8_redundancy_ledger_is_injected_for_followup(stub_config):
    proc, state, _ = make_processor(stub_config)
    state.teaching_ledger["explained_moments"] = [
        {"move_label": "12... Rc3", "diagnosis": "king_safety", "fen": "fen1"}
    ]
    state.teaching_ledger["explained_concepts"] = [
        {"concept": "king_safety", "move_label": "12... Rc3", "fen": "fen1"}
    ]

    result = proc.process("Summarize the game phase by phase.")

    assert "Backend redundancy ledger - P0-8" in result.prompt
    assert "12... Rc3" in result.prompt
    assert "king_safety" in result.prompt
    assert "Do not re-quiz or re-teach" in result.prompt


def test_p0_8_wrap_up_records_verified_theme_takeaway(stub_config):
    from app.memory import load_dossier
    from app.review import GameReview, MoveRecord, ReviewTheme

    proc, state, _ = make_processor(stub_config)
    rec = MoveRecord(ply=0, number=1, color="white", san="e4", uci="e2e4",
                     fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                     eval_before_cp=0, eval_after_cp=-100, drop_pct=12.0,
                     quality="inaccuracy", recommended_diagnosis="ignored_threat")
    state.current_game_id = "game-1"
    state.review = GameReview(headers={}, records=[rec], student_color="white",
                              agenda_plies=[0], themes=[
                                  ReviewTheme(
                                      id="ignored_threat:opening",
                                      label="ignored threat (opening)",
                                      plies=[0],
                                      confidence="low",
                                      safe_summary="a single ignored threat moment in the opening: 1. e4.",
                                  )
                              ])

    result = proc.process("wrap up")

    assert "intent: close_with_takeaway" in result.prompt
    assert "Backend session close" in result.prompt
    assert "a single ignored threat moment" in result.prompt
    assert "do NOT ask a new Socratic question" in result.prompt
    assert state.teaching_ledger["last_takeaway"]["theme_id"] == "ignored_threat:opening"
    dossier = load_dossier(stub_config.dossier_file)
    assert dossier["session_takeaways"][0]["game_id"] == "game-1"
    assert "a single ignored threat moment" in dossier["session_takeaways"][0]["text"]


def test_p0_8_lesson_complete_ack_closes_without_new_quiz(stub_config):
    from app.review import GameReview, MoveRecord, ReviewTheme

    proc, state, _ = make_processor(stub_config)
    rec = MoveRecord(ply=0, number=1, color="white", san="e4", uci="e2e4",
                     fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                     eval_before_cp=0, eval_after_cp=-100, drop_pct=12.0,
                     quality="inaccuracy")
    state.current_game_id = "game-2"
    state.lesson_metadata = {"lesson_status": "lesson_complete", "last_agenda_index": 0}
    state.review = GameReview(headers={}, records=[rec], student_color="white",
                              agenda_plies=[0], themes=[
                                  ReviewTheme(
                                      id="bad_trade:opening",
                                      label="bad trade (opening)",
                                      plies=[0],
                                      confidence="low",
                                      safe_summary="a single bad trade moment in the opening: 1. e4.",
                                  )
                              ])

    result = proc.process("ok")

    assert "intent: close_with_takeaway" in result.prompt
    assert "routing_reasons: intent:close_after_lesson_complete" in result.prompt
    assert "no_tail_redirect: true" in result.prompt
    assert state.teaching_ledger["last_plan"]["intent"] == "close_with_takeaway"


def test_memorize_annotates_diagnoses_with_typicality(stub_config):
    from app.memory import load_dossier
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)  # loads review; ply 4 (Qxf7+) is typical_for_level
    blunder_fen = state.review.records[4].fen_before
    state.diagnostics.append({"fen": blunder_fen, "error_category": "king_safety",
                              "preventative_habit": "Check checks first."})
    proc.process("/memorize")
    d = load_dossier(stub_config.dossier_file)
    assert d["typicality_history"][0]["typical_errors"] == 1
    assert d["recent_positions"][0]["error_typicality"] == "typical_for_level"


# --- CANDIDATE-LIST COMMITMENT (cherry-pick loophole observed live) ---

def test_multi_candidate_answer_gets_backend_note(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    # Legal candidates at the agenda moment (before 3. Qxf7+): hedged answer.
    result = proc.process("I would consider Qxf7, or Nf3, or maybe Bc4")
    assert "multiple legal candidate moves" in result.prompt
    assert "Qxf7+" in result.prompt and "Nf3" in result.prompt and "Bc4" in result.prompt
    assert "NEVER pick the best candidate for them" in result.prompt


def test_single_move_answer_gets_no_note(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    result = proc.process("I think the answer is Nf3")
    assert "multiple legal candidate moves" not in result.prompt


def test_no_note_outside_lesson_or_sparring_context(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("should I study e4 or d4 openings?")
    assert "multiple legal candidate moves" not in result.prompt


# --- LOCAL-GAME LESSON RESUME ---

def test_local_game_resumes_at_saved_moment(stub_config):
    from app.memory import save_lesson_metadata, load_lesson_metadata
    from app.commands import CommandProcessor
    from app.state import SessionState, TimelineManager
    from tests.test_review import PGN as RPGN, _scripted_engines

    # Session 1: load the game, note its local id, simulate mid-lesson state.
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    game_id = state.current_game_id
    assert game_id and game_id.startswith("local-")
    assert state.lesson_metadata["lesson_status"] == "fresh_review"
    save_lesson_metadata(stub_config.lesson_file,
                         {"last_game_id": game_id, "last_agenda_index": 0,
                          "lesson_status": "awaiting_diagnosis"})

    # Session 2: same game -> resume at the saved agenda moment.
    state2 = SessionState()
    proc2 = CommandProcessor(state2, TimelineManager(state2), _scripted_engines(), stub_config)
    result = proc2.process(RPGN)
    assert "RESUMES at Moment 1" in result.prompt
    assert "resumed at Moment 1" in result.message
    # Board is at the agenda moment's position, not the start.
    assert state2.board.fen() == state2.review.agenda[0].fen_before


def test_completed_lesson_loads_fresh(stub_config):
    from app.memory import save_lesson_metadata
    from app.commands import CommandProcessor
    from app.state import SessionState, TimelineManager
    from tests.test_review import PGN as RPGN, _scripted_engines

    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    save_lesson_metadata(stub_config.lesson_file,
                         {"last_game_id": state.current_game_id, "last_agenda_index": 0,
                          "lesson_status": "lesson_complete"})
    state2 = SessionState()
    proc2 = CommandProcessor(state2, TimelineManager(state2), _scripted_engines(), stub_config)
    result = proc2.process(RPGN)
    assert "STARTING position" in result.prompt
    import chess as _chess
    assert state2.board.fen() == _chess.STARTING_FEN


# --- PENDING-DIAGNOSIS NUDGE (the "ok, let's go" stall observed live) ---

def test_pending_diagnosis_nudge_injected_until_logged(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    record = state.review.agenda[0]
    # The reveal happened; the model owes a diagnosis for this moment.
    state.lesson_metadata = {"last_game_id": state.current_game_id,
                             "last_agenda_index": 0,
                             "lesson_status": "awaiting_diagnosis"}
    result = proc.process("because it defends the pawn and activates the rook")
    assert "has NOT been logged yet" in result.prompt
    assert record.move_label in result.prompt

    # Once logged, the nudge disappears.
    state.diagnostics.append({"fen": record.fen_before, "error_category": "bad_trade"})
    result = proc.process("ok, let's go")
    assert "has NOT been logged yet" not in result.prompt


def test_no_diagnosis_nudge_outside_awaiting_state(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    state.lesson_metadata = {"last_game_id": state.current_game_id,
                             "last_agenda_index": 0,
                             "lesson_status": "awaiting_answer"}
    result = proc.process("what should I consider here?")
    assert "has NOT been logged yet" not in result.prompt


# --- GIVE-UP HANDLING (model was inventing guesses to farm the reveal) ---

def test_give_up_spends_attempt_and_hints_first(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    record = state.review.agenda[0]
    state.lesson_metadata = {"last_game_id": state.current_game_id,
                             "last_agenda_index": 0, "lesson_status": "awaiting_answer"}
    result = proc.process("I don't know")
    assert "GAVE UP" in result.prompt
    assert "1 of 2 before reveal" in result.prompt
    assert "NEVER invent a guess" in result.prompt
    assert state.socratic_attempts[record.fen_before] == 1


def test_second_give_up_allows_reveal(stub_config):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    state.lesson_metadata = {"last_game_id": state.current_game_id,
                             "last_agenda_index": 0, "lesson_status": "awaiting_answer"}
    proc.process("no idea")
    result = proc.process("i give up")
    assert "Reveal is allowed" in result.prompt


def test_give_up_outside_lesson_is_plain_chat(stub_config):
    proc, state, _ = make_processor(stub_config)
    result = proc.process("I don't know much about the London system")
    assert "GAVE UP" not in result.prompt
