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
