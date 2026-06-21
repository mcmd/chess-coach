"""Tests for the agent loop refactor: the spinner wraps MODEL calls only, so
tools run outside any live display and can show their own progress bars."""
from types import SimpleNamespace

import chess

from app.agent_loop import (run_agent_turn, find_illegal_moves, apply_claim_gate,
                            candidate_boards)
from app.state import SessionState, TimelineManager
from app.tools import CoachTools
from tests.conftest import FakeEngines


# --- P0-3 render-time claim gate (shipped at L0 / shadow) ------------------

_AFTER_E4E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def test_find_illegal_moves_flags_only_impossible():
    # Qh5 is illegal on the starting position (the queen is blocked); Nf3 is legal.
    assert find_illegal_moves("Try Nf3, or even Qh5 right away.", [chess.Board()]) == ["Qh5"]


def test_find_illegal_moves_allows_move_legal_in_any_board():
    # Qh5 IS legal after 1.e4 e5, so with that board in play it must not be flagged.
    assert find_illegal_moves("White can play Qh5.", [chess.Board(), chess.Board(_AFTER_E4E5)]) == []


def test_find_illegal_moves_ignores_board_tokens():
    # Hex inside a board token must not masquerade as a move; e4 is legal.
    assert find_illegal_moves("Here [[BOARD_a1b2c3d4]] then e4.", [chess.Board()]) == []


def test_find_illegal_moves_never_flags_without_known_positions():
    assert find_illegal_moves("Qh5 Nf3 Zz9", []) == []


def test_claim_gate_shadow_is_log_only():
    out, findings = apply_claim_gate("Try Qh5 here.", [chess.Board()], level=0)
    assert out == "Try Qh5 here."                       # unchanged in shadow
    assert findings and findings[0]["token"] == "Qh5"


def test_claim_gate_level1_redacts_illegal_move():
    out, findings = apply_claim_gate("Try Qh5 here.", [chess.Board()], level=1)
    assert "Qh5" not in out and "unverified move" in out and findings


def test_candidate_boards_includes_board_and_agenda():
    from app.review import GameReview, MoveRecord
    rec = MoveRecord(ply=0, number=1, color="white", san="e4", uci="e2e4",
                     fen_before=_AFTER_E4E5, fen_after=chess.STARTING_FEN,
                     eval_before_cp=0, eval_after_cp=0, drop_pct=0.0, quality="best")
    state = SimpleNamespace(board=chess.Board(),
                            review=GameReview(headers={}, records=[rec],
                                              student_color="white", agenda_plies=[0]))
    fens = {b.fen() for b in candidate_boards(state)}
    assert chess.Board().fen() in fens and _AFTER_E4E5 in fens


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send_message(self, payload):
        self.sent.append(payload)
        return self.responses.pop(0)


def _text_response(text):
    part = SimpleNamespace(text=text, function_call=None, functionCall=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=None)


def _tool_response(name, args=None):
    call = SimpleNamespace(name=name, args=args or {})
    return SimpleNamespace(candidates=[], function_calls=[call])


def test_agent_loop_executes_tool_and_renders_final_text(stub_config, capsys):
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat = FakeChat([
        _tool_response("get_game_review_summary"),  # no review loaded -> error dict, fine
        _text_response("Lesson text here."),
    ])
    run_agent_turn(chat, tools, "hello coach")
    assert len(chat.sent) == 2                # message + tool results
    out = capsys.readouterr().out
    assert "Lesson text here." in out


def test_tool_loop_terminates_at_budget(stub_config, capsys):
    # Audit: a model that keeps requesting tools must not loop unboundedly.
    from app.agent_loop import MAX_TOOL_ROUNDS
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat = FakeChat([_tool_response("get_game_review_summary")
                     for _ in range(MAX_TOOL_ROUNDS + 10)])
    run_agent_turn(chat, tools, "hi")
    assert len(chat.sent) == MAX_TOOL_ROUNDS + 1          # initial send + one per round, capped
    assert chat.responses                                  # stopped before exhausting the stream


def test_agent_loop_progress_callback_reaches_review(stub_config):
    """The fetch tool passes the shared progress adapter into review_game."""
    from tests.test_review import PGN, _scripted_engines
    ticks = []
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), _scripted_engines(), stub_config,
                       on_progress=lambda done, total: ticks.append((done, total)))
    result = tools._review_and_load(PGN)
    assert "review_digest" in result
    assert ticks and ticks[-1] == (6, 6)      # one tick per ply, completed


# --- MOVE-STYLING NORMALIZATION (bold/bare moves -> consistent code spans) ---

def test_bold_move_with_number_and_glyph_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("You played **17. d5?!** here") == "You played `17. d5?!` here"


def test_bare_numbered_move_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("Black has just played 16... e5, challenging") == \
        "Black has just played `16... e5`, challenging"


def test_already_backticked_moves_untouched():
    from app.agent_loop import normalize_move_styling
    text = "The reply `16... e5` was strong, and `Nf3` too."
    assert normalize_move_styling(text) == text


def test_italic_move_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("consider *Nf3* instead") == "consider `Nf3` instead"


def test_code_fences_untouched():
    from app.agent_loop import normalize_move_styling
    board = "```text\n1. e4 e5 2. Nf3\n```"
    assert normalize_move_styling(board) == board


def test_ordinary_numbered_list_untouched():
    from app.agent_loop import normalize_move_styling
    text = "1. Look at your rooks. 2. Scan for checks."
    assert normalize_move_styling(text) == text


def test_castling_and_bold_emphasis_on_non_moves_untouched():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("**O-O** was right") == "`O-O` was right"
    assert normalize_move_styling("this is **really important**") == "this is **really important**"
