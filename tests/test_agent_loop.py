"""Tests for the agent loop refactor: the spinner wraps MODEL calls only, so
tools run outside any live display and can show their own progress bars."""
from types import SimpleNamespace

import chess

from app.agent_loop import (run_agent_turn, find_illegal_moves, apply_claim_gate,
                            candidate_boards, _record_tool_candidate_fens,
                            _run_claim_gate, _run_ambiguous_piece_lint,
                            _run_grader_verdict_lint,
                            _run_grading_integrity_guard,
                            _run_verdict_contradiction_lint,
                            _run_deliberation_lint)
from app.notation import ambiguous_piece_references
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


def test_find_illegal_moves_ignores_square_and_invalid_move_explanations():
    text = ("Black doesn't actually have a pawn on e5 to play exd4 here, "
            "and no pawn can move to d4 either.")
    assert find_illegal_moves(text, [chess.Board()]) == []


def test_find_illegal_moves_ignores_square_adjective_piece_references():
    fen = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
    text = ("Ne4 defends your passed pawn on d5. After Nxb5, Black gets pressure "
            "against your d5 pawn.")
    assert find_illegal_moves(text, [chess.Board(fen)]) == []
    out, findings = apply_claim_gate(text, [chess.Board(fen)], level=1)
    assert findings == []
    assert out == text


def test_find_illegal_moves_ignores_square_lists_before_piece_words():
    text = "This left your king with only the f2 and g4 pawns nearby, while the d5 pawn needed support."
    assert find_illegal_moves(text, [chess.Board()]) == []


def test_find_illegal_moves_never_flags_without_known_positions():
    assert find_illegal_moves("Qh5 Nf3 Zz9", []) == []


def test_claim_gate_shadow_is_log_only():
    out, findings = apply_claim_gate("Try Qh5 here.", [chess.Board()], level=0)
    assert out == "Try Qh5 here."                       # unchanged in shadow
    assert findings and findings[0]["token"] == "Qh5"


def test_claim_gate_level1_redacts_illegal_move():
    out, findings = apply_claim_gate("Try Qh5 here.", [chess.Board()], level=1)
    assert "Qh5" not in out and "unverified move" in out and findings


def test_claim_gate_records_structured_l1_observability():
    state = SessionState()
    state.current_game_id = "game-1"
    state.move_index = 3
    config = SimpleNamespace(claim_gate_level=1)

    out = _run_claim_gate("Try Qh5 here.", state, config)

    assert "unverified move" in out
    assert len(state.claim_gate_events) == 1
    event = state.claim_gate_events[0]
    assert event["level"] == 1
    assert event["mode"] == "redacted"
    assert event["redacted"] is True
    assert event["tokens"] == ["Qh5"]
    assert event["candidate_board_count"] >= 1
    assert event["game_id"] == "game-1"
    assert event["move_index"] == 3
    assert "Qh5" in event["before_excerpt"]
    assert "unverified move" in event["after_excerpt"]


def test_claim_gate_records_shadow_observability_without_redacting():
    state = SessionState()
    config = SimpleNamespace(claim_gate_level=0)

    out = _run_claim_gate("Try Qh5 here.", state, config)

    assert out == "Try Qh5 here."
    assert state.claim_gate_events[0]["mode"] == "shadow"
    assert state.claim_gate_events[0]["redacted"] is False
    assert state.claim_gate_events[0]["before_excerpt"] == "Try Qh5 here."
    assert state.claim_gate_events[0]["after_excerpt"] == "Try Qh5 here."


def test_ambiguous_piece_reference_lint_flags_rook_to_d8():
    board = chess.Board("r3r1k1/8/8/8/8/8/8/6K1 b - - 0 1")

    findings = ambiguous_piece_references("Black wants their rook to d8.", board)

    assert len(findings) == 1
    assert findings[0].piece == "rook"
    assert findings[0].square == "d8"
    assert findings[0].candidate_squares == ["a8", "e8"]
    assert ambiguous_piece_references("Black wants Rad8.", board) == []


def test_ambiguous_piece_reference_lint_records_shadow_event():
    from app.review import GameReview, MoveRecord
    fen = "r3r1k1/8/8/8/8/8/8/6K1 b - - 0 1"
    state = SessionState()
    state.review = GameReview(headers={}, records=[
        MoveRecord(ply=0, number=1, color="black", san="Rad8", uci="a8d8",
                   fen_before=fen, fen_after=fen, eval_before_cp=0,
                   eval_after_cp=0, drop_pct=0, quality="best")
    ], student_color="black", agenda_plies=[0])

    _run_ambiguous_piece_lint("Black wants their rook to d8.", state)

    assert state.ambiguous_piece_events
    event = state.ambiguous_piece_events[0]
    assert event["phrase"] == "their rook to d8"
    assert event["candidate_squares"] == ["a8", "e8"]


def test_claim_gate_observability_is_capped():
    state = SessionState()
    config = SimpleNamespace(claim_gate_level=1)

    for i in range(55):
        _run_claim_gate(f"Try Qh5 here {i}.", state, config)

    assert len(state.claim_gate_events) == 50
    assert state.claim_gate_events[0]["before_excerpt"].endswith("5.")
    assert state.claim_gate_events[-1]["before_excerpt"].endswith("54.")


def test_grader_verdict_lint_flags_unreported_verdict():
    state = SessionState()
    state._turn_grading_results = [
        {"tool": "evaluate_threat_answer", "args": {"threat_move_san": "Qg6"},
         "result": {"verdict": "not_a_real_threat", "reveal_allowed": True}}
    ]

    _run_grader_verdict_lint("Let's move on to the next critical moment.", state)

    assert state.grader_verdict_events
    assert state.grader_verdict_events[0]["type"] == "grader_verdict_unreported"
    assert state.grader_verdict_events[0]["verdict"] == "not_a_real_threat"


def test_grader_verdict_lint_accepts_reported_verdict():
    state = SessionState()
    state._turn_grading_results = [
        {"tool": "evaluate_student_answer", "args": {"student_move_san": "Rd4"},
         "result": {"verdict": "best", "reveal_allowed": True}}
    ]

    _run_grader_verdict_lint("Yes, Rd4 is the best move here.", state)

    assert state.grader_verdict_events == []


def test_grading_integrity_guard_blocks_affirming_fabricated_answer():
    state = SessionState()
    state._turn_grading_results = [
        {
            "tool": "evaluate_threat_answer",
            "args": {"threat_move_san": "Nf6"},
            "result": {
                "verdict": "stale_or_fabricated_student_input",
                "error": "submitted grading text does not match latest message",
            },
        }
    ]

    output = _run_grading_integrity_guard("Yes, Nf6 is exactly what Black wants to do.", state)

    assert "Nf6 is exactly" not in output
    assert "not actually in your latest message" in output
    assert state.grading_integrity_events
    assert state.grading_integrity_events[0]["type"] == "stale_grading_response_blocked"


def test_rejected_grading_attempt_superseded_by_an_accepted_one_does_not_block():
    # Run hmrpy5ys S21: evaluate_explanation("pass") was rejected, the model then graded the
    # student's real message and logged the diagnosis, but the guard still replaced the reply.
    state = SessionState()
    fen = "1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31"
    state._turn_grading_results = [
        {"tool": "evaluate_explanation", "args": {"fen": fen, "student_explanation": "pass"},
         "result": {"verdict": "stale_or_fabricated_student_input"}},
        {"tool": "evaluate_explanation", "args": {"fen": fen, "student_explanation": "OK, let's finish."},
         "result": {"gradable": True, "key_reason_named": False}},
    ]

    assert _run_grading_integrity_guard("Kh2 was the move: it covers h3.", state) == "Kh2 was the move: it covers h3."
    assert not state.grading_integrity_events


def test_verdict_contradiction_lint_flags_praised_blunder():
    from app.review import GameReview, MoveRecord
    fen = "1r2r1k1/p4ppp/6q1/2QP4/1P4P1/P7/5PK1/3R1R2 w - - 3 26"
    state = SessionState()
    state.review = GameReview(headers={}, records=[
        MoveRecord(ply=60, number=31, color="white", san="Kh1", uci="g2h1",
                   fen_before=fen, fen_after=fen, eval_before_cp=0, eval_after_cp=-600,
                   drop_pct=33.7, quality="blunder", best_move_san="Rd4")
    ], student_color="white", agenda_plies=[0])

    _run_verdict_contradiction_lint("31. Kh1 is the best move and holds the position.", state)

    assert state.verdict_contradiction_events
    assert state.verdict_contradiction_events[0]["claim"] == "praised_played_mistake"


def test_deliberation_lint_flags_raw_thought_and_fake_checking():
    state = SessionState()
    state._turn_tool_names = []

    _run_deliberation_lint("But wait, that's confusing. Let me check the engine line again.", state)

    event_types = [event["type"] for event in state.deliberation_events]
    assert "deliberation_marker" in event_types
    assert "claimed_verification_without_tool" in event_types


def test_deliberation_lint_allows_checking_language_after_tool_call():
    state = SessionState()
    state._turn_tool_names = ["analyze_position"]

    _run_deliberation_lint("I checked the engine line and the point is Qxg4+.", state)

    assert not any(event["type"] == "claimed_verification_without_tool"
                   for event in state.deliberation_events)


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


def test_candidate_boards_include_previous_move_origin_for_orientation_text():
    from app.review import GameReview, MoveRecord
    rec = MoveRecord(ply=1, number=1, color="black", san="e5", uci="e7e5",
                     fen_before=_AFTER_E4E5, fen_after=_AFTER_E4E5,
                     eval_before_cp=0, eval_after_cp=0, drop_pct=0.0, quality="best",
                     prev_move_san="e5", history_uci=["e2e4", "e7e5"])
    state = SimpleNamespace(board=chess.Board(),
                            review=GameReview(headers={}, records=[rec],
                                              student_color="white", agenda_plies=[0]))

    boards = candidate_boards(state)

    assert find_illegal_moves("Black has just played e5.", boards) == []


def test_candidate_boards_include_verified_line_positions_for_l1_gate():
    from app.review import GameReview, MoveRecord
    rec = MoveRecord(ply=0, number=1, color="white", san="d4", uci="d2d4",
                     fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                     eval_before_cp=0, eval_after_cp=0, drop_pct=0.0, quality="best",
                     best_line=["e4", "e5", "Qh5"])
    state = SimpleNamespace(board=chess.Board(),
                            review=GameReview(headers={}, records=[rec],
                                              student_color="white", agenda_plies=[0]))
    boards = candidate_boards(state)
    assert find_illegal_moves("The line can continue Qh5.", boards) == []


def test_candidate_boards_include_tool_returned_pv_positions():
    state = SessionState()
    _record_tool_candidate_fens(
        state,
        "analyze_position",
        {"fen": chess.STARTING_FEN},
        {"top_stockfish_lines": [{"rank": 1, "expected_engine_line": "e4 e5 Qh5"}]},
    )

    boards = candidate_boards(state)

    assert find_illegal_moves("The returned line continues Qh5.", boards) == []


def test_candidate_boards_include_tool_returned_threat_null_move_positions():
    board = chess.Board()
    for san in ("f3", "e5"):
        board.push_san(san)
    state = SessionState()

    _record_tool_candidate_fens(
        state,
        "evaluate_threat_answer",
        {"fen": board.fen()},
        {"actual_threats": [{"threat": "queen check", "threat_line": "Qh4+"}]},
    )

    boards = candidate_boards(state)
    assert find_illegal_moves("Black was threatening Qh4+.", boards) == []


def test_candidate_boards_include_every_position_of_the_loaded_game():
    # Recorded 2026-10-12: "8. cxd3!" and "23. hxg4" (both played in the game, outside any
    # agenda moment) were redacted. Real game moves must always be verifiable.
    from app.review import GameReview, MoveRecord
    board = chess.Board()
    for san in ("e4", "d5", "Nf3"):
        board.push_san(san)
    before = board.fen()  # black can play dxe4 here
    after = board.copy(stack=False)
    after.push_san("dxe4")
    rec = MoveRecord(ply=3, number=2, color="black", san="dxe4", uci="d5e4",
                     fen_before=before, fen_after=after.fen(),
                     eval_before_cp=0, eval_after_cp=0, drop_pct=0.0, quality="ok")
    state = SessionState()
    state.review = GameReview(headers={}, records=[rec], student_color="white", agenda_plies=[])

    boards = candidate_boards(state)
    assert find_illegal_moves("Black played 2... dxe4, and the f2 and g4 pawns mattered.", boards) == []


_MOVE24_FEN = "r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5Pp1/3R1RK1 w - - 0 24"
_MOVE4_FEN = "rn1qkbnr/ppp2ppp/4p3/3p1b2/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq - 0 4"


def test_bare_squares_in_prose_are_not_move_claims():
    # Recorded 2026-10-12 replies, verbatim fragments; every square here was redacted.
    move24 = [chess.Board(_MOVE24_FEN)]
    assert find_illegal_moves("After `24. Rfe1`, what tactical idea around e1 changes?", move24) == []
    texts = [
        "jumping the knight to b5 to attack the c7 pawn. You are threatening to take on c7, "
        "but Black can easily defend it. Once Black defends c7, your knight on b5 isn't doing much.",
        "their king on g8 still had its original f7, g7, and h7 pawns sitting in front of it.",
        "Black's h7 pawn was still there, sitting safely on g7 and h7, blocking the files.",
    ]
    for text in texts:
        assert find_illegal_moves(text, [chess.Board(_MOVE4_FEN)]) == []


def test_numbered_or_code_span_pawn_moves_are_still_checked():
    start = [chess.Board()]
    assert find_illegal_moves("Now 1. e5 grabs space.", start) == ["e5"]
    assert find_illegal_moves("The engine wants `e5` here.", start) == ["e5"]


def test_redaction_replaces_only_flagged_occurrences():
    out, findings = apply_claim_gate("Play 1. c5 to put pressure on the c5 square.", [chess.Board()], level=1)
    assert [f["token"] for f in findings] == ["c5"]
    assert out == "Play 1. (an unverified move) to put pressure on the c5 square."


def test_claim_gate_verifies_moves_from_dossier_positions_without_a_loaded_game(tmp_path):
    # Recorded 2026-10-12 after /memorize: "24. Kxg2, 26. Qxa7, and 31. Kh1" were all
    # redacted because no game was loaded; they are the dossier's own diagnosed positions.
    import json
    dossier = tmp_path / "player_dossier.json"
    dossier.write_text(json.dumps({"recent_positions": [
        {"fen": _MOVE24_FEN, "error_category": "ignored_threat"}]}))
    state = SessionState()
    config = SimpleNamespace(claim_gate_level=1, dossier_file=str(dossier))
    assert _run_claim_gate("We looked at 24. Kxg2 last time.", state, config) == "We looked at 24. Kxg2 last time."


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


# --- Leaked model reasoning (live eval 2026-10-12, run par4cj6m) -------------
# Gemini began returning its reasoning as an ordinary text part ("s Thinking Process:
# 1. ..."). One leaked reply in the chat history made every later reply leak (replays:
# 14/14 leaked with it, 0/4 from clean history), and the drafts revealed withheld answers.

_LEAK = "s Thinking Process:\n\n1.  **Analyze the Tool Response:**\n    *   Best move: Rd4\n"


class _ScriptedModels:
    """Stands in for the network call under a real google.genai Chat."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append(list(contents))
        return self.responses.pop(0)


def _gemini_response(*parts):
    from google.genai import types
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)))])


def _text(text):
    from google.genai import types
    return types.Part(text=text)


def _call(name):
    from google.genai import types
    return types.Part(function_call=types.FunctionCall(name=name, args={}))


def _real_chat(responses):
    from google.genai import chats, types
    models = _ScriptedModels(responses)
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    return chats.Chat(modules=models, model="gemini-test", config=config, history=[]), models


def _history_text(chat):
    return " ".join(p.text or "" for c in chat.get_history() for p in (c.parts or []))


def test_leaked_reasoning_reply_is_regenerated_and_never_kept(stub_config, capsys):
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat, models = _real_chat([_gemini_response(_text(_LEAK + "Draft: Rd4 wins.")),
                               _gemini_response(_text("Which move defends g4?"))])

    run_agent_turn(chat, tools, "hello coach")

    out = capsys.readouterr().out
    assert "Which move defends g4?" in out
    assert "Thinking Process" not in out and "Rd4" not in out
    assert "Thinking Process" not in _history_text(chat)
    assert "Thinking Process" not in " ".join(p.text or "" for c in models.requests[1] for p in c.parts or [])
    assert [c.role for c in chat.get_history()] == ["user", "model"]
    assert [c.role for c in chat.get_history(curated=True)] == ["user", "model"]


def test_reasoning_leaked_twice_shows_a_retry_notice_and_keeps_history_valid(stub_config, capsys):
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat, _ = _real_chat([_gemini_response(_call("get_game_review_summary")),
                          _gemini_response(_text(_LEAK + "Draft one.")),
                          _gemini_response(_text(_LEAK + "Draft two."))])

    run_agent_turn(chat, tools, "hello coach")

    out = capsys.readouterr().out
    assert "Thinking Process" not in out and "Draft" not in out
    assert "send your last message again" in " ".join(out.split())
    assert "Thinking Process" not in _history_text(chat)
    # Each function call is still followed by its function response, then a model turn.
    assert [c.role for c in chat.get_history()] == ["user", "model", "user", "model"]


def test_reasoning_beside_a_tool_call_is_dropped_from_history_without_a_resend(stub_config, capsys):
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat, models = _real_chat([_gemini_response(_text(_LEAK), _call("get_game_review_summary")),
                               _gemini_response(_text("Here is the summary."))])

    run_agent_turn(chat, tools, "hello coach")

    assert len(models.requests) == 2
    assert "Thinking Process" not in _history_text(chat)
    assert any(p.function_call for p in chat.get_history()[1].parts)
    assert "Here is the summary." in capsys.readouterr().out


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
