"""Opt-in message router: the backend grades the student's own words before the coach
model runs, and a router failure leaves the normal flow unchanged."""
import chess

from app.router import Route, resolve_move, resolve_threat
from app.tools import CoachTools
from tests.test_commands import make_processor


class _FakeRouter:
    def __init__(self, route):
        self.route = route
        self.calls = []

    def classify(self, student_text, coach_text, moment):
        self.calls.append((student_text, coach_text, moment))
        return self.route


def _routed_lesson(stub_config, route, status="awaiting_answer"):
    from tests.test_review import PGN as RPGN, _scripted_engines
    proc, state, _ = make_processor(stub_config, _scripted_engines())
    proc.process(RPGN)
    record = state.review.agenda[0]
    state.lesson_metadata = {"last_game_id": state.current_game_id, "last_agenda_index": 0,
                             "lesson_status": status}
    proc.tools = CoachTools(state, proc.timeline, proc.engines, stub_config)
    proc.router = _FakeRouter(route)
    return proc, state, record


def _legal_move(record):
    board = chess.Board(record.fen_before)
    return record.best_move_san or board.san(next(iter(board.legal_moves)))


def test_move_is_graded_once_and_a_repeated_tool_call_does_not_spend_another_attempt(stub_config):
    proc, state, record = _routed_lesson(stub_config, None)
    move = _legal_move(record)
    proc.router.route = Route(label="move", move=move, about_current_moment=True)

    result = proc.process(f"I would play {move}")

    assert "already graded by the backend" in result.prompt
    spent = dict(state.socratic_attempts)
    again = proc.tools.evaluate_student_answer(record.fen_before, move)
    assert again["already_graded_this_turn"] is True
    assert state.socratic_attempts == spent
    assert state.router_events[-1]["action"] == "pregraded:evaluate_student_answer"


def test_next_message_grades_normally_again(stub_config):
    proc, state, record = _routed_lesson(stub_config, None)
    move = _legal_move(record)
    proc.router.route = Route(label="move", move=move, about_current_moment=True)
    proc.process(f"I would play {move}")

    proc.router.route = Route(label="question")
    proc.process("Why is that?")

    assert state.pregraded == {}
    assert "already_graded_this_turn" not in proc.tools.evaluate_student_answer(record.fen_before, move)


def test_move_the_student_did_not_write_is_never_graded(stub_config):
    proc, state, record = _routed_lesson(stub_config, None)
    proc.router.route = Route(label="move", move=_legal_move(record), about_current_moment=True)

    result = proc.process("I'm not sure what to play here")

    assert "already graded" not in result.prompt
    assert state.socratic_attempts == {}


def test_answer_about_another_position_is_not_graded_against_the_current_moment(stub_config):
    proc, state, record = _routed_lesson(stub_config, None)
    move = _legal_move(record)
    proc.router.route = Route(label="move", move=move, about_current_moment=False)

    result = proc.process(f"Back at move 4 I would have played {move}")

    assert "already graded" not in result.prompt
    assert state.socratic_attempts == {}


def test_router_failure_leaves_the_prompt_unchanged(stub_config):
    routed_proc, _, _ = _routed_lesson(stub_config, None)       # classify() fails -> None
    plain_proc, _, _ = _routed_lesson(stub_config, None)
    plain_proc.router = None
    message = "What is Black threatening here?"
    assert routed_proc.process(message).prompt == plain_proc.process(message).prompt


def test_correct_threat_note_keeps_the_better_move_withheld(stub_config):
    # Router run e054prsf, S2: after the backend graded "e4" as the correct threat, the coach
    # revealed the withheld best move (20. Ne4) and graded an explanation the student never gave.
    proc, state, record = _routed_lesson(stub_config, None)
    board = chess.Board(record.fen_before)
    board.push(chess.Move.null())
    threat = board.san(next(iter(board.legal_moves)))
    proc.router.route = Route(label="threat", threat=threat, about_current_moment=True)
    proc.tools.evaluate_threat_answer = lambda fen, threat_move_san: {
        "verdict": "correct_threat", "student_threat": threat_move_san, "reveal_allowed": True}

    prompt = proc.process(f"Black is threatening {threat}").prompt

    assert "ask which move" in prompt and "better move stays withheld" in prompt
    assert "Do not call any grading tool this turn" in prompt


def test_give_up_missed_by_the_rules_is_still_recorded(stub_config):
    proc, state, record = _routed_lesson(stub_config, Route(label="give_up"))

    result = proc.process("No clue, please just show it")

    assert "GAVE UP" in result.prompt
    assert state.socratic_attempts[record.fen_before] == 1


def test_explanation_span_is_graded_alike_in_both_arms_and_its_source_is_recorded(stub_config):
    # Quality-SOL §6.1/§9.4: the router's span and the coach's quoted span reach the same backend grading
    # of the student's whole sentence; an unquoted router span falls back to the message, recorded.
    message = "Hmm, it keeps the knight active. That's my reason."
    routed, routed_state, record = _routed_lesson(stub_config, None, status="awaiting_explanation")
    routed.router.route = Route(label="explanation", explanation="keeps the knight active",
                                about_current_moment=True)
    routed.process(message)
    entry = routed_state.pregraded[f"evaluate_explanation:{record.fen_before}"]
    assert routed_state.router_events[-1]["explanation_source"] == "router_span"

    coach, coach_state, _ = _routed_lesson(stub_config, None, status="awaiting_explanation")
    coach.router = None
    coach.process(message)
    coach_state.latest_user_text = message  # what coach.py passes run_agent_turn as student_text
    direct = coach.tools.evaluate_explanation(record.fen_before, "keeps the knight active")
    assert direct == entry["result"] and direct["graded_text"] == "Hmm, it keeps the knight active."
    assert coach_state.socratic_attempts == routed_state.socratic_attempts

    routed.router.route = Route(label="explanation", explanation="it attacks the queen", about_current_moment=True)
    routed.process("Because it is active.")
    assert routed_state.router_events[-1]["explanation_source"] == "whole_message_fallback"


def test_resolvers_accept_only_quoted_legal_moves():
    board = chess.Board()
    assert resolve_move(board, "knight to f3", "I'd go knight to f3") == "Nf3"
    assert resolve_move(board, "Nf3", "I'd play e4") is None          # not in the message
    assert resolve_move(board, "Qh5", "I'd play Qh5") is None         # illegal here
    board.push_san("e4")
    assert resolve_threat(board, "Qh4", "Black wants Qh4") is None    # not legal for White after a pass
    assert resolve_threat(board, "Qh5", "White threatens Qh5") == "Qh5"
