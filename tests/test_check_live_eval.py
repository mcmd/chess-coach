import copy
import json

import chess
import chess.pgn
import pytest
from rich.text import Text

from app.render import render_board_markup
from scripts.check_live_eval import check_recording, main


# The checks a run performs are declared in its player-script snapshot.
SCRIPT = """### S4 — Ask about an earlier position

**Checks:** detour

**Send:**
```text
Wait, go back to the previous position for a second.
```

### S5 — Return to the pending position

**Checks:** detour, return 24

**Send:**
```text
Thanks. Let's return to move 24, just after 23...fxg2.
```

### S6 — Stop
"""

class Recording:
    def __init__(self, root, status="SCRIPT_COMPLETE"):
        self.root = root
        self.events = []
        self.execution = {"status": status, "sent_steps": []}
        self.add("session_start", config={"model": "recorded-coach"})
        self.add("ready")

    def add(self, kind, turn=0, **data):
        event = {"schema_version": 1, "id": len(self.events) + 1,
                 "timestamp": "2026-10-06T12:00:00+00:00", "kind": kind,
                 "turn": turn, "data": data}
        self.events.append(event)
        return event

    def turn(self, turn, state, text="Explain this position", reply="A recorded reply", raw=None):
        self.add("student", turn, text=text)
        self.add("turn_start", turn, prompt=text, state=copy.deepcopy(state))
        self.add("model_request", turn, payload={"messages": [{"role": "user", "text": text}]})
        self.add("model_response", turn, response={"text": reply if raw is None else raw})
        event = self.add("coach_reply", turn, text=reply, state=copy.deepcopy(state))
        self.add("turn_end", turn, state=copy.deepcopy(state))
        return event

    def tool(self, name, arguments, result, state=None, engine=False, turn=1, error=None):
        prefix = "engine" if engine else "tool"
        call_id = f"call-{len(self.events)}"
        call = self.add(prefix + "_call", turn, call_id=call_id, name=name,
                        arguments=arguments, state=copy.deepcopy(state or snapshot()))
        data = {"call_id": call_id, "name": name, "result": result,
                "state": copy.deepcopy(state or snapshot())}
        if error:
            data["error"] = error
        output = self.add(prefix + "_result", turn, **data)
        return call, output

    def diagram(self, board, turn=1, pov="white", token=None, rendered_board=None):
        token = token or f"[[BOARD_{len(self.events):08x}]]"
        markup = render_board_markup(rendered_board or board, pov)
        return self.add("board_diagram", turn, token=token, fen=board.fen(), pov=pov,
                        text=Text.from_markup(markup).plain, markup=markup)

    def save(self, final=True, suffix=""):
        if final:
            self.add("session_end", status="normal", exit_code=0, state=snapshot())
        (self.root / "events.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self.events) + suffix,
            encoding="utf-8")
        (self.root / "execution.json").write_text(json.dumps(self.execution), encoding="utf-8")
        (self.root / "2dtDqQgR_script.md").write_text(SCRIPT, encoding="utf-8")
        return check_recording(self.root)


def snapshot(board=None, **overrides):
    state = {"board_fen": (board or chess.Board()).fen(), "loaded_moves_uci": [],
             "move_index": 0, "current_game_id": None, "lesson_metadata": {},
             "diagnostics": [], "socratic_attempts": {}, "review_agenda": []}
    state.update(overrides)
    return state


def findings(result, check):
    return [item for item in result["checks"] if item["check"] == check]


def verdicts(result, check):
    return {item["verdict"] for item in findings(result, check)}


def source_game(recording):
    game = chess.pgn.Game()
    game.headers.update({"Site": "https://lichess.org/2dtDqQgR", "White": "mcmd", "Black": "opponent"})
    node = game
    board = game.board()
    positions = [board.copy()]
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"] * 12:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
        positions.append(board.copy(stack=False))
    event = recording.add("http_response", url="https://lichess.org/game/export/2dtDqQgR",
                          status=200, body=str(game), params={})
    return game, positions, event


def game_state(game, positions, index):
    return snapshot(positions[index], current_game_id="2dtDqQgR", move_index=index,
                    loaded_moves_uci=[m.uci() for m in game.mainline_moves()],
                    review_headers=dict(game.headers), lesson_metadata={"last_agenda_index": 1,
                    "lesson_status": "awaiting_answer", "last_game_id": "2dtDqQgR"},
                    diagnostics=[{"fen": positions[38].fen(), "error_category": "activity"}],
                    socratic_attempts={positions[38].fen(): 1},
                    review_agenda=[{"fen_before": positions[38].fen(), "fen_after": positions[39].fen(),
                                    "ply": 38, "san": "Ng1", "move_label": "20. Ng1"}])


def test_complete_evidence_is_not_an_overall_coaching_pass(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot())
    result = recording.save()
    assert not result["recording_gaps"]
    assert verdicts(result, "recording_completeness") == {"PASS"}
    assert verdicts(result, "pv_legality") == {"NOT OBSERVED"}
    assert verdicts(result, "execution") == {"NOT OBSERVED"}
    assert "overall" not in result


@pytest.mark.parametrize("line", ["1. e4 e5 2. Nf3 Qh5", "Continuation: e4 e5 Nf3 Qh5"])
def test_illegal_consumer_visible_line_is_detected(tmp_path, line):
    recording = Recording(tmp_path)
    reply = recording.turn(1, snapshot(), reply=line, raw="1. e4 e5 2. Nf3 Nc6")
    result = recording.save()
    failed = [c for c in findings(result, "visible_line_legality") if c["verdict"] == "FAIL"]
    assert failed and reply["id"] in failed[0]["evidence"]
    assert "Qh5" in failed[0]["explanation"]


def test_raw_model_line_is_not_student_visible_evidence(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot(), reply="1. e4 e5 2. Nf3 Nc6", raw="1. e4 e5 2. Nf3 Qh5")
    result = recording.save()
    assert verdicts(result, "visible_line_legality") == {"PASS"}
    assert "best, likely, or forced" in findings(result, "visible_line_legality")[0]["explanation"]


def test_earlier_position_line_uses_source_not_current_board(tmp_path):
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    recording.turn(1, game_state(game, positions, 46), reply="From the earlier position: 1. Nf3 Nf6 2. Ng1 Ng8")
    result = recording.save()
    assert verdicts(result, "visible_line_legality") == {"PASS"}
    assert source["id"] in findings(result, "visible_line_legality")[0]["evidence"]


def test_verified_hypothetical_line_is_legal_and_unverified_one_is_not(tmp_path):
    # Recorded live eval (#41): "After 25. Rfe1 Qxg4+ ..." narrated the verified punishment
    # line, but the checker judged it from the moves actually played at move 25.
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    state = game_state(game, positions, 2)
    state["review_agenda"] = [{"fen_before": positions[0].fen(), "fen_after": positions[1].fen(),
                               "ply": 0, "san": "Nf3", "move_label": "1. Nf3",
                               "punishment_line": ["d5", "d4", "Bf5"], "best_line": []}]
    recording.turn(1, state, reply="Black answers 1... d5, and after 2. d4 Bf5 Black is comfortable.")
    recording.turn(2, state, reply="Black answers 1... d5, and after 2. d4 Bb4 Black is comfortable.")
    result = recording.save()
    by_turn = {next(e["turn"] for e in recording.events if e["id"] == c["evidence"][0]): c["verdict"]
               for c in findings(result, "visible_line_legality")}
    assert by_turn == {1: "PASS", 2: "FAIL"}


def test_a_line_from_the_better_moves_refutation_of_a_resource_is_known(tmp_path):
    # Run nnbbjvjq: "24... Qf4 25. Rxe8+ Rxe8 26. Qxe8#" (the slice B lesson after 24. Rfe1) was judged
    # illegal because only the game, best and punishment lines were known positions.
    def run(with_refutation):
        (tmp_path / str(with_refutation)).mkdir()
        recording = Recording(tmp_path / str(with_refutation))
        game, positions, source = source_game(recording)
        state = game_state(game, positions, 2)
        record = {"fen_before": positions[0].fen(), "fen_after": positions[1].fen(), "ply": 0, "san": "Nf3",
                  "move_label": "1. Nf3", "best_move_san": "e4", "best_line": [], "punishment_line": ["e5"]}
        if with_refutation:
            record["resource_refutation"] = {"resource": "e5", "line": ["Nf3", "Nc6"], "alternatives": []}
        state["review_agenda"] = [record]
        recording.turn(1, state, reply="After 1. e4, the reply 1... e5 runs into 2. Nf3 Nc6.")
        return {c["verdict"] for c in findings(recording.save(), "visible_line_legality")}
    assert run(True) == {"PASS"} and "FAIL" in run(False)


def test_a_line_analyze_position_returned_is_a_known_position(tmp_path):
    # Run bmcabgeq narrated the 31. Kh2 continuation ("32. Kg2 ... 32... Qg5+") from an analyze_position
    # line; with no known position at move 32 the checker listed recording gaps instead of checking it.
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    state = game_state(game, positions, 2)
    recording.tool("analyze_position", {"fen": chess.STARTING_FEN, "moves_to_apply": ["e4"]},
                   {"top_stockfish_lines": [{"expected_engine_line": "e5 Nf3 Nc6"}]})
    recording.turn(1, state, reply="After 1. e4 e5 the game goes 2. Nf3 Nc6.")
    recording.turn(2, state, reply="After 1. e4 e5 the game goes 2. Nf3 Nf6.")
    result = recording.save()
    by_turn = {next(e["turn"] for e in recording.events if e["id"] == c["evidence"][0]): c["verdict"]
               for c in findings(result, "visible_line_legality")}
    assert by_turn == {1: "PASS", 2: "PASS"} and not result.get("recording_gaps")


def test_single_numbered_move_in_prose_is_checked(tmp_path):
    # Run par4cj6m: "after Black recaptures with 20... Qxb5" was illegal (the queen on d6
    # cannot reach b5) but the checker only read numbered lines of two or more moves.
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    state = game_state(game, positions, 2)
    recording.turn(1, state, reply="After you played 1. Nf3, Black answered with 1... Nf6.")
    recording.turn(2, state, reply="After you played 1. Nf3, Black recaptured with 1... Qxb5.")
    # A ranked engine list is not a move number (run par4cj6m: "1. Qd7 (+4.79), 2. Qg6 (+5.34)").
    recording.turn(3, state, reply="Top engine lines: 1. Qd7 (+4.79), 2. Qg6 (+5.34).")
    result = recording.save()
    by_turn = {next(e["turn"] for e in recording.events if e["id"] == c["evidence"][0]): c["verdict"]
               for c in findings(result, "visible_line_legality")}
    assert by_turn[2] == "FAIL" and "Qxb5" in next(
        c["explanation"] for c in findings(result, "visible_line_legality") if c["verdict"] == "FAIL")
    assert by_turn[1] == "PASS" and 3 not in by_turn


def test_free_prose_moves_without_line_context_are_not_naively_graded(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot(), reply="Earlier Ne4 attacked the queen. Qh5 might matter in another position.")
    result = recording.save()
    assert verdicts(result, "visible_line_legality") == {"NOT OBSERVED"}


def test_hypothetical_illegal_move_remains_a_separate_finding_from_tool_error(tmp_path):
    recording = Recording(tmp_path)
    call, output = recording.tool("analyze_position", {"fen": chess.STARTING_FEN,
        "moves_to_apply": ["e4", "e5", "Nf3", "Qh5"]}, {"error": "Move Qh5 is illegal"})
    result = recording.save()
    assert verdicts(result, "hypothetical_legality") == {"FAIL"}
    assert verdicts(result, "tool_availability") == {"BLOCKED"}
    assert findings(result, "hypothetical_legality")[0]["evidence"] == [call["id"], output["id"]]


def test_tool_infrastructure_failure_does_not_fabricate_illegal_pv(tmp_path):
    recording = Recording(tmp_path, "INFRASTRUCTURE_BLOCKED")
    recording.tool("analyze_position", {"fen": chess.STARTING_FEN,
        "moves_to_apply": ["e4", "e5"]}, {"error": "Engine unavailable"})
    result = recording.save()
    assert verdicts(result, "hypothetical_legality") == {"PASS"}
    assert verdicts(result, "pv_legality") == {"BLOCKED"}
    assert not any(c["verdict"] == "FAIL" for c in findings(result, "tool_availability"))


@pytest.mark.parametrize("name,move,result,expected", [
    ("evaluate_student_answer", "e4", {"verdict": "best"}, "PASS"),
    ("evaluate_student_answer", "Qh5", {"verdict": "illegal"}, "PASS"),
    ("evaluate_student_answer", "Qh5", {"verdict": "best"}, "FAIL"),
    ("evaluate_threat_answer", "e5", {"verdict": "correct_threat"}, "PASS"),
    ("evaluate_threat_answer", "e4", {"verdict": "correct_threat"}, "FAIL"),
    ("evaluate_student_answer", "e5", {"verdict": "correct_threat",
        "interpreted_as": "opponent_threat"}, "PASS"),
])
def test_grader_legal_moves_and_null_move_threats(tmp_path, name, move, result, expected):
    recording = Recording(tmp_path)
    key = "threat_move_san" if name == "evaluate_threat_answer" else "student_move_san"
    recording.tool(name, {"fen": chess.STARTING_FEN, key: move}, result)
    checked = recording.save()
    assert verdicts(checked, "grader_legality") == {expected}


def test_engine_uci_pv_and_tool_san_pv_are_checked_in_sequence(tmp_path):
    recording = Recording(tmp_path)
    recording.tool("top_lines", {"board": {"fen": chess.STARTING_FEN, "moves_uci": []}},
                   [{"pv": ["e2e4", "e7e5", "g1f3", "d8h5"]}], engine=True)
    recording.tool("analyze_position", {"fen": chess.STARTING_FEN, "moves_to_apply": ["e4", "e5"]},
                   {"top_stockfish_lines": [{"expected_engine_line": "Nf3 Nc6"}],
                    "threats_if_side_to_move_passes": [{"threat_line": "Nf6 Nc3"}]})
    result = recording.save()
    assert verdicts(result, "pv_legality") == {"PASS", "FAIL"}
    assert any("d8h5" in c["explanation"] for c in findings(result, "pv_legality") if c["verdict"] == "FAIL")


def test_absent_fen_blocks_legality_without_guessing_starting_position(tmp_path):
    recording = Recording(tmp_path)
    recording.tool("analyze_position", {"moves_to_apply": ["e4", "Qh5"]},
                   {"top_stockfish_lines": [{"expected_engine_line": "Qh5 Nf6"}]})
    result = recording.save()
    assert verdicts(result, "hypothetical_legality") == {"NOT OBSERVED"}
    assert verdicts(result, "pv_legality") == {"NOT OBSERVED"}


@pytest.mark.parametrize("pov", ["white", "black"])
def test_diagram_backend_surface_and_piece_layout_match_actual_fen(tmp_path, pov):
    recording = Recording(tmp_path)
    diagram = recording.diagram(chess.Board(), pov=pov)
    reply = recording.turn(1, snapshot(), reply=diagram["data"]["text"], raw=diagram["data"]["token"])
    result = recording.save()
    assert verdicts(result, "diagram_provenance") == {"PASS"}
    assert findings(result, "diagram_provenance")[0]["evidence"] == [diagram["id"], reply["id"]]


def test_diagram_with_correct_token_and_wrong_piece_layout_fails(tmp_path):
    recording = Recording(tmp_path)
    changed = chess.Board()
    changed.push_san("e4")
    diagram = recording.diagram(chess.Board(), rendered_board=changed)
    recording.turn(1, snapshot(), reply=diagram["data"]["text"], raw=diagram["data"]["token"])
    result = recording.save()
    assert verdicts(result, "diagram_provenance") == {"FAIL"}


def test_unrecorded_diagram_and_leaked_token_are_failures(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot(), reply=Text.from_markup(render_board_markup(chess.Board())).plain + "\n[[BOARD_missing]]")
    result = recording.save()
    assert verdicts(result, "diagram_provenance") == {"FAIL"}


@pytest.mark.parametrize("wrong_board", [False, True])
def test_detour_return_board_and_diagram_are_checked_against_source_pgn(tmp_path, wrong_board):
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    pending = game_state(game, positions, 46)
    recording.execution["sent_steps"] = [{"step": "S4", "turn": 4}, {"step": "S5", "turn": 5}]
    recording.turn(4, pending, reply="We are discussing the earlier move, not grading another answer.")
    returned = game_state(game, positions, 38) if wrong_board else pending
    diagram = recording.diagram(positions[38] if wrong_board else positions[46], turn=5)
    recording.turn(5, returned, reply=diagram["data"]["text"], raw=diagram["data"]["token"])
    result = recording.save()
    expected = "FAIL" if wrong_board else "PASS"
    assert verdicts(result, "return_position") == {expected}
    assert verdicts(result, "diagram_source_consistency") == {expected}
    assert source["id"] in findings(result, "return_position")[0]["evidence"]
    assert verdicts(result, "detour_lesson_state") == {"PASS"}


def test_wrong_visible_return_diagram_is_detected_even_when_physical_board_is_correct(tmp_path):
    recording = Recording(tmp_path)
    game, positions, _ = source_game(recording)
    pending = game_state(game, positions, 46)
    recording.execution["sent_steps"] = [{"step": "S4", "turn": 4}, {"step": "S5", "turn": 5}]
    recording.turn(4, pending)
    diagram = recording.diagram(positions[0], turn=5)
    recording.turn(5, pending, reply=diagram["data"]["text"], raw=diagram["data"]["token"])
    result = recording.save()
    assert verdicts(result, "return_position") == {"PASS"}
    assert verdicts(result, "diagram_source_consistency") == {"FAIL"}


def test_lesson_diagram_can_be_move_24_while_physical_timeline_stays_at_start(tmp_path):
    recording = Recording(tmp_path)
    game, positions, _ = source_game(recording)
    pending = game_state(game, positions, 0)
    pending["lesson_metadata"] = {"last_agenda_index": 1}
    recording.execution["sent_steps"] = [{"step": "S4", "turn": 4}, {"step": "S5", "turn": 5}]
    recording.turn(4, pending)
    diagram = recording.diagram(positions[46], turn=5)
    recording.turn(5, pending, reply=diagram["data"]["text"], raw=diagram["data"]["token"])
    result = recording.save()
    assert verdicts(result, "return_position") == {"PASS"}
    assert verdicts(result, "diagram_source_consistency") == {"PASS"}
    assert verdicts(result, "detour_lesson_state") == {"PASS"}


def test_detour_agenda_advancement_and_duplicate_diagnosis_fail_without_double_counting_snapshots(tmp_path):
    recording = Recording(tmp_path)
    state = snapshot(diagnostics=[{"fen": chess.STARTING_FEN}],
                     lesson_metadata={"last_agenda_index": 1})
    recording.execution["sent_steps"] = [{"step": "S4", "turn": 4}]
    recording.turn(4, state)
    changed = copy.deepcopy(state)
    changed["lesson_metadata"]["last_agenda_index"] = 2
    changed["diagnostics"].append({"fen": chess.STARTING_FEN})
    recording.add("tool_result", 4, call_id="diagnosis", name="log_diagnosis", result={"success": True}, state=changed)
    result = recording.save()
    assert verdicts(result, "duplicate_diagnoses") == {"PASS", "FAIL"}
    assert verdicts(result, "detour_lesson_state") == {"PASS", "FAIL"}


def test_source_pgn_header_and_move_list_mismatch_are_evidence_linked(tmp_path):
    recording = Recording(tmp_path)
    game, positions, source = source_game(recording)
    state = game_state(game, positions, 46)
    state["review_headers"]["White"] = "someone-else"
    state["loaded_moves_uci"] = ["e2e4"]
    reply = recording.turn(1, state)
    result = recording.save()
    failed = [c for c in findings(result, "pgn_consistency") if c["verdict"] == "FAIL"]
    assert failed
    assert any(source["id"] in c["evidence"] and reply["id"] in c["evidence"] for c in failed)


def test_interrupted_jsonl_preserves_valid_previous_chess_checks(tmp_path):
    recording = Recording(tmp_path, "TIMED_OUT")
    recording.tool("analyze_position", {"fen": chess.STARTING_FEN, "moves_to_apply": ["e4", "e5"]}, {})
    recording.add("student", 2, text="A pending question")
    result = recording.save(final=False, suffix='{"schema_version":1,"id":99,"kind":"model_response"')
    assert verdicts(result, "hypothetical_legality") == {"PASS"}
    assert verdicts(result, "execution") == {"BLOCKED"}
    assert verdicts(result, "return_position") == {"BLOCKED"}
    assert any("interrupted JSONL" in gap for gap in result["recording_gaps"])
    assert any("Turn 2" in gap and "interrupted" in gap for gap in result["recording_gaps"])


def test_missing_rendered_reply_payload_is_not_substituted_with_model_text(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot(), raw="1. e4 e5")
    recording.events = [e for e in recording.events if e["kind"] != "coach_reply"]
    # Keep IDs monotonic without claiming the omitted event was recorded.
    result = recording.save(final=False)
    assert any("no coach_reply" in gap for gap in result["recording_gaps"])
    assert verdicts(result, "visible_line_legality") == {"NOT OBSERVED"}
    assert verdicts(result, "recording_completeness") == {"FAIL"}


def test_explicit_truncation_and_unknown_payload_are_recording_gaps(tmp_path):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot())
    next(e for e in recording.events if e["kind"] == "model_response")["data"]["response"] = {"truncated": True, "text": "short"}
    result = recording.save()
    assert any("truncated" in gap for gap in result["recording_gaps"])
    assert verdicts(result, "recording_completeness") == {"FAIL"}


def test_backend_handled_turn_needs_no_model_evidence(tmp_path):
    # Live eval t3hufhwt S22: "Annotate every move" is declined by the command layer with
    # no model call (recorded prompt=None), which was flagged as a recording gap.
    recording = Recording(tmp_path)
    recording.add("student", 1, text="Annotate every move of the game.")
    recording.add("turn_start", 1, prompt=None, state=snapshot())
    recording.add("coach_reply", 1, text="I won't annotate every move.", state=snapshot())
    recording.add("turn_end", 1, state=snapshot())
    recording.add("student", 2, text="Explain this position")
    recording.add("turn_start", 2, prompt="Explain this position", state=snapshot())
    recording.add("model_request", 2, payload={"messages": []})
    recording.add("coach_reply", 2, text="A recorded reply", state=snapshot())
    recording.add("turn_end", 2, state=snapshot())
    result = recording.save()
    assert result["recording_gaps"] == ["Completed turn 2 has no model_response evidence"]


@pytest.mark.parametrize("status", ["SCRIPT_COMPLETE", "STOPPED_UNEXPECTED", "TIMED_OUT", "INFRASTRUCTURE_BLOCKED"])
def test_execution_status_never_overwrites_observed_quality_failure(tmp_path, status):
    recording = Recording(tmp_path, status)
    recording.turn(1, snapshot(), reply="1. e4 e5 2. Nf3 Qh5")
    result = recording.save()
    assert result["execution_status"] == status
    assert verdicts(result, "visible_line_legality") == {"FAIL"}
    assert verdicts(result, "execution") == ({"BLOCKED"} if status in ("TIMED_OUT", "INFRASTRUCTURE_BLOCKED") else {"NOT OBSERVED"})


def test_missing_recording_produces_uncertainty_not_blanket_pass(tmp_path):
    result = check_recording(tmp_path)
    assert result["recording_gaps"]
    assert verdicts(result, "return_position") == {"NOT OBSERVED"}
    assert not any(c["verdict"] == "PASS" for c in result["checks"])


def test_cli_writes_checks_json_and_does_not_modify_evidence(tmp_path, capsys):
    recording = Recording(tmp_path)
    recording.turn(1, snapshot(), reply="1. e4 e5 2. Nf3 Nc6")
    recording.save()
    originals = {name: (tmp_path / name).read_bytes() for name in ("events.jsonl", "execution.json")}
    assert main([str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads((tmp_path / "checks.json").read_text())
    assert printed == written
    assert verdicts(written, "visible_line_legality") == {"PASS"}
    assert originals == {name: (tmp_path / name).read_bytes() for name in originals}


def test_review_agenda_pvs_use_actual_fens_without_network_pgn(tmp_path):
    recording = Recording(tmp_path)
    after = chess.Board()
    after.push_san("e4")
    state = snapshot(review_agenda=[{"fen_before": chess.STARTING_FEN,
                     "fen_after": after.fen(), "best_line": ["e4", "e5"],
                     "punishment_line": ["Qh5"]}])
    recording.turn(1, state)
    result = recording.save()
    assert verdicts(result, "pv_legality") == {"PASS", "FAIL"}
    assert verdicts(result, "pgn_consistency") == {"NOT OBSERVED"}


def test_turn_error_preserves_prior_legality_and_blocks_unavailable_checks(tmp_path):
    recording = Recording(tmp_path, "INFRASTRUCTURE_BLOCKED")
    recording.tool("top_lines", {"board": {"fen": chess.STARTING_FEN}},
                   [{"pv": ["e2e4", "e7e5"]}], engine=True)
    recording.add("student", 1, text="A real submitted question")
    recording.add("turn_start", 1, prompt="A real submitted question", state=snapshot())
    recording.add("turn_error", 1, state=snapshot(),
                  error={"type": "ConnectionError", "message": "Actual request failed"})
    result = recording.save(final=False)
    assert verdicts(result, "pv_legality") == {"PASS"}
    assert verdicts(result, "session_lifecycle") == {"BLOCKED"}
    assert verdicts(result, "return_position") == {"BLOCKED"}
    assert not any(c["verdict"] == "FAIL" for c in findings(result, "pv_legality"))


def test_unknown_lesson_snapshot_does_not_pass_detour_consistency(tmp_path):
    recording = Recording(tmp_path)
    recording.execution["sent_steps"] = [{"step": "S4", "turn": 4}]
    recording.turn(4, snapshot(lesson_metadata=None, socratic_attempts=None))
    result = recording.save()
    assert verdicts(result, "detour_lesson_state") == {"NOT OBSERVED"}


def test_multiple_student_candidates_are_not_misreported_as_illegal_grading(tmp_path):
    recording = Recording(tmp_path)
    recording.tool("evaluate_student_answer", {"fen": chess.STARTING_FEN,
                   "student_move_san": "e4 or d4"}, {"verdict": "needs_commitment",
                   "candidate_moves": ["e4", "d4"]})
    result = recording.save()
    assert verdicts(result, "grader_legality") == {"NOT OBSERVED"}
