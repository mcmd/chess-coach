"""Evidence-scoped, offline checks of frozen live-coach recordings.

Legality is not a claim of best play, likelihood, or forced continuation. Raw model
text is never treated as the student-visible reply. No app/engine imports are used.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import chess
import chess.pgn
from rich.text import Text


CATEGORIES = (
    "hypothetical_legality", "grader_legality", "pv_legality",
    "visible_line_legality", "diagram_provenance", "diagram_source_consistency",
    "pgn_consistency", "duplicate_diagnoses", "detour_lesson_state", "return_position",
)
TOKEN = re.compile(r"\[\[BOARD_[^\]\s]+\]\]")
SAN = r"(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?[!?]*"
NUMBERED_LINE = re.compile(rf"(?<!\w)(\d+)\.(\.\.)?\s*({SAN}(?:\s+(?:\d+\.(?:\.\.)?\s*)?{SAN})+)")


def _board(value):
    if isinstance(value, dict):
        value = value.get("fen")
    if not isinstance(value, str):
        return None
    try:
        board = chess.Board(value)
        return board if board.is_valid() else None
    except (ValueError, TypeError):
        return None


def _state(event):
    state = event.get("data", {}).get("state")
    return state if isinstance(state, dict) else {}


def _agenda(state):
    agenda = state.get("review_agenda", (state.get("review") or {}).get("agenda", []))
    return agenda if isinstance(agenda, list) else []


def _moves(value):
    if isinstance(value, str):
        value = re.sub(r"\d+\.(?:\.\.)?", " ", value).split()
    if not isinstance(value, list):
        return None
    return value


def _line(board, moves):
    """Return an error and final board; never push an unchecked/null PV move."""
    moves = _moves(moves)
    if board is None or not moves:
        return None, None
    board = board.copy(stack=False)
    for index, token in enumerate(moves, 1):
        if isinstance(token, dict):
            token = token.get("uci", token.get("san"))
        try:
            if not isinstance(token, str):
                raise ValueError("move is not recorded as SAN/UCI")
            move = (chess.Move.from_uci(token) if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", token)
                    else board.parse_san(token.rstrip("!?")))
            if move not in board.legal_moves:
                raise ValueError("not a legal move")
            board.push(move)
        except ValueError as exc:
            return f"Move {index} ({token!r}) is illegal at {board.fen()}: {exc}", board
    return "", board


def _plain(value):
    if not isinstance(value, str):
        return ""
    # Rich markup is evidence from the backend, not a reconstructed board.
    return Text.from_markup(value).plain if "[on " in value or "[/]" in value else value


def _surface(value):
    return "\n".join(line.rstrip() for line in _plain(value).replace("\ufe0e", "").splitlines()).strip()


def _diagram_matches_fen(data):
    board = _board(data.get("fen"))
    text = _plain(data.get("text")).replace("\ufe0e", "")
    if board is None:
        return False
    ranks = {}
    black = str(data.get("pov", "")).lower() == "black"
    glyphs = dict(zip("KQRBNPkqrbnp", "♔♕♖♗♘♙♚♛♜♝♞♟"))
    for line in text.splitlines():
        match = re.fullmatch(r"([1-8]) (.*)", line)
        if not match:
            continue
        row = match[2]
        # Recorder preserves the full row, including trailing empty squares.
        if len(row) % 8 or len(row) < 16:
            return False
        width = len(row) // 8
        files = range(7, -1, -1) if black else range(8)
        for column, file in enumerate(files):
            cell = row[column * width:(column + 1) * width].strip()
            piece = board.piece_at(chess.square(file, int(match[1]) - 1))
            if cell != (glyphs[piece.symbol()] if piece else ""):
                return False
        ranks[int(match[1])] = True
    return len(ranks) == 8


def _line_positions(start_fen, san_line) -> list:
    """Every position along a recorded SAN line, stopping at the first illegal move."""
    board = _board(start_fen)
    moves = _moves(san_line)
    if board is None or not moves:
        return []
    positions = [board.copy(stack=False)]
    for san in moves:
        try:
            board.push(board.parse_san(str(san).rstrip("!?")))
        except ValueError:
            break
        positions.append(board.copy(stack=False))
    return positions


_STEP_HEADING = re.compile(r"^### (S\d+)\s+—.*$", re.MULTILINE)
_CHECKS_LINE = re.compile(r"^\*\*Checks:\*\*\s*(.+)$", re.MULTILINE)
_SEND_BLOCK = re.compile(r"\*\*(?:Send|Allowed branch S\d+[a-z]):\*\*.*?```text\n(.*?)\n```", re.DOTALL)


def _script_steps(root: Path) -> dict:
    """Per-step declared checks and send texts from the run's player-script snapshot.

    `**Checks:** detour` — lesson state must not change during that turn.
    `**Checks:** return 24` — the reply must show the agenda moment at move 24, and the
    physical board must equal its value at the start of the most recent detour step.
    """
    script = next(iter(sorted(root.glob("*_script.md"))), None)
    if script is None:
        return {}
    text = script.read_text(encoding="utf-8", errors="replace")
    headings = list(_STEP_HEADING.finditer(text))
    steps = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.end():end]
        checks = _CHECKS_LINE.search(section)
        tokens = checks.group(1).replace(",", " ").split() if checks else []
        return_move = next((int(tokens[i + 1]) for i, t in enumerate(tokens[:-1])
                            if t == "return" and tokens[i + 1].isdigit()), None)
        steps[heading.group(1)] = {"detour": "detour" in tokens, "return": return_move,
                                   "sends": [m.group(1) for m in _SEND_BLOCK.finditer(section)]}
    return steps


def check_recording(run_dir: str | Path) -> dict:
    """Check independent recorded facts, preserving useful findings on partial runs."""
    root = Path(run_dir)
    checks, gaps, events = [], [], []

    def add(check, verdict, evidence, explanation):
        checks.append({"check": check, "verdict": verdict,
                       "evidence": list(dict.fromkeys(evidence)), "explanation": explanation})

    path = root / "events.jsonl"
    if not path.exists():
        gaps.append("events.jsonl is missing")
    else:
        previous = 0
        for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            ref = f"events.jsonl:{line_number}"
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                gaps.append(f"{ref}: invalid/interrupted JSONL record; earlier valid events retained")
                continue
            valid = (isinstance(event, dict) and event.get("schema_version") == 1
                     and type(event.get("id")) is int and event["id"] > previous
                     and isinstance(event.get("timestamp"), str)
                     and isinstance(event.get("kind"), str)
                     and type(event.get("turn")) is int and isinstance(event.get("data"), dict))
            if not valid:
                gaps.append(f"{ref}: invalid event schema or nonmonotonic ID")
                continue
            previous = event["id"]
            events.append(event)
    add("recording_schema", "FAIL" if gaps else ("PASS" if events else "NOT OBSERVED"),
        ["events.jsonl"], "Valid complete JSONL records are retained independently of malformed records." if events
        else "No usable event records are available.")

    execution = {}
    if (root / "execution.json").exists():
        try:
            execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
            if not isinstance(execution, dict):
                raise ValueError("execution must be an object")
        except (ValueError, UnicodeError) as exc:
            gaps.append(f"execution.json: {exc}")
            execution = {}
    else:
        gaps.append("execution.json is missing; parent execution status is unknown")
    status = execution.get("status", execution.get("execution_status", "UNKNOWN"))
    if not isinstance(status, str):
        status = "UNKNOWN"
    blocked = status in ("INFRASTRUCTURE_BLOCKED", "TIMED_OUT")
    add("execution", "BLOCKED" if blocked else "NOT OBSERVED", ["execution.json"],
        f"Execution status is {status}; execution completion is not a coaching-quality verdict.")
    errors = [e for e in events if e["kind"] == "turn_error" or
              (e["kind"] == "session_end" and
               (e["data"].get("exit_code") not in (None, 0) or
                e["data"].get("status") in ("error", "interrupted", "failed")))]
    if errors:
        blocked = True
        add("session_lifecycle", "BLOCKED", [e["id"] for e in errors],
            "Recorded session/turn errors make unobserved downstream checks unavailable; earlier findings remain valid.")

    starts = {e["turn"]: e for e in events if e["kind"] == "turn_start"}
    ends = {e["turn"]: e for e in events if e["kind"] in ("turn_end", "turn_error")}
    students = {e["turn"]: e for e in events if e["kind"] == "student"}
    replies = [e for e in events if e["kind"] == "coach_reply"]
    call_events = [e for e in events if e["kind"] in ("tool_call", "engine_call")]
    results = {(e["kind"], e["data"].get("call_id")): e for e in events
               if e["kind"] in ("tool_result", "engine_result") and e["data"].get("call_id") is not None}
    if not students:
        gaps.append("No student turns recorded")
    for sent in execution.get("sent_steps", []) or []:
        if isinstance(sent, dict):
            turn = sent.get("turn")
            if type(turn) is not int or turn not in students:
                gaps.append(f"execution.json sent step {sent.get('step')} has no matching recorded student turn")
            elif sent.get("message") is not None and sent["message"] != students[turn]["data"].get("text"):
                gaps.append(f"execution.json step {sent.get('step')} message differs from actual student event")
    for turn, student in students.items():
        if turn not in starts:
            gaps.append(f"Turn {turn} (event {student['id']}) has no turn_start/state")
        if turn not in ends:
            gaps.append(f"Turn {turn} (event {student['id']}) is interrupted: no turn_end/turn_error")
        elif ends[turn]["kind"] == "turn_end":
            for kind in ("model_request", "model_response", "coach_reply"):
                if not any(e["turn"] == turn and e["kind"] == kind for e in events):
                    gaps.append(f"Completed turn {turn} has no {kind} evidence")
    for e in events:
        required = {"student": ("text",), "turn_start": ("prompt", "state"),
                    "model_request": ("payload",), "model_response": ("response",),
                    "coach_reply": ("text", "state"), "tool_call": ("call_id", "name", "arguments", "state"),
                    "tool_result": ("call_id", "name", "result", "state"),
                    "board_diagram": ("token", "fen", "pov", "text")}.get(e["kind"], ())
        for key in required:
            if key not in e["data"] or e["data"][key] is None:
                gaps.append(f"Event {e['id']} ({e['kind']}) has missing/unknown {key}")
    for call in call_events:
        result_kind = call["kind"].replace("call", "result")
        result = results.get((result_kind, call["data"].get("call_id")))
        if result is None:
            gaps.append(f"Event {call['id']} has no matching {result_kind}")
        elif result["id"] <= call["id"] or result["data"].get("name") != call["data"].get("name"):
            gaps.append(f"Event {call['id']} has an out-of-order/mismatched result")
    for kind in ("session_start", "session_end", "ready"):
        if not any(e["kind"] == kind for e in events):
            gaps.append(f"No {kind} event recorded")
    for event in events:
        state = _state(event)
        if state:
            for key in ("board_fen", "loaded_moves_uci", "move_index", "current_game_id",
                        "lesson_metadata", "diagnostics", "socratic_attempts"):
                if key not in state:
                    gaps.append(f"Event {event['id']} snapshot lacks {key}")
        for key in ("payload", "response", "result", "arguments"):
            value = event["data"].get(key)
            if isinstance(value, dict) and (value.get("truncated") or value.get("unavailable") or value.get("unknown")):
                gaps.append(f"Event {event['id']} {key} explicitly truncated/unavailable/unknown")

    def legality(category, board, moves, evidence, label):
        error, final = _line(board, moves)
        if error is None:
            add(category, "NOT OBSERVED", evidence, f"{label}: actual starting FEN or move payload unavailable.")
            gaps.append(f"{label} at evidence {evidence}: starting position/move payload unavailable")
        else:
            add(category, "FAIL" if error else "PASS", evidence,
                f"{label}: {error}" if error else f"{label}: all recorded moves legal in sequence. This does not establish best, likely, or forced play.")
        return final if error == "" else None

    for call in call_events:
        data = call["data"]
        args = data.get("arguments")
        args = args if isinstance(args, dict) else {}
        board = _board(args.get("fen", args.get("board")))
        name = data.get("name", "unknown")
        result_event = results.get((call["kind"].replace("call", "result"), data.get("call_id")))
        result = result_event["data"].get("result") if result_event else None
        evidence = [call["id"]] + ([result_event["id"]] if result_event else [])
        failed = (result_event and (result_event["data"].get("error") or
                  (isinstance(result, dict) and result.get("error"))))
        if failed:
            add("tool_availability", "BLOCKED", evidence,
                f"{name} recorded an error. An engine/network/tool failure alone is not an illegal-move finding.")
        if name == "analyze_position" and args.get("moves_to_apply"):
            requested = args["moves_to_apply"]
            applied = requested[:10] if isinstance(requested, list) else requested
            board = legality("hypothetical_legality", board, applied, evidence, "Backend hypothetical SAN")
            if isinstance(requested, list) and len(requested) > 10:
                add("hypothetical_legality", "NOT OBSERVED", evidence,
                    "analyze_position executes only its first ten SAN moves; excess requested moves were not executed.")
        if name in ("evaluate_student_answer", "evaluate_threat_answer"):
            move = args.get("student_move_san", args.get("threat_move_san"))
            grading_board = board.copy(stack=False) if board else None
            routed = isinstance(result, dict) and result.get("interpreted_as") == "opponent_threat"
            if grading_board and (name == "evaluate_threat_answer" or routed):
                if grading_board.is_check():
                    add("grader_legality", "NOT OBSERVED", evidence, "Threat null-move grading is inapplicable while in check.")
                    grading_board = None
                else:
                    grading_board.push(chess.Move.null())
            error, _ = _line(grading_board, [move] if move else None)
            verdict = result.get("verdict") if isinstance(result, dict) else None
            if verdict == "needs_commitment":
                add("grader_legality", "NOT OBSERVED", evidence,
                    "Grader requested a single candidate rather than grading multiple prose moves.")
            elif error is None or failed or result_event is None:
                add("grader_legality", "NOT OBSERVED", evidence, "Grader result or applicable FEN/move unavailable.")
            else:
                rejected = verdict in ("illegal", "not_a_legal_opponent_move")
                add("grader_legality", "PASS" if bool(error) == rejected else "FAIL", evidence,
                    (f"Student input is illegal and was {'correctly rejected' if rejected else 'not rejected'}: {error}" if error
                     else f"Student input is legal for the recorded {'opponent after null move' if name == 'evaluate_threat_answer' or routed else 'side to move'}; grader verdict: {verdict}."))
        if result is None or failed:
            continue
        if name == "quick_eval" and isinstance(result, list) and len(result) > 1 and result[1]:
            legality("pv_legality", board, [result[1]], evidence, "quick_eval best move")
        if isinstance(result, dict):
            result = [result]
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict):
                continue
            for key in ("pv", "best_line", "expected_engine_line"):
                if item.get(key):
                    legality("pv_legality", board, item[key], evidence, f"{name}.{key}")
            for entry in item.get("top_stockfish_lines", []) or []:
                if isinstance(entry, dict) and entry.get("expected_engine_line"):
                    legality("pv_legality", board, entry["expected_engine_line"], evidence, "analyze_position engine line")
            threat_items = item.get("threats_if_side_to_move_passes", []) or []
            threat_lines = [item.get("threat_line"), item.get("engine_main_threat_line")]
            threat_lines += [entry.get("threat_line") for entry in threat_items if isinstance(entry, dict)]
            for line in filter(None, threat_lines):
                probe = board.copy(stack=False) if board and not board.is_check() else None
                if probe:
                    probe.push(chess.Move.null())
                legality("pv_legality", probe, line, evidence, f"{name} opponent threat after null move")
            human = item.get("human_tendency_at_student_elo") or item
            if isinstance(human, dict):
                for entry in human.get("top_human_moves", human.get("top_moves", [])) or []:
                    if isinstance(entry, dict) and entry.get("move"):
                        legality("pv_legality", board, [entry["move"]], evidence, "Recorded human-model candidate (legality only)")
                for move in (human.get("raw_probs") or {}):
                    legality("pv_legality", board, [move], evidence, "Recorded human-model distribution move (legality only)")

    # Only actual HTTP PGNs establish source positions; no online lookup or guessed FEN.
    games = []
    for event in events:
        if event["kind"] != "http_response":
            continue
        body = event["data"].get("body")
        if not isinstance(body, str) or not re.search(r'^\[(?:Event|Site|FEN) "', body, re.M):
            continue
        stream = io.StringIO(body)
        while True:
            try:
                game = chess.pgn.read_game(stream)
            except (ValueError, IndexError) as exc:
                add("pgn_consistency", "FAIL", [event["id"]], f"Recorded PGN could not be parsed: {exc}")
                break
            if game is None:
                break
            if game.errors:
                add("pgn_consistency", "FAIL", [event["id"]], f"Recorded PGN contains parsing errors: {game.errors}")
                continue
            positions = [game.board()]
            for move in game.mainline_moves():
                after = positions[-1].copy(stack=False)
                after.push(move)
                positions.append(after)
            games.append((event, game, positions))

    def source_for(state):
        game_id = state.get("current_game_id")
        matches = [(e, g, p) for e, g, p in games if game_id and
                   (str(game_id) in e["data"].get("url", "") or str(game_id) in g.headers.get("Site", ""))]
        return matches[0] if len(matches) == 1 else None

    for event in events:
        state = _state(event)
        for record in _agenda(state):
            if not isinstance(record, dict):
                continue
            if record.get("best_line"):
                legality("pv_legality", _board(record.get("fen_before")), record["best_line"],
                         [event["id"]], "Review agenda best line")
            if record.get("punishment_line"):
                legality("pv_legality", _board(record.get("fen_after")), record["punishment_line"],
                         [event["id"]], "Review agenda punishment after played move")
        source = source_for(state)
        if source is None:
            continue
        source_event, game, positions = source
        loaded = state.get("loaded_moves_uci")
        if isinstance(loaded, list) and loaded:
            expected = [m.uci() for m in game.mainline_moves()]
            add("pgn_consistency", "PASS" if loaded == expected else "FAIL", [event["id"], source_event["id"]],
                "Loaded move list compared with the actual network PGN mainline.")
        headers = state.get("review_headers", (state.get("review") or {}).get("headers"))
        if isinstance(headers, dict) and headers:
            differences = [key for key, value in headers.items() if key in game.headers and str(value) != game.headers[key]]
            add("pgn_consistency", "FAIL" if differences else "PASS", [event["id"], source_event["id"]],
                f"Recorded review headers compared with source PGN; differing fields: {differences}.")
        board = _board(state.get("board_fen"))
        index = state.get("move_index")
        if board and type(index) is int and isinstance(loaded, list) and loaded == [m.uci() for m in game.mainline_moves()]:
            same = 0 <= index < len(positions) and board.fen() == positions[index].fen()
            add("pgn_consistency", "PASS" if same else "FAIL", [event["id"], source_event["id"]],
                f"Physical board checked against source PGN at move_index {index}.")
        for record in _agenda(state):
            if not isinstance(record, dict):
                continue
            ply = record.get("ply")
            if type(ply) is int and 0 <= ply < len(positions) - 1:
                mismatches = [key for key, expected in
                              (("fen_before", positions[ply].fen()), ("fen_after", positions[ply + 1].fen()),
                               ("san", positions[ply].san(list(game.mainline_moves())[ply])))
                              if record.get(key) is not None and record[key] != expected]
                add("pgn_consistency", "FAIL" if mismatches else "PASS",
                    [event["id"], source_event["id"]],
                    f"Review agenda ply {ply} actual positions/SAN compared with source PGN; differing fields: {mismatches}.")

    diagrams = {e["data"].get("token"): e for e in events if e["kind"] == "board_diagram" and e["data"].get("token")}

    def visible_diagrams(reply):
        tokens = set()
        for event in events:
            if event["kind"] == "model_response" and event["turn"] == reply["turn"] and event["id"] < reply["id"]:
                tokens.update(TOKEN.findall(json.dumps(event["data"].get("response"), ensure_ascii=False)))
        matches = [d for d in diagrams.values() if d["id"] < reply["id"] and
                   _surface(d["data"].get("text")) and
                   _surface(d["data"].get("text")) in _surface(reply["data"].get("text", ""))]
        cited = [d for d in matches if d["data"].get("token") in tokens]
        return cited or matches
    for reply in replies:
        text = reply["data"].get("text")
        if not isinstance(text, str):
            continue
        leaked = TOKEN.findall(text)
        if leaked:
            add("diagram_provenance", "FAIL", [reply["id"]], f"Unrendered board tokens on consumer-visible surface: {leaked}.")
        visible = visible_diagrams(reply)
        for diagram in visible:
            add("diagram_provenance", "PASS" if _diagram_matches_fen(diagram["data"]) else "FAIL",
                [diagram["id"], reply["id"]], "Visible diagram compared with recorded backend token text and square-by-square actual FEN/perspective.")
        if re.search(r"[♔-♟]", text) and not visible:
            add("diagram_provenance", "FAIL", [reply["id"]], "Visible chess diagram has no matching recorded backend rendering.")
        # Check explicit numbered continuations from every position with that move
        # number and side: the actual game, the agenda's verified best/punishment lines,
        # the state board, and shown diagrams. Narrating a verified hypothetical line
        # (e.g. the punishment after the played mistake) is legal even though the
        # actual game went differently.
        state = _state(reply)
        source = source_for(state)
        line_positions = []
        for record in _agenda(state):
            if isinstance(record, dict):
                line_positions += _line_positions(record.get("fen_before"), record.get("best_line"))
                line_positions += _line_positions(record.get("fen_after"), record.get("punishment_line"))
        for match in NUMBERED_LINE.finditer(text):
            number, black = int(match[1]), bool(match[2])
            pool = list(source[2]) if source else []
            pool += [_board(v) for v in [state.get("board_fen")] + [d["data"].get("fen") for d in visible]
                     + [r.get("fen_before") for r in _agenda(state) if isinstance(r, dict)]]
            pool += line_positions
            unique = {b.fen(): b for b in pool
                      if b is not None and b.fullmove_number == number and b.turn == (not black)}
            evidence = [reply["id"]] + ([source[0]["id"]] if source else [])
            label = f"Consumer-visible line {match[0]!r}"
            legal = next((b for b in unique.values() if _line(b, match[3])[0] == ""), None)
            if legal is not None or not unique:
                legality("visible_line_legality", legal, match[3], evidence, label)
            else:
                legality("visible_line_legality", next(iter(unique.values())), match[3], evidence,
                         f"{label} (illegal from all {len(unique)} known position(s) with this move number)")
        for match in re.finditer(rf"(?im)^(?:Line|Continuation|Follow-up|Verified line):\s*({SAN}(?:\s+{SAN})+)\s*$", text):
            contexts = {_board(d["data"].get("fen")).fen(): _board(d["data"].get("fen"))
                        for d in visible if _board(d["data"].get("fen"))}
            board = next(iter(contexts.values())) if len(contexts) == 1 else None
            if not visible:
                board = _board(state.get("board_fen"))
            legality("visible_line_legality", board, match[1], [reply["id"]] + [d["id"] for d in visible],
                     "Explicit consumer-visible continuation with recorded board context")

    # Snapshots repeat diagnostics: only duplicate entries within a snapshot or
    # new detour diagnoses are failures, not repeated recording of one entry.
    for event in events:
        diagnostics = _state(event).get("diagnostics")
        if isinstance(diagnostics, list):
            fens = [d.get("fen") for d in diagnostics if isinstance(d, dict) and d.get("fen")]
            duplicates = sorted({fen for fen in fens if fens.count(fen) > 1})
            add("duplicate_diagnoses", "FAIL" if duplicates else "PASS", [event["id"]],
                f"Diagnosis entries in this actual snapshot checked by FEN; duplicate positions: {duplicates}.")

    script_steps = _script_steps(root)
    steps = execution.get("sent_steps", [])
    step_turns = {str(s.get("step")): s.get("turn") for s in steps if isinstance(s, dict)} if isinstance(steps, list) else {}
    # The actual scripted messages can establish step turns if parent status is
    # absent; do not infer steps from coach prose or numbering of turns.
    for turn, event in students.items():
        text = event["data"].get("text", "")
        for step, info in script_steps.items():
            if text in info["sends"]:
                step_turns.setdefault(step, turn)
    ordered = sorted(((turn, step) for step, turn in step_turns.items() if turn is not None and step in script_steps))
    snapshot_keys = ("lesson_metadata", "diagnostics", "socratic_attempts", "current_game_id")
    types = {"lesson_metadata": dict, "diagnostics": list, "socratic_attempts": dict}
    last_detour_start = None
    for turn, step in ordered:
        info = script_steps[step]
        start_event = starts.get(turn)
        return_baseline = last_detour_start or start_event
        if info["detour"]:
            last_detour_start = start_event
            baseline = _state(start_event) if start_event else {}
            for event in [e for e in events if start_event and e["turn"] == turn
                          and e["id"] >= start_event["id"] and _state(e)]:
                state = _state(event)
                missing = [key for key in snapshot_keys if key not in baseline or key not in state or
                           (key in types and (not isinstance(baseline[key], types[key]) or
                                              not isinstance(state[key], types[key])))]
                if missing:
                    add("detour_lesson_state", "NOT OBSERVED", [start_event["id"], event["id"]],
                        f"{step}: snapshot fields unavailable: {missing}.")
                    continue
                changes = [key for key in snapshot_keys if baseline[key] != state[key]]
                add("detour_lesson_state", "FAIL" if changes else "PASS", [start_event["id"], event["id"]],
                    f"{step} question/detour must not advance agenda, regrade, or add diagnoses; changed fields: {changes}.")
        if info["return"] is None:
            continue
        step_replies = [e for e in replies if e["turn"] == turn]
        if not step_replies:
            continue
        reply = step_replies[-1]
        baseline_event = return_baseline
        baseline = _state(baseline_event) if baseline_event else {}
        agenda_fen = next((r.get("fen_before") for r in _agenda(_state(reply)) + _agenda(baseline)
                           if isinstance(r, dict) and r.get("number") == info["return"]), None)
        source = source_for(_state(reply)) or source_for(baseline)
        target = _board(agenda_fen) or (next((b for b in source[2] if b.fullmove_number == info["return"]
                                              and b.turn == chess.WHITE), None) if source else None)
        actual = _board(_state(reply).get("board_fen"))
        pending = _board(baseline.get("board_fen"))
        evidence = [reply["id"]] + ([source[0]["id"]] if source else []) + ([baseline_event["id"]] if baseline_event else [])
        if actual and pending:
            add("return_position", "PASS" if actual.fen() == pending.fen() else "FAIL", evidence,
                f"{step}: physical timeline board preserved from the pre-detour snapshot; the displayed "
                "lesson position is checked separately.")
        else:
            add("return_position", "NOT OBSERVED", evidence, "Actual return board or pre-detour board unavailable.")
        visible = visible_diagrams(reply)
        if target:
            for diagram in visible:
                board = _board(diagram["data"].get("fen"))
                add("diagram_source_consistency", "PASS" if board and board.fen() == target.fen() else "FAIL",
                    [diagram["id"], reply["id"]] + ([source[0]["id"]] if source else []),
                    f"{step}: shown diagram compared with the agenda moment at move {info['return']}.")
        if not visible:
            gaps.append(f"{step} reply event {reply['id']} has no evidenced visible board diagram")

    for category in CATEGORIES:
        if not any(c["check"] == category for c in checks):
            add(category, "BLOCKED" if blocked else "NOT OBSERVED", ["execution.json"] if blocked else [],
                "Downstream evidence unavailable after infrastructure failure/timeout." if blocked else "No applicable actual payload/context was recorded; no quality conclusion is possible.")
    add("recording_completeness", "FAIL" if gaps else "PASS", ["events.jsonl", "execution.json"],
        "Recording gaps are listed separately; this check does not erase independent chess findings." if gaps else
        "Required lifecycle, turn, model/tool, and rendered-surface evidence present for observed turns; semantic quality remains separately scoped.")
    return {"checks": checks, "recording_gaps": gaps, "execution_status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = check_recording(args.run_dir)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    (args.run_dir / "checks.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
