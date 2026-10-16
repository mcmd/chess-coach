"""The Gemini tool-execution loop and response rendering.

Extracted from coach.py. Internal <think> scratchpads are stripped before
display — the student sees the lesson, not the model's bookkeeping.
"""
import re
import time

import chess
from google.genai import types
from rich.markdown import Markdown

from app.memory import load_dossier
from app.notation import ambiguous_piece_references, san_mention_spans
from app.tools import GRADING_TOOLS, feedback_sentence, grading_next_step
from app.trace import fmt_args, trace
from app.ui import console

# A single user turn may need several analyse→grade→explain tool rounds, but never
# dozens. Bounds a model that gets stuck requesting tools in a loop (plan audit).
MAX_TOOL_ROUNDS = 12

TOOL_MESSAGES = {
    "fetch_lichess_game_by_id": "[dim cyan]↳ Fetching game from Lichess...[/dim cyan]",
    "fetch_chessgames_com_game": "[dim cyan]↳ Fetching Master Game from Chessgames.com...[/dim cyan]",
    "query_endgame_tablebase": "[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]",
    "analyze_position": "[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]",
    "evaluate_student_answer": "[dim magenta]↳ Grading your answer against the engine...[/dim magenta]",
    "evaluate_threat_answer": "[dim magenta]↳ Grading your threat-read against the engine...[/dim magenta]",
    "explain_pawn_structure_evolution": "[dim cyan]↳ Building the pawn-structure story...[/dim cyan]",
    "get_opening_repertoire": "[dim yellow]↳ Checking Lichess Opening Databases...[/dim yellow]",
    "play_sparring_reply": "[dim green]↳ The coach is choosing a human-realistic reply...[/dim green]",
    "execute_coach_move": "[dim green]↳ The Logic Tutor is evaluating the move...[/dim green]",
    "submit_diagnosis": "[dim magenta]↳ Recording diagnosis to your dossier...[/dim magenta]",
    "get_game_review_summary": "[dim cyan]↳ Re-fetching the game review digest...[/dim cyan]",
}


def _extract_function_calls(response):
    calls = []
    if response.function_calls:
        return list(response.function_calls)
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for p in response.candidates[0].content.parts:
            if getattr(p, 'function_call', None):
                calls.append(p.function_call)
            elif getattr(p, 'functionCall', None):
                calls.append(p.functionCall)
    return calls


# #38: one call thought for 341 s (62,914 thinking tokens) while every other call used under
# 2,000. Each call has a time limit (prompts.create_chat_session); a call that hits it is
# retried once at this thinking level with twice the limit. The same limit gave up while the
# API was merely slow (run zyijwnhs); no limit hung for 8 minutes during an API stall (run
# 92lqnauv). 90 s + 180 s still fits the 300 s turn. A failed call leaves nothing in history.
TIMEOUT_RETRY_THINKING_LEVEL = "medium"
TIMEOUT_RETRY_LIMIT_FACTOR = 2


def _is_timeout(exc) -> bool:
    import httpx
    from google.genai import errors
    if isinstance(exc, httpx.TimeoutException):
        return True
    return isinstance(exc, errors.APIError) and (getattr(exc, "code", None) == 504
                                                 or "DEADLINE_EXCEEDED" in str(exc))


def _send_with_spinner(chat, payload):
    """The spinner only covers MODEL calls. Tools run outside any live display
    so long-running ones (game reviews) can show their own progress bars —
    Rich allows a single live display at a time."""
    with console.status("[bold yellow]Coach is thinking...[/bold yellow]", spinner="dots"):
        try:
            return chat.send_message(payload)
        except Exception as exc:
            base = getattr(chat, "coach_config", None)
            if base is None or not _is_timeout(exc):
                raise
            limit_ms = (base.http_options.timeout if base.http_options and base.http_options.timeout else None)
            retry_limit = limit_ms * TIMEOUT_RETRY_LIMIT_FACTOR if limit_ms else None
            trace(f"model call timed out ({type(exc).__name__}); retrying once at "
                  f"thinking_level={TIMEOUT_RETRY_THINKING_LEVEL} with a {retry_limit}-ms limit")
            retry = base.model_copy(update={
                "thinking_config": types.ThinkingConfig(thinking_level=TIMEOUT_RETRY_THINKING_LEVEL),
                "http_options": (base.http_options.model_copy(update={"timeout": retry_limit})
                                 if base.http_options else None)})
            return chat.send_message(payload, config=retry)


# Gemini sometimes returns its reasoning as an ordinary text part ("s Thinking Process:
# 1. **Analyze ...**"). Kept in the chat history it is imitated by every later reply
# (live eval 2026-10-12: 22/40 replies; replays leaked 14/14 with it, 0/4 without),
# and the drafts reveal withheld answers. Such parts never reach the student or history.
_LEAKED_REASONING_RE = re.compile(r"\A\W*(?:\w+'?s?\s+)?\**\s*thinking process\b", re.IGNORECASE)
_LEAK_NOTICE = "_(My reply came out garbled, so I discarded it. Please send your last message again.)_"


def _leaked_parts(response) -> list:
    content = response.candidates[0].content if getattr(response, "candidates", None) else None
    return [p for p in (getattr(content, "parts", None) or [])
            if not getattr(p, "thought", None) and _LEAKED_REASONING_RE.match(getattr(p, "text", None) or "")]


def _drop_last_exchange(chat) -> None:
    """Remove the last input/output pair from both chat histories (public get_history
    returns the live lists); curated history only holds it when the reply was valid."""
    history = chat.get_history()
    last = history[-1]
    del history[-2:]
    curated = chat.get_history(curated=True)
    if curated and curated[-1] is last:
        del curated[-2:]


def _send_without_leaks(chat, payload, state):
    """Send, keeping leaked reasoning out of what the student sees and the history.
    Beside a tool call the leaked text is dropped; a leaked reply is regenerated once,
    then replaced by a retry notice that keeps the history's turn order valid."""
    response = _send_with_spinner(chat, payload)
    for attempt in range(2):
        leaked = _leaked_parts(response)
        if not leaked:
            return response
        _event_append(state, "deliberation_events", {"type": "leaked_reasoning", "attempt": attempt + 1,
                                                      "game_id": getattr(state, "current_game_id", None)})
        trace(f"leaked reasoning in model reply (attempt {attempt + 1})")
        content = response.candidates[0].content
        if _extract_function_calls(response):
            content.parts = [p for p in content.parts if not any(p is q for q in leaked)]
            return response
        if attempt == 0:
            _drop_last_exchange(chat)
            response = _send_with_spinner(chat, payload)
    content.parts = [types.Part(text=_LEAK_NOTICE)]
    return response


def run_agent_turn(chat, coach_tools, message) -> None:
    """Send `message`, execute any tool calls until the model stops asking,
    then render the final response."""
    coach_tools.state.latest_user_text = message
    state = coach_tools.state
    if hasattr(coach_tools.state, "claim_gate_extra_fens"):
        coach_tools.state.claim_gate_extra_fens.clear()
    # Results the backend router graded before this turn count as this turn's gradings.
    turn_grading_results = [dict(entry) for entry in (getattr(state, "pregraded", None) or {}).values()]
    turn_tool_names = []
    response = _send_without_leaks(chat, message, state)
    full_text = ""
    rounds = 0

    while True:
        this_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'text', None):
                    this_text += part.text + "\n"

        f_calls = _extract_function_calls(response)
        if not f_calls:
            # Reveal-gating integrity: render ONLY the final, tool-free response.
            # Text emitted ALONGSIDE a grading tool call (before reveal_allowed) is
            # discarded — otherwise the model could blurt the withheld answer in the
            # pre-tool text and have it shown regardless of the backend's verdict.
            full_text = this_text
            break

        rounds += 1
        if rounds > MAX_TOOL_ROUNDS:
            # A confused model can otherwise request tools forever; per-call engine
            # caps don't bound the NUMBER of round trips. Stop the turn cleanly.
            trace(f"tool loop hit the {MAX_TOOL_ROUNDS}-round budget; aborting turn")
            full_text = ("_(I got stuck cycling through analysis tools and stopped to avoid "
                         "looping — please re-ask, perhaps more specifically.)_")
            break

        tool_parts = []
        for fn_call in f_calls:
            args_dict = dict(fn_call.args) if fn_call.args else {}
            turn_tool_names.append(fn_call.name)
            console.print(TOOL_MESSAGES.get(fn_call.name, f"[dim cyan]↳ Executing {fn_call.name}...[/dim cyan]"))
            trace(f"tool call: {fn_call.name}({fmt_args(args_dict)})")

            started = time.monotonic()
            if hasattr(coach_tools, fn_call.name):
                method = getattr(coach_tools, fn_call.name)
                try:
                    result = method(**args_dict)
                except TypeError as e:
                    result = {"error": f"Invalid arguments passed to tool '{fn_call.name}': {e}. "
                                       "Please check the tool's parameters and try again."}
                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}
            else:
                result = {"error": f"Tool {fn_call.name} not found."}
            _record_tool_candidate_fens(coach_tools.state, fn_call.name, args_dict, result)
            if fn_call.name in GRADING_TOOLS:
                step = grading_next_step(fn_call.name, result)
                if step:
                    result = {**result, "next_step": step + (" " + result["next_step"] if result.get("next_step") else "")}
                turn_grading_results.append({"tool": fn_call.name, "args": args_dict, "result": result})
            elif fn_call.name == "submit_diagnosis" and isinstance(result, dict) and result.get("success"):
                # Run kjofd3ug S4/S11: after grading and logging, the final reply presented the next
                # moment without the feedback (text written beside tool calls is not shown).
                feedback = [s for item in turn_grading_results
                            if (s := feedback_sentence(item["tool"], item["result"]))]
                if feedback:
                    result = {**result, "next_step": ("Before presenting the next moment, tell the student "
                                                      "this turn's feedback first: " + " ".join(feedback))}
            trace(f"tool result: {fn_call.name} in {time.monotonic() - started:.1f}s -> {_summarize_result(result)}")

            tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))

        response = _send_without_leaks(chat, tool_parts, state)

    coach_tools.state._turn_grading_results = turn_grading_results
    coach_tools.state._turn_tool_names = turn_tool_names
    _render(full_text, coach_tools.state, getattr(coach_tools, "config", None))
    if hasattr(coach_tools.state, "_turn_grading_results"):
        coach_tools.state._turn_grading_results = []
    if hasattr(coach_tools.state, "_turn_tool_names"):
        coach_tools.state._turn_tool_names = []
    if hasattr(coach_tools.state, "claim_gate_extra_fens"):
        coach_tools.state.claim_gate_extra_fens.clear()


def _record_tool_candidate_fens(state, tool_name: str, args: dict, result) -> None:
    """Register positions licensed by live tool results for this turn's L1 gate.

    The static gate knows the live board and review agenda lines. A same-turn tool
    may return a fresh PV or threat line from another position; without these FENs,
    a legal move deep in that returned line can be redacted as "unverified".
    """
    if not isinstance(result, dict) or "error" in result:
        return
    if not hasattr(state, "claim_gate_extra_fens"):
        return
    fens = []
    start = args.get("fen") if isinstance(args, dict) else None
    if not start:
        return

    if tool_name == "analyze_position":
        board = _board_after_moves(start, args.get("moves_to_apply") or [])
        if board is None:
            return
        fens.append(board.fen())
        for line in result.get("top_stockfish_lines") or []:
            fens.extend(_line_position_fens(board.fen(), (line.get("expected_engine_line") or "").split()))
    elif tool_name == "evaluate_student_answer":
        if result.get("best_line"):
            fens.extend(_line_position_fens(start, str(result["best_line"]).split()))
    elif tool_name == "evaluate_threat_answer":
        board = _null_move_board(start)
        if board is None:
            return
        fens.append(board.fen())
        for line in _threat_lines_from_result(result):
            fens.extend(_line_position_fens(board.fen(), line.split()))

    seen = set(state.claim_gate_extra_fens)
    for fen in fens:
        if fen and fen not in seen:
            state.claim_gate_extra_fens.append(fen)
            seen.add(fen)


def _board_after_moves(fen: str, moves: list) -> chess.Board | None:
    try:
        board = chess.Board(fen)
        for san in (moves or [])[:10]:
            board.push_san(str(san))
        return board
    except (ValueError, TypeError):
        return None


def _null_move_board(fen: str) -> chess.Board | None:
    try:
        board = chess.Board(fen)
        board.push(chess.Move.null())
        return board
    except (ValueError, TypeError):
        return None


def _threat_lines_from_result(result: dict) -> list[str]:
    lines = []
    if result.get("threat_line"):
        lines.append(result["threat_line"])
    comparison = result.get("comparison") or {}
    if comparison.get("engine_main_threat_line"):
        lines.append(comparison["engine_main_threat_line"])
    for threat in result.get("actual_threats") or []:
        if threat.get("threat_line"):
            lines.append(threat["threat_line"])
    return lines


def _summarize_result(result) -> str:
    if not isinstance(result, dict):
        return type(result).__name__
    if "error" in result:
        return f"ERROR: {str(result['error'])[:100]}"
    interesting = ("verdict", "reveal_allowed", "coach_played", "best_move",
                   "evaluation_after", "status", "success")
    parts = [f"{k}={result[k]}" for k in interesting if k in result]
    if parts:
        return ", ".join(parts)
    return "keys: " + ", ".join(list(result.keys())[:8])


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


# Matches a board token the backend never minted (model-invented or mangled),
# optionally wrapped in backticks.
_UNRESOLVED_TOKEN_RE = re.compile(r"`?\[\[BOARD_[A-Za-z0-9]+\]\]`?")

# Chess-move styling normalization: the prompt asks for moves in backticks
# (rendered as blue inline code), but the model drifts — `16... e5` in one
# sentence, **17. d5?!** in the next. Normalize deterministically at render time.
_MOVE_CORE = r"(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)[!?]{0,2}"
_NUMBERED_MOVE = rf"\d{{1,3}}\.(?:\.\.)?\s?{_MOVE_CORE}"
_BOLD_MOVE_RE = re.compile(rf"\*\*((?:{_NUMBERED_MOVE})|{_MOVE_CORE})\*\*")
_ITALIC_MOVE_RE = re.compile(rf"(?<!\*)\*((?:{_NUMBERED_MOVE})|{_MOVE_CORE})\*(?!\*)")
_BARE_NUMBERED_MOVE_RE = re.compile(rf"(?<![`\w*])({_NUMBERED_MOVE})(?![`\w])")
_CODE_SEGMENT_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")


def normalize_move_styling(text: str) -> str:
    """Render every chess move the same way (inline code) regardless of how the
    model styled it: **17. d5?!**, *Nf3*, and bare '17. d5?!' all become
    `17. d5?!`-style code spans. Existing code spans and fences are untouched."""
    parts = _CODE_SEGMENT_RE.split(text)
    for i in range(0, len(parts), 2):  # even indexes are outside code spans/fences
        segment = _BOLD_MOVE_RE.sub(r"`\1`", parts[i])
        segment = _ITALIC_MOVE_RE.sub(r"`\1`", segment)
        segment = _BARE_NUMBERED_MOVE_RE.sub(r"`\1`", segment)
        parts[i] = segment
    return "".join(parts)


def substitute_board_tokens(text: str, visuals: dict) -> tuple[str, int]:
    """Replace known board tokens with their diagrams; scrub any leftover
    (invented) tokens instead of showing them raw. Returns (text, n_scrubbed).

    Retained for callers that need a flat-text expansion (and its regression
    tests); the live CLI now renders via _split_board_segments so coloured board
    markup is printed through the Rich console rather than Markdown."""
    for token, visual in visuals.items():
        text = text.replace(token, f"\n{visual}\n")
    return _UNRESOLVED_TOKEN_RE.subn("\n*(diagram unavailable — the coach referenced a board it never generated)*\n", text)


_DIAGRAM_UNAVAILABLE = "*(diagram unavailable — the coach referenced a board it never generated)*"


def _split_board_segments(text: str, visuals: dict) -> tuple[list[tuple[str, str]], int]:
    """Split prose around board tokens so each can be rendered with the right
    engine: ('text', markdown) prose goes through Rich Markdown; ('board', markup)
    coloured diagrams are printed as Rich console markup. Unknown (model-invented)
    tokens are scrubbed to a plain notice. Returns (segments, n_scrubbed)."""
    segments: list[tuple[str, str]] = []
    scrubbed = 0
    last = 0
    for m in _UNRESOLVED_TOKEN_RE.finditer(text):
        pre = text[last:m.start()]
        if pre:
            segments.append(("text", pre))
        token = m.group(0).strip("`")
        if token in visuals:
            segments.append(("board", visuals[token]))
        else:
            segments.append(("text", f"\n{_DIAGRAM_UNAVAILABLE}\n"))
            scrubbed += 1
        last = m.end()
    tail = text[last:]
    if tail:
        segments.append(("text", tail))
    return segments, scrubbed


# --- Render-time claim gate (plan P0-3) -----------------------------------
# A bounded verifier: it does NOT try to validate every adjective. It checks the
# one thing that is cheap and unambiguous — that a chess move named in the prose is
# legal in some position actually in play — and is staged by enforcement level so
# it ships in shadow (log-only) and is promoted only once the logs are clean.

_BOARD_TOKEN_RE = re.compile(r"`?\[\[BOARD_[A-Za-z0-9]+\]\]`?")


def _parses(board: chess.Board, san: str) -> bool:
    try:
        board.parse_san(san)
        return True
    except ValueError:
        return False


def candidate_boards(state, extra_fens=()) -> list:
    """The positions the coach may legitimately reference this turn: the live
    board, every position of the loaded game, agenda verified lines, same-turn
    tool positions, and any caller-supplied positions (e.g. the dossier's).
    A move legal in any of these is allowed; one legal in none is 'impossible'
    (likely hallucinated)."""
    fens, boards = set(), []
    board = getattr(state, "board", None)
    if board is not None:
        fens.add(board.fen())
    for f in list(getattr(state, "claim_gate_extra_fens", []) or []) + list(extra_fens):
        if f:
            fens.add(f)
    review = getattr(state, "review", None)
    # Moves actually played in the loaded game are verified truth wherever they occur.
    for r in (getattr(review, "records", None) or []):
        for f in (getattr(r, "fen_before", None), getattr(r, "fen_after", None)):
            if f:
                fens.add(f)
    for r in (getattr(review, "agenda", None) or []):
        for f in (getattr(r, "fen_before", None), getattr(r, "fen_after", None)):
            if f:
                fens.add(f)
        prev_fen = _previous_move_origin_fen(r)
        if prev_fen:
            fens.add(prev_fen)
        for f in _line_position_fens(getattr(r, "fen_before", None), getattr(r, "best_line", None)):
            fens.add(f)
        # The opponent's threats at the moment ("if you don't move the rook, Black plays
        # gxf1=Q+"): the position with the opponent to move, as the threat probe uses.
        threat_fen = _opponent_to_move_fen(getattr(r, "fen_before", None))
        if threat_fen:
            fens.add(threat_fen)
        for f in _line_position_fens(getattr(r, "fen_after", None), getattr(r, "punishment_line", None)):
            fens.add(f)
    for f in fens:
        try:
            boards.append(chess.Board(f))
        except ValueError:
            pass
    return boards


def _opponent_to_move_fen(fen) -> str | None:
    """`fen` with the other side to move (a pass), unless the side to move is in check."""
    try:
        board = chess.Board(fen)
    except (TypeError, ValueError):
        return None
    if board.is_check():
        return None
    board.push(chess.Move.null())
    return board.fen()


def _previous_move_origin_fen(record) -> str | None:
    """Board where record.prev_move_san was legal.

    Orientation text often says "Black has just played 19...Rfe8". The agenda
    FEN is after that move, so include the position before the previous move.
    """
    history = list(getattr(record, "history_uci", None) or [])
    if not history or not getattr(record, "prev_move_san", None):
        return None
    try:
        board = chess.Board()
        for uci in history[:-1]:
            board.push(chess.Move.from_uci(uci))
        return board.fen()
    except (ValueError, TypeError):
        return None


def _line_position_fens(start_fen: str, san_line: list) -> list:
    """All positions where a verified SAN line may be referenced.

    The L1 claim gate checks whether a named SAN move is legal in any known
    position. A PV move like `Qh5` may be illegal in the agenda's root position
    but legal after the first two verified line moves, so include each
    intermediate board before and after pushing the line.
    """
    if not start_fen or not san_line:
        return []
    try:
        board = chess.Board(start_fen)
    except ValueError:
        return []
    fens = [board.fen()]
    for san in san_line[:10]:
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        board.push(move)
        fens.append(board.fen())
    return fens


_BARE_SQUARE_RE = re.compile(r"[a-h][1-8]")
_MOVE_NUMBER_BEFORE_RE = re.compile(r"\d{1,3}\.(?:\.\.)?\s*$")


def _is_move_claim(text: str, start: int, end: int, san: str) -> bool:
    """A bare square ("e1", "c7") is a square reference unless it is written in move
    notation: after a move number ("24. e5", "24... e5") or as a standalone code span
    (`e5`). Pawn captures, piece moves, and castling are claims wherever they appear."""
    if not _BARE_SQUARE_RE.fullmatch(san.rstrip("+#")):
        return True
    if _MOVE_NUMBER_BEFORE_RE.search(text[max(0, start - 8):start]):
        return True
    return start > 0 and text[start - 1] == "`" and text[end:end + 1] == "`"


def _illegal_move_spans(text: str, boards: list) -> list:
    """(start, end, san) for each move claim illegal in EVERY candidate board.
    Masking preserves string length, so spans index the original text."""
    if not boards:
        return []  # no known positions to check against — never flag blind
    clean = _BOARD_TOKEN_RE.sub(lambda m: " " * len(m.group(0)), text)
    clean = _mask_non_claim_move_mentions(clean)
    legal = {}
    spans = []
    for start, end, san in san_mention_spans(clean):
        if not _is_move_claim(clean, start, end, san):
            continue
        if san not in legal:
            legal[san] = any(_parses(b, san) for b in boards)
        if not legal[san]:
            spans.append((start, end, san))
    return spans


def find_illegal_moves(text: str, boards: list) -> list:
    """SAN-looking move claims that are illegal in EVERY candidate board (impossible
    given the positions in play), first-seen order. A move legal in any known position
    is allowed. Board tokens and square references are masked first."""
    seen, result = set(), []
    for _, _, san in _illegal_move_spans(text, boards):
        if san not in seen:
            seen.add(san)
            result.append(san)
    return result


def _mask_non_claim_move_mentions(text: str) -> str:
    """Mask square references and invalid-move discussion before SAN checks."""
    if not text:
        return ""
    square_context = (
        r"\b(?:on|onto|to|from|at|square|squares|pawn|piece|king|queen|rook|bishop|"
        r"knight|file|rank)\s+"
    )
    masked = re.sub(square_context + r"`?[a-h][1-8]`?", lambda m: " " * len(m.group(0)),
                    text, flags=re.IGNORECASE)
    piece_context = (
        r"\b`?[a-h][1-8]`?(?:-|\s+)"
        r"(?:pawn|piece|king|queen|rook|bishop|knight|file|rank|square)\b"
    )
    masked = re.sub(piece_context, lambda m: " " * len(m.group(0)),
                    masked, flags=re.IGNORECASE)
    square_list_context = (
        r"\b(?:[a-h][1-8]\s*(?:,|and|or)?\s*){1,6}"
        r"(?=(?:pawns?|pieces?|squares?|files?|ranks?)\b)"
    )
    masked = re.sub(square_list_context, lambda m: " " * len(m.group(0)),
                    masked, flags=re.IGNORECASE)
    negative_context = (
        r"\b(?:not|isn't|isnt|wasn't|wasnt|doesn't|doesnt|don't|dont|can't|cant|"
        r"cannot|couldn't|couldnt|illegal|invalid|no)\b"
        r"[^.\n]{0,80}?"
    )
    move_core = (
        r"`?(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"
        r"[a-h]x?[a-h][1-8](?:=[QRBN])?[+#]?)`?"
    )
    return re.sub(negative_context + move_core, lambda m: " " * len(m.group(0)),
                  masked, flags=re.IGNORECASE)


def apply_claim_gate(text: str, boards: list, level: int) -> tuple:
    """Run the claim gate at `level`. Returns (text, findings).
    L0: detect only (text unchanged) — shadow/log mode.
    L1+: redact moves illegal in every known position (degrade to less detail,
         never a non-sequitur — the redaction is visible, not a silent hole)."""
    spans = _illegal_move_spans(text, boards)
    findings, seen = [], set()
    for _, _, san in spans:
        if san not in seen:
            seen.add(san)
            findings.append({"type": "illegal_move", "token": san})
    if level >= 1 and spans:
        # Replace only the flagged occurrences (right to left keeps offsets valid);
        # the same string elsewhere may be a legitimate square reference.
        for start, end, _ in reversed(spans):
            if start > 0 and text[start - 1] == "`" and text[end:end + 1] == "`":
                start, end = start - 1, end + 1
            text = text[:start] + "(an unverified move)" + text[end:]
    return text, findings


def _claim_gate_excerpt(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit - 1].rstrip() + "…"


def _record_claim_gate_event(state, *, level: int, boards: list, before: str, after: str, findings: list) -> None:
    if not findings or not hasattr(state, "claim_gate_events"):
        return
    event = {
        "level": level,
        "mode": "shadow" if level < 1 else "redacted",
        "redacted": level >= 1,
        "tokens": [f.get("token") for f in findings if f.get("token")],
        "findings": [dict(f) for f in findings],
        "candidate_board_count": len(boards),
        "game_id": getattr(state, "current_game_id", None),
        "move_index": getattr(state, "move_index", None),
        "before_excerpt": _claim_gate_excerpt(before),
        "after_excerpt": _claim_gate_excerpt(after),
    }
    state.claim_gate_events.append(event)
    state.claim_gate_events[:] = state.claim_gate_events[-50:]


def _run_claim_gate(output_text: str, state, config) -> str:
    level = int(getattr(config, "claim_gate_level", 0)) if config else 0
    boards = candidate_boards(state, _dossier_fens(config))
    new_text, findings = apply_claim_gate(output_text, boards, level)
    if findings:
        _record_claim_gate_event(state, level=level, boards=boards,
                                 before=output_text, after=new_text, findings=findings)
        tokens = ", ".join(f["token"] for f in findings)
        trace(f"claim-gate (L{level}): {len(findings)} illegal-move finding(s): {tokens} "
              + ("[shadow: not modified]" if level < 1 else "[redacted]"))
    return new_text


def _dossier_fens(config) -> list:
    """Diagnosed positions from the student's dossier: after /memorize no game is
    loaded, but recalling those moments must not be redacted as unverifiable."""
    path = getattr(config, "dossier_file", None) if config else None
    if not path:
        return []
    try:
        dossier = load_dossier(path)
    except Exception:
        return []
    return [p.get("fen") for p in dossier.get("recent_positions") or [] if p.get("fen")]


def _record_ambiguous_piece_event(state, *, board_fen: str, finding) -> None:
    if not hasattr(state, "ambiguous_piece_events"):
        return
    event = {
        "type": "ambiguous_piece_reference",
        "phrase": finding.phrase,
        "piece": finding.piece,
        "square": finding.square,
        "color": finding.color,
        "candidate_squares": list(finding.candidate_squares),
        "board_fen": board_fen,
        "game_id": getattr(state, "current_game_id", None),
        "move_index": getattr(state, "move_index", None),
    }
    state.ambiguous_piece_events.append(event)
    state.ambiguous_piece_events[:] = state.ambiguous_piece_events[-50:]


def _run_ambiguous_piece_lint(output_text: str, state) -> None:
    """Shadow-only prose-fidelity lint: exact SAN should replace phrases like
    "rook to d8" when multiple same-color pieces could be meant."""
    boards = candidate_boards(state)
    seen = set()
    for board in boards:
        for finding in ambiguous_piece_references(output_text, board):
            key = (finding.phrase, finding.color, finding.piece, finding.square,
                   tuple(finding.candidate_squares))
            if key in seen:
                continue
            seen.add(key)
            _record_ambiguous_piece_event(state, board_fen=board.fen(), finding=finding)
            trace("ambiguous-piece lint: "
                  f"{finding.phrase} could mean {finding.color} {finding.piece}s on "
                  f"{', '.join(finding.candidate_squares)}")


def _event_append(state, attr: str, event: dict) -> None:
    if not hasattr(state, attr):
        return
    events = getattr(state, attr)
    events.append(event)
    events[:] = events[-50:]


_VERDICT_ACK_TOKENS = {
    "correct_threat": ("correct", "yes", "spot on", "right", "threatening"),
    "real_but_secondary_threat": ("secondary", "not the main", "not black's main", "legal", "real threat"),
    "not_a_real_threat": ("not a real threat", "not the threat", "doesn't threaten", "does not threaten"),
    "no_significant_threat_exists": ("no significant threat", "no immediate threat", "no forcing threat"),
    "not_a_legal_opponent_move": ("not legal", "illegal", "not a legal reply"),
    "best": ("best", "correct", "yes", "right"),
    "excellent_alternative": ("excellent alternative", "good alternative"),
    "reasonable_but_imprecise": ("reasonable", "imprecise"),
    "inaccurate": ("inaccurate", "not quite"),
    "wrong": ("wrong", "doesn't work", "does not work"),
    "illegal": ("illegal", "not legal"),
    "needs_commitment": ("which one", "commit", "choose one"),
}


def _verdict_acknowledged(text: str, verdict: str) -> bool:
    lower = (text or "").lower()
    return any(token in lower for token in _VERDICT_ACK_TOKENS.get(verdict, (str(verdict).lower(),)))


_EXPLANATION_ACK_TOKENS = ("right", "correct", "exactly", "yes", "good", "well done", "spot on", "true")


def _feedback_spoken(text: str, item: dict) -> bool:
    result = item.get("result") or {}
    lower = (text or "").lower()
    if item.get("tool") != "evaluate_explanation":
        return _verdict_acknowledged(text, result.get("verdict"))
    if result.get("key_reason_named") is True:
        return any(token in lower for token in _EXPLANATION_ACK_TOKENS)
    words_in = set(re.findall(r"[a-z]{5,}", lower))
    return any(len(set(re.findall(r"[a-z]{5,}", c.lower())) & words_in) >= 2
               for c in result.get("credited") or [])


def _state_unreported_feedback(output_text: str, state) -> str:
    """A grading verdict or verified explanation credit the reply leaves out is stated by the
    backend first, in its own verified words (run kjofd3ug S4 and S11: graded, logged, moved on)."""
    missing = []
    for item in getattr(state, "_turn_grading_results", []) or []:
        sentence = feedback_sentence(item.get("tool"), item.get("result") or {})
        if not sentence or _feedback_spoken(output_text, item) or sentence in missing:
            continue
        missing.append(sentence)
        _event_append(state, "grader_verdict_events", {
            "type": "grader_verdict_unreported", "tool": item.get("tool"),
            "verdict": (item.get("result") or {}).get("verdict"), "args": dict(item.get("args") or {}),
            "game_id": getattr(state, "current_game_id", None), "stated_by_backend": sentence,
        })
        trace(f"grader-verdict check: {item.get('tool')} feedback was missing; stated by the backend")
    return (" ".join(missing) + "\n\n" + output_text) if missing else output_text


def _run_grading_integrity_guard(output_text: str, state) -> str:
    # A rejected attempt is superseded when the same tool later accepted a call on the
    # same position this turn (the model regraded the student's real message): blocking
    # the reply then would hide a correct answer behind a confusing notice.
    results = getattr(state, "_turn_grading_results", []) or []

    def superseded(index, item):
        fen = (item.get("args") or {}).get("fen")
        return any(later.get("tool") == item.get("tool") and (later.get("args") or {}).get("fen") == fen
                   and (later.get("result") or {}).get("verdict") != "stale_or_fabricated_student_input"
                   and not (later.get("result") or {}).get("error")
                   for later in results[index + 1:])

    stale = [item for index, item in enumerate(results)
             if (item.get("result") or {}).get("verdict") == "stale_or_fabricated_student_input"
             and not superseded(index, item)]
    if not stale:
        return output_text
    for item in stale:
        _event_append(state, "grading_integrity_events", {
            "type": "stale_grading_response_blocked",
            "tool": item.get("tool"),
            "args": dict(item.get("args") or {}),
            "game_id": getattr(state, "current_game_id", None),
        })
    trace("grading-integrity guard: blocked response after stale/fabricated grading input")
    return (
        "I need to correct the flow: I tried to grade a move or threat that was not actually in your "
        "latest message, so I will not treat it as verified. Please restate the candidate move or "
        "threat you want me to check, or ask me to explain the position directly."
    )


def _move_mentioned_near_label(text: str, move: str, label_pattern: str) -> bool:
    if not move:
        return False
    return bool(re.search(
        rf"`?{re.escape(move)}`?[^.?!\n]{{0,80}}\b{label_pattern}\b|"
        rf"\b{label_pattern}\b[^.?!\n]{{0,80}}`?{re.escape(move)}`?",
        text,
        flags=re.IGNORECASE,
    ))


def _run_verdict_contradiction_lint(output_text: str, state) -> None:
    review = getattr(state, "review", None)
    if not review:
        return
    for record in getattr(review, "agenda", []) or []:
        bad_quality = getattr(record, "quality", None) in ("blunder", "mistake", "inaccuracy")
        if bad_quality and _move_mentioned_near_label(output_text, record.san, r"(?:best|good|correct|holds)"):
            _event_append(state, "verdict_contradiction_events", {
                "type": "verdict_contradiction",
                "move": record.san,
                "move_label": record.move_label,
                "verified_quality": record.quality,
                "claim": "praised_played_mistake",
            })
            trace(f"verdict-contradiction lint: {record.san} praised despite {record.quality}")
        if getattr(record, "best_move_san", None):
            if _move_mentioned_near_label(output_text, record.best_move_san, r"(?:blunder|mistake|bad|wrong)"):
                _event_append(state, "verdict_contradiction_events", {
                    "type": "verdict_contradiction",
                    "move": record.best_move_san,
                    "move_label": record.best_move_label,
                    "verified_quality": "best_move",
                    "claim": "criticized_verified_best",
                })
                trace(f"verdict-contradiction lint: best move {record.best_move_san} was criticized")


_DELIBERATION_RE = re.compile(
    r"\b(?:but wait|wait,?\s+that'?s confusing|ah,?\s+i see|let me think|"
    r"thinking out loud|scratch that)\b",
    re.IGNORECASE,
)
_VERIFY_LANGUAGE_RE = re.compile(
    r"\b(?:let me|i(?:'ll| will| need to)?)\s+(?:check|verify|re-check|recheck)\b|"
    r"\b(?:checking|verifying|rechecking)\s+(?:the\s+)?(?:engine|line|position|variation)\b",
    re.IGNORECASE,
)


def _run_deliberation_lint(output_text: str, state) -> None:
    for match in _DELIBERATION_RE.finditer(output_text or ""):
        _event_append(state, "deliberation_events", {
            "type": "deliberation_marker",
            "marker": match.group(0),
            "game_id": getattr(state, "current_game_id", None),
        })
        trace(f"deliberation lint: marker '{match.group(0)}'")
    if _VERIFY_LANGUAGE_RE.search(output_text or "") and not (getattr(state, "_turn_tool_names", []) or []):
        _event_append(state, "deliberation_events", {
            "type": "claimed_verification_without_tool",
            "game_id": getattr(state, "current_game_id", None),
        })
        trace("deliberation lint: claimed verification without tool call")


def _render(full_text: str, state, config=None) -> None:
    if not full_text:
        console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")
        return

    output_text = strip_think_blocks(full_text.strip())
    output_text = re.sub(r"^(?:(?:Output|Done|End|Yes|I will output now)\.\s*)+", "", output_text, flags=re.IGNORECASE).strip()

    output_text = _run_grading_integrity_guard(output_text, state)
    output_text = _run_claim_gate(output_text, state, config)
    _run_ambiguous_piece_lint(output_text, state)
    output_text = _state_unreported_feedback(output_text, state)
    _run_verdict_contradiction_lint(output_text, state)
    _run_deliberation_lint(output_text, state)
    output_text = normalize_move_styling(output_text)
    segments, scrubbed = _split_board_segments(output_text, state.board_visuals)
    if hasattr(state, "last_coach_reply"):
        state.last_coach_reply = "\n".join(payload for kind, payload in segments if kind != "board")
    if scrubbed:
        console.print(f"[dim red]({scrubbed} invented board token(s) scrubbed from the response)[/dim red]")

    if any(kind == "board" or payload.strip() for kind, payload in segments):
        console.print("\n[bold purple]Coach:[/bold purple]")
        for kind, payload in segments:
            if kind == "board":
                console.print()
                console.print(payload)
                console.print()
            elif payload.strip():
                console.print(Markdown(payload))
