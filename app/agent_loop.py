"""The Gemini tool-execution loop and response rendering.

Extracted from coach.py. Internal <think> scratchpads are stripped before
display — the student sees the lesson, not the model's bookkeeping.
"""
import re
import time

import chess
from google.genai import types
from rich.markdown import Markdown

from app.notation import san_mentions
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


def _send_with_spinner(chat, payload):
    """The spinner only covers MODEL calls. Tools run outside any live display
    so long-running ones (game reviews) can show their own progress bars —
    Rich allows a single live display at a time."""
    with console.status("[bold yellow]Coach is thinking...[/bold yellow]", spinner="dots"):
        return chat.send_message(payload)


def run_agent_turn(chat, coach_tools, message) -> None:
    """Send `message`, execute any tool calls until the model stops asking,
    then render the final response."""
    if hasattr(coach_tools.state, "claim_gate_extra_fens"):
        coach_tools.state.claim_gate_extra_fens.clear()
    response = _send_with_spinner(chat, message)
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
            trace(f"tool result: {fn_call.name} in {time.monotonic() - started:.1f}s -> {_summarize_result(result)}")

            tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))

        response = _send_with_spinner(chat, tool_parts)

    _render(full_text, coach_tools.state, getattr(coach_tools, "config", None))
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


def candidate_boards(state) -> list:
    """The positions the coach may legitimately reference this turn: the live
    board plus every agenda moment's before/after FEN and verified line positions.
    A move legal in any of these is allowed; one legal in none is 'impossible'
    (likely hallucinated)."""
    fens, boards = set(), []
    board = getattr(state, "board", None)
    if board is not None:
        fens.add(board.fen())
    for f in getattr(state, "claim_gate_extra_fens", []) or []:
        if f:
            fens.add(f)
    review = getattr(state, "review", None)
    for r in (getattr(review, "agenda", None) or []):
        for f in (getattr(r, "fen_before", None), getattr(r, "fen_after", None)):
            if f:
                fens.add(f)
        for f in _line_position_fens(getattr(r, "fen_before", None), getattr(r, "best_line", None)):
            fens.add(f)
        for f in _line_position_fens(getattr(r, "fen_after", None), getattr(r, "punishment_line", None)):
            fens.add(f)
    for f in fens:
        try:
            boards.append(chess.Board(f))
        except ValueError:
            pass
    return boards


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


def find_illegal_moves(text: str, boards: list) -> list:
    """SAN-looking moves that are illegal in EVERY candidate board (impossible
    given the positions in play). Zero false positives by construction: a move
    legal in any known position is allowed. Board tokens are stripped first so
    their hex ids can't masquerade as moves."""
    if not boards:
        return []  # no known positions to check against — never flag blind
    clean = _BOARD_TOKEN_RE.sub(" ", text)
    return [san for san in san_mentions(clean)
            if not any(_parses(b, san) for b in boards)]


def apply_claim_gate(text: str, boards: list, level: int) -> tuple:
    """Run the claim gate at `level`. Returns (text, findings).
    L0: detect only (text unchanged) — shadow/log mode.
    L1+: redact moves illegal in every known position (degrade to less detail,
         never a non-sequitur — the redaction is visible, not a silent hole)."""
    findings = [{"type": "illegal_move", "token": san} for san in find_illegal_moves(text, boards)]
    if level >= 1 and findings:
        for f in findings:
            text = re.sub(rf"`?\b{re.escape(f['token'])}\b`?", "(an unverified move)", text)
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
    boards = candidate_boards(state)
    new_text, findings = apply_claim_gate(output_text, boards, level)
    if findings:
        _record_claim_gate_event(state, level=level, boards=boards,
                                 before=output_text, after=new_text, findings=findings)
        tokens = ", ".join(f["token"] for f in findings)
        trace(f"claim-gate (L{level}): {len(findings)} illegal-move finding(s): {tokens} "
              + ("[shadow: not modified]" if level < 1 else "[redacted]"))
    return new_text


def _render(full_text: str, state, config=None) -> None:
    if not full_text:
        console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")
        return

    output_text = strip_think_blocks(full_text.strip())
    output_text = re.sub(r"^(?:(?:Output|Done|End|Yes|I will output now)\.\s*)+", "", output_text, flags=re.IGNORECASE).strip()

    output_text = _run_claim_gate(output_text, state, config)
    output_text = normalize_move_styling(output_text)
    segments, scrubbed = _split_board_segments(output_text, state.board_visuals)
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
