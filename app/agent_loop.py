"""The Gemini tool-execution loop and response rendering.

Extracted from coach.py. Internal <think> scratchpads are stripped before
display — the student sees the lesson, not the model's bookkeeping.
"""
import re
import time

from google.genai import types
from rich.markdown import Markdown

from app.trace import fmt_args, trace
from app.ui import console

TOOL_MESSAGES = {
    "fetch_lichess_game_by_id": "[dim cyan]↳ Fetching game from Lichess...[/dim cyan]",
    "fetch_chessgames_com_game": "[dim cyan]↳ Fetching Master Game from Chessgames.com...[/dim cyan]",
    "query_endgame_tablebase": "[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]",
    "analyze_position": "[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]",
    "evaluate_student_answer": "[dim magenta]↳ Grading your answer against the engine...[/dim magenta]",
    "predict_human_move": "[dim magenta]↳ Asking Maia what a human would do here...[/dim magenta]",
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


def run_agent_turn(chat, coach_tools, message) -> None:
    """Send `message`, execute any tool calls until the model stops asking,
    then render the final response."""
    response = chat.send_message(message)
    full_text = ""

    while True:
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'text', None):
                    full_text += part.text + "\n"

        f_calls = _extract_function_calls(response)
        if not f_calls:
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
            trace(f"tool result: {fn_call.name} in {time.monotonic() - started:.1f}s -> {_summarize_result(result)}")

            tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))

        response = chat.send_message(tool_parts)

    _render(full_text, coach_tools.state)


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


def substitute_board_tokens(text: str, visuals: dict) -> tuple[str, int]:
    """Replace known board tokens with their diagrams; scrub any leftover
    (invented) tokens instead of showing them raw. Returns (text, n_scrubbed)."""
    for token, visual in visuals.items():
        text = text.replace(token, f"\n{visual}\n")
    return _UNRESOLVED_TOKEN_RE.subn("\n*(diagram unavailable — the coach referenced a board it never generated)*\n", text)


def _render(full_text: str, state) -> None:
    if not full_text:
        console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")
        return

    output_text = strip_think_blocks(full_text.strip())
    output_text = re.sub(r"^(?:(?:Output|Done|End|Yes|I will output now)\.\s*)+", "", output_text, flags=re.IGNORECASE).strip()

    output_text, scrubbed = substitute_board_tokens(output_text, state.board_visuals)
    if scrubbed:
        console.print(f"[dim red]({scrubbed} invented board token(s) scrubbed from the response)[/dim red]")

    if output_text:
        console.print("\n[bold purple]Coach:[/bold purple]")
        console.print(Markdown(output_text))
