"""Deterministic board rendering, bound to opaque tokens.

The LLM never draws boards itself: it outputs a token like [[BOARD_a1b2c3d4]]
and the CLI substitutes the backend-rendered ASCII diagram at display time.
"""
import uuid

import chess


def render_board_token(state, board: chess.Board, pov: str = "white", invert: bool = False) -> str:
    """Render `board` to ASCII, store it on the session state, and return the
    placeholder token the model should embed in its prose."""
    raw = board.unicode(empty_square="·", invert_color=invert)
    lines = raw.split('\n')

    labeled = []
    if pov.lower() == "black":
        lines.reverse()
        for i, line in enumerate(lines):
            chars = line.split(" ")
            labeled.append(f"{i+1} │ {' '.join(chars[::-1])} │")
        labeled.append("  └" + "─" * 17 + "┘")
        labeled.append("    h g f e d c b a")
    else:
        for i, line in enumerate(lines):
            labeled.append(f"{8-i} │ {line} │")
        labeled.append("  └" + "─" * 17 + "┘")
        labeled.append("    a b c d e f g h")

    visual = "```text\n" + "\n".join(labeled) + "\n```"
    if len(state.board_visuals) > 50:
        state.board_visuals.pop(next(iter(state.board_visuals)))

    token = f"[[BOARD_{str(uuid.uuid4())[:8]}]]"
    state.board_visuals[token] = visual
    return token
