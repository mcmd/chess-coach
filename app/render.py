"""Deterministic board rendering, bound to opaque tokens.

The LLM never draws boards itself: it outputs a token like [[BOARD_a1b2c3d4]]
and the CLI substitutes the backend-rendered diagram at display time.

Boards are emitted as Rich console markup (not a Markdown code fence) so each
square can carry its own background colour; app.agent_loop renders these
segments straight through the Rich console rather than through Markdown, which
would otherwise discard the styling and impose its own code-block background.
"""
import uuid

import chess

# Side is conveyed by glyph fill, the canonical Unicode convention: WHITE pieces
# use the hollow/outline glyphs (U+2654–9), BLACK pieces use the solid/filled
# glyphs (U+265A–F). U+FE0E (text-presentation selector) stops terminals from
# rendering an emoji for the solid pawn (U+265F), which otherwise came out
# double-width and knocked the columns out of alignment.
_VS_TEXT = "︎"
_WHITE_GLYPH = {
    chess.KING: "♔",
    chess.QUEEN: "♕",
    chess.ROOK: "♖",
    chess.BISHOP: "♗",
    chess.KNIGHT: "♘",
    chess.PAWN: "♙",
}
_BLACK_GLYPH = {
    chess.KING: "♚",
    chess.QUEEN: "♛",
    chess.ROOK: "♜",
    chess.BISHOP: "♝",
    chess.KNIGHT: "♞",
    chess.PAWN: "♟",
}

# Defaults: lichess "green" board. Square colours are configurable; piece ink is
# derived from each square's luminance so pieces stay legible on any palette.
DEFAULT_LIGHT = "#eeeed2"
DEFAULT_DARK = "#769656"
DEFAULT_SCALE = 2

_DARK_INK = "#1c1c1c"   # for pieces sitting on a light square
_LIGHT_INK = "#f2f2f2"  # for pieces sitting on a dark square


def _parse_hex(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _luminance(color: str) -> float:
    r, g, b = _parse_hex(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _board_ink(light: str, dark: str) -> str:
    """One ink for the whole board, chosen from its overall lightness. A single
    ink keeps each piece a fixed colour on every square — side is conveyed by
    glyph fill (hollow = white, solid = black), never by the ink — so a black
    rook never reads as white just because it sits on a dark square."""
    try:
        overall = (_luminance(light) + _luminance(dark)) / 2
    except (ValueError, IndexError):
        return _DARK_INK
    return _DARK_INK if overall >= 150 else _LIGHT_INK


def render_board_markup(board: chess.Board, pov: str = "white",
                        light: str = DEFAULT_LIGHT, dark: str = DEFAULT_DARK,
                        scale: int = DEFAULT_SCALE) -> str:
    """Build the coloured Rich-markup diagram. Each square is a `2*scale`-wide by
    `scale`-tall block of coloured cells with the piece glyph centred, so the
    board reads larger relative to the surrounding text without changing the
    terminal font."""
    black_pov = pov.lower() == "black"
    ranks = list(range(8)) if black_pov else list(range(7, -1, -1))
    files = list(range(7, -1, -1)) if black_pov else list(range(8))

    cell_w = max(2, 2 * scale)
    cell_h = max(1, scale)
    glyph_row = cell_h // 2
    left = (cell_w - 1) // 2
    right = cell_w - 1 - left

    ink = _board_ink(light, dark)

    lines = []
    for rank in ranks:
        for sub in range(cell_h):
            prefix = f"{rank + 1} " if sub == glyph_row else "  "
            row = prefix
            for f in files:
                is_light = (f + rank) % 2 == 1
                bg = light if is_light else dark
                piece = board.piece_at(chess.square(f, rank))
                if piece is not None and sub == glyph_row:
                    table = _WHITE_GLYPH if piece.color == chess.WHITE else _BLACK_GLYPH
                    glyph = table[piece.piece_type] + _VS_TEXT
                    body = (" " * left) + glyph + (" " * right)
                    row += f"[{ink} on {bg}]{body}[/]"
                else:
                    row += f"[on {bg}]{' ' * cell_w}[/]"
            lines.append(row)

    file_letters = "hgfedcba" if black_pov else "abcdefgh"
    label = "  " + "".join((" " * left) + c + (" " * right) for c in file_letters)
    lines.append(label)
    return "\n".join(lines)


def render_board_token(state, board: chess.Board, pov: str = "white", invert: bool = False) -> str:
    """Render `board` to coloured Rich markup, store it on the session state, and
    return the placeholder token the model embeds in its prose.

    Square colours and scale come from the session theme (set from config at
    startup). `invert` is accepted for backward compatibility; explicit per-square
    colours now make the diagram terminal-independent, so it no longer flips glyphs.
    """
    visual = render_board_markup(
        board, pov,
        light=getattr(state, "board_light_color", DEFAULT_LIGHT),
        dark=getattr(state, "board_dark_color", DEFAULT_DARK),
        scale=getattr(state, "board_scale", DEFAULT_SCALE),
    )

    if len(state.board_visuals) > 50:
        state.board_visuals.pop(next(iter(state.board_visuals)))

    token = f"[[BOARD_{str(uuid.uuid4())[:8]}]]"
    state.board_visuals[token] = visual
    return token
