"""Shared SAN text-parsing helpers (no engine, no UI)."""
import re

import chess

_SAN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h]x?[a-h][1-8](?:=[QRBN])?[+#]?)[!?]*(?![A-Za-z0-9_])")


def san_mentions(text: str) -> list:
    """SAN-looking tokens in free text (annotations stripped), first-seen order.
    Unlike legal_san_mentions there is no legality filter — callers decide how
    to validate each token."""
    if not text:
        return []
    seen, result = set(), []
    for token in _SAN_RE.findall(text):
        san = token.rstrip("!?")
        if san not in seen:
            seen.add(san)
            result.append(san)
    return result


def legal_san_mentions(board: chess.Board, text: str) -> list:
    """Legal SAN moves mentioned in free text, preserving first-seen order.
    Only moves that are actually legal on `board` count — the regex proposes,
    parse_san disposes."""
    if not text:
        return []
    seen, result = set(), []
    for token in _SAN_RE.findall(text):
        san = token.rstrip("!?")
        try:
            move = board.parse_san(san)
        except ValueError:
            continue
        clean = board.san(move)
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
