"""Shared SAN text-parsing helpers (no engine, no UI)."""
import re
from dataclasses import dataclass

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


def san_mention_spans(text: str) -> list:
    """Every SAN-looking token with its (start, end) span, annotations excluded."""
    if not text:
        return []
    spans = []
    for match in _SAN_RE.finditer(text):
        san = match.group(0).rstrip("!?")
        spans.append((match.start(), match.start() + len(san), san))
    return spans


_PIECE_WORDS_TO_TYPE = {"king": chess.KING, "queen": chess.QUEEN, "rook": chess.ROOK,
                        "bishop": chess.BISHOP, "knight": chess.KNIGHT, "pawn": chess.PAWN}
# "queen to g6", "knight takes e4", "rook captures on d8". "knight on b5" is a location,
# not a move, so "on" alone is deliberately absent.
_DESCRIBED_MOVE_RE = re.compile(
    r"\b(king|queen|rook|bishop|knight|pawn)\s+"
    r"(?:to|takes(?:\s+on)?|captures(?:\s+on)?|goes\s+to|moves\s+to|jumps\s+to|x)\s+([a-h][1-8])\b",
    re.IGNORECASE)
_CASTLE_RE = re.compile(
    r"\bcastl(?:e|es|ing)\s+(?:on\s+the\s+)?(king\s*side|short|queen\s*side|long)\b", re.IGNORECASE)


def _described_moves(board: chess.Board, text: str) -> tuple:
    """Legal moves described in words, plus the text with those phrases blanked so their
    destination squares are not misread as pawn moves ("queen to g6" is not "g6")."""
    found = []
    masked = text
    for match in _DESCRIBED_MOVE_RE.finditer(text):
        piece_type = _PIECE_WORDS_TO_TYPE[match.group(1).lower()]
        target = chess.parse_square(match.group(2).lower())
        found.append((match.start(), [board.san(m) for m in board.legal_moves
                                      if m.to_square == target
                                      and board.piece_type_at(m.from_square) == piece_type]))
        masked = masked[:match.start()] + " " * (match.end() - match.start()) + masked[match.end():]
    for match in _CASTLE_RE.finditer(text):
        san = "O-O" if match.group(1).lower().startswith(("king", "short")) else "O-O-O"
        try:
            found.append((match.start(), [board.san(board.parse_san(san))]))
        except ValueError:
            found.append((match.start(), []))
    return [san for _, sans in sorted(found, key=lambda item: item[0]) for san in sans], masked


def legal_san_mentions(board: chess.Board, text: str) -> list:
    """Legal moves mentioned in free text, in SAN or in words ("knight to e4",
    "castle kingside"), preserving first-seen order. Only moves that are actually
    legal on `board` count — the regex proposes, parse_san disposes."""
    if not text:
        return []
    described, masked = _described_moves(board, text)
    seen, result = set(), []
    for san in described:
        if san not in seen:
            seen.add(san)
            result.append(san)
    for token in _SAN_RE.findall(masked):
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


@dataclass
class AmbiguousPieceReferenceFinding:
    phrase: str
    piece: str
    square: str
    color: str
    candidate_squares: list[str]


_PIECE_WORDS = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
}

_AMBIGUOUS_PIECE_REF_RE = re.compile(
    r"\b(?:a|an|the|their|your|his|her|black'?s|white'?s|that|this)?\s*"
    r"(?P<piece>king|queen|rook|bishop|knight)\s+"
    r"(?P<prep>to|on)\s+"
    r"(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)


def ambiguous_piece_references(text: str, board: chess.Board) -> list[AmbiguousPieceReferenceFinding]:
    """Shadow-lint ambiguous prose like "rook to d8".

    If multiple same-color pieces of that type could already relate to the
    square, the coach should echo exact SAN from the verified line instead
    (e.g. `Rad8`). This function never decides legality or mutates output.
    """
    if not text:
        return []
    findings = []
    for match in _AMBIGUOUS_PIECE_REF_RE.finditer(text):
        piece_word = match.group("piece").lower()
        piece_type = _PIECE_WORDS[piece_word]
        square_name = match.group("square").lower()
        target = chess.parse_square(square_name)
        for color in (chess.WHITE, chess.BLACK):
            candidates = []
            for from_sq in board.pieces(piece_type, color):
                if from_sq == target or target in board.attacks(from_sq):
                    candidates.append(chess.square_name(from_sq))
            if len(candidates) >= 2:
                findings.append(AmbiguousPieceReferenceFinding(
                    phrase=match.group(0).strip(),
                    piece=piece_word,
                    square=square_name,
                    color="white" if color == chess.WHITE else "black",
                    candidate_squares=sorted(candidates),
                ))
    return findings
