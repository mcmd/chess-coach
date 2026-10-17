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


_SAN_PARTS_RE = re.compile(
    r"^(?P<piece>[KQRBN])?(?P<file>[a-h])?(?P<rank>[1-8])?(?P<capture>x)?(?P<to>[a-h][1-8])"
    r"(?:=?(?P<promo>[QRBN]))?[+#]?[!?]*$")


def illegal_move_reason(board: chess.Board, san: str, who: str = "your") -> str | None:
    """Why `san` is not a legal move for the side to move, read from the board itself (owner decision D5:
    live runs invented "the queen can't jump to h5" when the queen's own d5 pawn blocked it). `who` names
    the mover ("your", or "Black's" for an opponent reply). None when no specific reason is found."""
    side = board.turn
    own = "your own" if who == "your" else f"{who} own"
    other = "black" if side == chess.WHITE else "white"
    text = (san or "").strip().replace("0", "O").rstrip("+#!?")
    try:
        board.parse_san(text)
        return None
    except ValueError:
        pass
    if text in ("O-O", "O-O-O"):
        long = text == "O-O-O"
        rook_file = 0 if long else 7
        if not (board.has_queenside_castling_rights(side) if long else board.has_kingside_castling_rights(side)):
            return "castling that way is no longer possible: the king or that rook has already moved"
        if board.is_check():
            return "you cannot castle while in check"
        king = board.king(side)
        rook = chess.square(rook_file, chess.square_rank(king))
        if chess.SquareSet(chess.between(king, rook)) & board.occupied:
            return "there are pieces between the king and the rook"
        return "the king would pass through or land on an attacked square"
    parts = _SAN_PARTS_RE.match(text)
    if not parts:
        return "that is not a move written in chess notation"
    piece_type = chess.PIECE_SYMBOLS.index(parts["piece"].lower()) if parts["piece"] else chess.PAWN
    name = chess.piece_name(piece_type)
    to = chess.parse_square(parts["to"])

    def matches(sq):
        return ((not parts["file"] or chess.square_file(sq) == chess.FILE_NAMES.index(parts["file"]))
                and (not parts["rank"] or chess.square_rank(sq) == int(parts["rank"]) - 1))

    pieces = [sq for sq in board.pieces(piece_type, side) if matches(sq)]
    if not pieces:
        owner = "you have" if who == "your" else f"{who.removesuffix(chr(39) + 's')} has"
        return f"{who} {name} is not there" if parts["file"] or parts["rank"] else f"{owner} no {name}"
    target = board.piece_at(to)
    if target and target.color == side:
        return f"{own} {chess.piece_name(target.piece_type)} is already on {parts['to']}"
    if piece_type == chess.PAWN:
        step = 8 if side == chess.WHITE else -8
        diagonal = parts["capture"] or (parts["file"] and chess.square_file(to) != chess.FILE_NAMES.index(parts["file"]))
        if diagonal:
            if not any(to in chess.SquareSet(chess.BB_PAWN_ATTACKS[side][sq]) for sq in pieces):
                return f"none of {who} pawns can capture on {parts['to']}"
            if not target and board.ep_square != to:
                return f"a pawn captures only onto an enemy piece, and {parts['to']} is empty"
        else:
            if target:
                return f"a pawn cannot capture straight ahead: the {other} {chess.piece_name(target.piece_type)} on {parts['to']} blocks it"
            starts = [sq for sq in pieces if sq + step == to or (sq + 2 * step == to and chess.square_rank(sq) in (1, 6))]
            if not starts:
                return "a pawn moves straight forward one square (two from its starting square)"
            if all(sq + 2 * step == to and board.piece_at(sq + step) for sq in starts):
                return f"the square in front of the pawn ({chess.square_name(starts[0] + step)}) is occupied"
        if chess.square_rank(to) in (0, 7) and not parts["promo"]:
            return f"a pawn reaching {parts['to']} must promote (for example {parts['to']}=Q)"
    else:
        empty = chess.Board.empty()
        reachable = []
        for sq in pieces:
            empty.clear()
            empty.set_piece_at(sq, chess.Piece(piece_type, side))
            if to in empty.attacks(sq):
                reachable.append(sq)
        if not reachable:
            if len(pieces) == 1:
                return f"{who} {name} on {chess.square_name(pieces[0])} does not move like that"
            return f"none of {who} {name}s can move like that to {parts['to']}"
        for sq in reachable:
            blockers = chess.SquareSet(chess.between(sq, to)) & board.occupied
            if blockers and len(reachable) == 1:
                nearest = min(blockers, key=lambda b: chess.square_distance(b, sq))
                blocker = board.piece_at(nearest)
                owner = own if blocker.color == side else f"the {other}"
                return (f"{owner} {chess.piece_name(blocker.piece_type)} on {chess.square_name(nearest)} is in the way "
                        f"of {who} {name} on {chess.square_name(sq)}")
    pseudo = [m for m in board.pseudo_legal_moves
              if m.to_square == to and board.piece_type_at(m.from_square) == piece_type and matches(m.from_square)]
    if pseudo:
        if board.is_check():
            return f"{who} king is in check, and {san.strip()} does not get out of it"
        if piece_type == chess.KING:
            return f"{who} king would be in check on {parts['to']}"
        frm = pseudo[0].from_square
        if board.is_pinned(side, frm):
            ray = board.pin(side, frm)
            pinner = next((s for s in ray if board.piece_at(s) and board.piece_at(s).color != side), None)
            if pinner is not None:
                return (f"{who} {name} on {chess.square_name(frm)} is pinned to {who} king by the {other} "
                        f"{chess.piece_name(board.piece_type_at(pinner))} on {chess.square_name(pinner)}")
        return f"it would leave {who} king in check"
    return None
