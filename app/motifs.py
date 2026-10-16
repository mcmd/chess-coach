"""Tactical motif detection over forcing engine lines.

These detectors are deliberately run on engine principal variations (short,
forced sequences) rather than arbitrary quiet positions: that keeps precision
high. Each motif kind is reported at most once per line.
"""
import chess

from app.concepts import PIECE_VALUES, see_exchange_gain

_RAY_DIRS = {
    chess.BISHOP: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
    chess.ROOK: [(1, 0), (-1, 0), (0, 1), (0, -1)],
}
_RAY_DIRS[chess.QUEEN] = _RAY_DIRS[chess.BISHOP] + _RAY_DIRS[chess.ROOK]


def _piece_label(piece: chess.Piece, sq: chess.Square) -> str:
    return f"{chess.piece_name(piece.piece_type)} on {chess.square_name(sq)}"


def _walk_ray(board: chess.Board, start: chess.Square, df: int, dr: int):
    """Yield (square, piece) for occupied squares along a ray, near to far."""
    f, r = chess.square_file(start) + df, chess.square_rank(start) + dr
    while 0 <= f <= 7 and 0 <= r <= 7:
        sq = chess.square(f, r)
        piece = board.piece_at(sq)
        if piece:
            yield sq, piece
        f, r = f + df, r + dr


def _detect_fork(board: chess.Board, move: chess.Move, mover: chess.Color):
    """After `move` was pushed: does the moved piece attack 2+ valuable targets
    while not being profitably capturable itself? Returns [(label, square, is_king)] or None."""
    piece = board.piece_at(move.to_square)
    if piece is None:
        return None
    attacker_val = PIECE_VALUES[piece.piece_type]
    targets = []
    for tsq in board.attacks(move.to_square):
        target = board.piece_at(tsq)
        if not target or target.color == mover:
            continue
        defended = bool(board.attackers(target.color, tsq))
        if (target.piece_type == chess.KING
                or PIECE_VALUES[target.piece_type] > attacker_val
                or not defended):
            targets.append((_piece_label(target, tsq), tsq, target.piece_type == chess.KING))
    if len(targets) < 2:
        return None
    # A "fork" the opponent can simply capture for profit is not a fork.
    if see_exchange_gain(board, move.to_square) > 0:
        return None
    return targets


def _detect_pin_skewer(board: chess.Board, move: chess.Move, mover: chess.Color):
    """After `move` was pushed: did the moved piece create a pin or skewer?"""
    piece = board.piece_at(move.to_square)
    if piece is None or piece.piece_type not in _RAY_DIRS:
        return None
    for df, dr in _RAY_DIRS[piece.piece_type]:
        hits = []
        for sq, hit in _walk_ray(board, move.to_square, df, dr):
            hits.append((sq, hit))
            if len(hits) == 2:
                break
        if len(hits) < 2:
            continue
        (front_sq, front), (back_sq, back) = hits
        if front.color == mover or back.color == mover:
            continue
        front_val, back_val = PIECE_VALUES[front.piece_type], PIECE_VALUES[back.piece_type]
        if front.piece_type == chess.KING:
            continue  # that's a check, not a pin/skewer of the front piece
        if back.piece_type == chess.KING or back_val > front_val:
            return ("pin", f"{_piece_label(front, front_sq)} is pinned against {_piece_label(back, back_sq)}")
        if front_val > back_val:
            return ("skewer", f"{_piece_label(front, front_sq)} is skewered, exposing {_piece_label(back, back_sq)}")
    return None


def _detect_discovered_attack(board: chess.Board, move: chess.Move, mover: chess.Color):
    """After `move` was pushed: did it unveil a slider attack through the
    vacated square onto a valuable target (queen/rook, or an undefended minor)?
    Discovered CHECKS are reported separately."""
    for slider_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s_sq in board.pieces(slider_type, mover):
            if s_sq == move.to_square:
                continue  # the moved piece itself is the direct attacker, not a discovery
            for t_sq in board.attacks(s_sq):
                target = board.piece_at(t_sq)
                if (not target or target.color == mover
                        or target.piece_type in (chess.KING, chess.PAWN)):
                    continue
                if PIECE_VALUES[target.piece_type] < 5 and board.attackers(target.color, t_sq):
                    continue  # minor piece that is defended: not a meaningful discovery
                if move.from_square in chess.SquareSet.between(s_sq, t_sq):
                    return (f"{_piece_label(board.piece_at(s_sq), s_sq)} is unveiled, "
                            f"attacking {_piece_label(target, t_sq)}")
    return None


def _is_smothered_mate(board: chess.Board) -> bool:
    """Checkmate by a lone knight with the king buried under its own pieces."""
    king_sq = board.king(board.turn)
    checkers = list(board.checkers())
    if king_sq is None or len(checkers) != 1:
        return False
    checker = board.piece_at(checkers[0])
    if not checker or checker.piece_type != chess.KNIGHT:
        return False
    return all(board.piece_at(adj) and board.piece_at(adj).color == board.turn
               for adj in board.attacks(king_sq))


def _detect_removal_of_defender(board: chess.Board, move: chess.Move, mover: chess.Color):
    """Before `move` is pushed: does it capture a piece that was the SOLE
    defender of another attacked piece?"""
    if not board.is_capture(move) or board.is_en_passant(move):
        return None
    victim_sq = move.to_square
    victim = board.piece_at(victim_sq)
    if victim is None:
        return None
    for psq in chess.SQUARES:
        protected = board.piece_at(psq)
        if (not protected or protected.color != victim.color or psq == victim_sq
                or PIECE_VALUES[protected.piece_type] < 3):
            continue
        defenders = board.attackers(victim.color, psq)
        if defenders == chess.SquareSet([victim_sq]) and board.attackers(mover, psq):
            return f"capturing {_piece_label(victim, victim_sq)} removes the only defender of {_piece_label(protected, psq)}"
    return None


def detect_motifs_in_line(board: chess.Board, moves) -> list[dict]:
    """Detect tactical motifs along a line of moves (SAN strings or Moves)
    played from `board`. Returns [{motif, move, detail, side, ply}], each kind at most once.
    `side` is the colour that played the motif's move and `ply` its offset in the line. A fork also
    has `material_won`: whether the forking side captures one of its non-king targets later in this
    line. Geometry alone is not a won fork (2dtDqQgR 26. Qxa7 Qxg4+ "forked" b4 and d1, winning
    neither, and was diagnosed missed_fork)."""
    b = board.copy(stack=False)
    found, seen = [], set()
    forks = []  # every fork occurrence; the reported one per kind is chosen at the end
    current = {}

    def add(kind, san, detail):
        if kind not in seen:
            seen.add(kind)
            found.append({"motif": kind, "move": san, "detail": detail, **current})

    prev_capture_sq = None        # square where the previous mover just captured
    zwischenzug_candidates = []   # declined recaptures awaiting confirmation

    for ply, mv in enumerate(moves):
        try:
            move = b.parse_san(mv) if isinstance(mv, str) else mv
            if move not in b.legal_moves:
                break
            san = b.san(move)
        except ValueError:
            break
        mover = b.turn
        current = {"side": "white" if mover == chess.WHITE else "black", "ply": ply}
        is_capture = b.is_capture(move)
        for fork in forks:
            if (fork["side"] == current["side"] and is_capture and move.to_square in fork["targets"]
                    and b.piece_at(move.to_square) and b.piece_at(move.to_square).piece_type == fork["targets"][move.to_square]):
                fork["material_won"] = True

        # Zwischenzug: a recapture is available, but the line plays a forcing
        # in-between move first and only recaptures within the next two moves.
        if prev_capture_sq is not None and move.to_square != prev_capture_sq:
            recapture_available = any(m.to_square == prev_capture_sq
                                      for m in b.generate_legal_captures())
            victim = b.piece_at(prev_capture_sq)
            if recapture_available and victim is not None:
                bigger_capture = (is_capture and b.piece_at(move.to_square) is not None
                                  and PIECE_VALUES[b.piece_at(move.to_square).piece_type]
                                  >= PIECE_VALUES[victim.piece_type])
                if b.gives_check(move) or bigger_capture:
                    zwischenzug_candidates.append(
                        {"san": san, "sq": prev_capture_sq, "side": mover, "expires": ply + 4})
        if is_capture:
            for cand in zwischenzug_candidates:
                if (cand["side"] == mover and cand["sq"] == move.to_square
                        and ply <= cand["expires"]):
                    add("zwischenzug", cand["san"],
                        f"{cand['san']} is a zwischenzug — the recapture on "
                        f"{chess.square_name(cand['sq'])} can wait; the in-between move comes first")

        removal = _detect_removal_of_defender(b, move, mover)
        if removal:
            add("removal_of_defender", san, removal)

        if move.promotion:
            add("promotion", san,
                f"{san} promotes — the passed pawn becomes a new {chess.piece_name(move.promotion)}")

        gives_check = b.gives_check(move)
        b.push(move)
        prev_capture_sq = move.to_square if is_capture else None

        if gives_check and len(b.checkers()) >= 2:
            add("double_check", san,
                f"{san} is DOUBLE CHECK — blocking and capturing are impossible; the king MUST move")

        if gives_check and move.to_square not in b.checkers():
            add("discovered_check", san, f"{san} unleashes a discovered check")

        discovery = _detect_discovered_attack(b, move, mover)
        if discovery:
            add("discovered_attack", san, f"after {san}, {discovery}")

        fork_targets = _detect_fork(b, move, mover)
        if fork_targets:
            piece = b.piece_at(move.to_square)
            kind = f"{chess.piece_name(piece.piece_type)}_fork" if piece else "fork"
            forks.append({"motif": kind, "move": san, **current, "material_won": False,
                          "detail": f"{san} attacks {' and '.join(t[0] for t in fork_targets)} simultaneously",
                          "targets": {sq: b.piece_at(sq).piece_type for _, sq, is_king in fork_targets
                                      if not is_king}})

        pin_skewer = _detect_pin_skewer(b, move, mover)
        if pin_skewer:
            add(pin_skewer[0], san, f"after {san}, {pin_skewer[1]}")

        if b.is_checkmate():
            ksq = b.king(b.turn)
            checkers = list(b.checkers())
            if _is_smothered_mate(b):
                add("smothered_mate", san,
                    f"{san} is a SMOTHERED MATE — the king is buried under its own pieces")
            elif (ksq is not None and chess.square_rank(ksq) in (0, 7) and checkers
                    and b.piece_at(checkers[0]) and b.piece_at(checkers[0]).piece_type in (chess.ROOK, chess.QUEEN)
                    and chess.square_rank(checkers[0]) == chess.square_rank(ksq)):
                add("back_rank_mate", san, f"{san} delivers a back-rank mate")
            else:
                add("checkmate", san, f"{san} is checkmate")
            break

    # One fork per kind: the first that wins material, else the first.
    for kind in dict.fromkeys(f["motif"] for f in forks):
        same = [f for f in forks if f["motif"] == kind]
        chosen = next((f for f in same if f["material_won"]), same[0])
        found.append({k: v for k, v in chosen.items() if k != "targets"})
    found.sort(key=lambda m: m["ply"])
    return found
