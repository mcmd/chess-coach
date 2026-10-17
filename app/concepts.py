import chess

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
}


def _get_pawn_data(board: chess.Board, color: chess.Color) -> dict:
    files = "abcdefgh"
    counts = {f: 0 for f in files}
    pawns = list(board.pieces(chess.PAWN, color))
    for sq in pawns: counts[files[chess.square_file(sq)]] += 1

    islands, current_island = [], []
    for f in files:
        if counts[f] > 0: current_island.append(f)
        else:
            if current_island:
                islands.append(f"{current_island[0]}-{current_island[-1]}" if len(current_island) > 1 else current_island[0])
                current_island = []
    if current_island:
        islands.append(f"{current_island[0]}-{current_island[-1]}" if len(current_island) > 1 else current_island[0])

    isolated = []
    for sq in pawns:
        f = chess.square_file(sq)
        left = (counts[files[f - 1]] > 0) if f > 0 else False
        right = (counts[files[f + 1]] > 0) if f < 7 else False
        if not left and not right:
            isolated.append(chess.square_name(sq))

    doubled = [f for f in files if counts[f] > 1]
    return {"total_pawns": len(pawns), "island_count": len(islands), "island_spans": islands, "isolated": isolated, "doubled_files": doubled, "file_counts": counts}


def _get_passed_pawns(board: chess.Board, color: chess.Color) -> list:
    passed = []
    opp_color = not color
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if color == chess.WHITE:
             blockers = [osq for osq in board.pieces(chess.PAWN, opp_color) if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) > r]
        else:
             blockers = [osq for osq in board.pieces(chess.PAWN, opp_color) if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) < r]

        if not blockers:
            passed.append(chess.square_name(sq))
    return passed


# --- STATIC EXCHANGE EVALUATION (pin- and x-ray-aware via legal-move simulation) ---

def _least_valuable_capture(board: chess.Board, target_sq: chess.Square):
    """Cheapest LEGAL capture on target_sq for the side to move (None if there isn't one).

    Using legal moves (not attacker bitboards) means absolutely pinned pieces are
    automatically excluded, and pieces pinned along the capture ray are allowed.
    """
    best, best_val = None, None
    for move in board.generate_legal_captures():
        if move.to_square != target_sq:
            continue
        piece = board.piece_at(move.from_square)
        val = PIECE_VALUES[piece.piece_type]
        if best is None or val < best_val:
            best, best_val = move, val
    return best


def see_exchange_gain(board: chess.Board, target_sq: chess.Square) -> int:
    """Material (in pawns) the side to move wins by capturing on target_sq.

    Standard swap-off evaluation: each side may stand pat, so the result is
    never negative. X-rays are handled naturally because each capture is
    played on a copy of the board before recursing.
    """
    victim = board.piece_at(target_sq)
    if victim is None:
        return 0
    move = _least_valuable_capture(board, target_sq)
    if move is None:
        return 0
    captured_val = PIECE_VALUES[victim.piece_type]
    child = board.copy(stack=False)
    child.push(move)
    return max(0, captured_val - see_exchange_gain(child, target_sq))


def _flip_turn(board: chess.Board, to_move: chess.Color):
    """Board copy with `to_move` to play, or None if that position is illegal
    (e.g. the other side is in check, so 'passing' makes no sense)."""
    if board.turn == to_move:
        b = board.copy(stack=False)
        b.ep_square = None
        return b
    b = board.copy(stack=False)
    b.turn = to_move
    b.ep_square = None
    return b if b.is_valid() else None


def _get_piece_safety(board: chess.Board, color: chess.Color) -> dict:
    """SEE-based piece safety for `color`'s pieces.

    hanging: the opponent wins the full piece value by capturing (e.g. truly
             undefended, or every recapture is illegal/insufficient).
    weakly_defended: the opponent wins material through the exchange, but less
             than the full piece (a profitable but contested capture).
    """
    hanging, weakly_defended, pinned = [], [], []
    opp = not color
    flipped = _flip_turn(board, opp)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color:
            continue
        sq_name = chess.square_name(sq)

        if board.is_pinned(color, sq):
            pinned.append(f"{piece.symbol()} on {sq_name}")

        if piece.piece_type == chess.KING:
            continue

        if flipped is not None:
            gain = see_exchange_gain(flipped, sq)
            if gain >= PIECE_VALUES[piece.piece_type]:
                hanging.append(f"{piece.symbol()} on {sq_name}")
            elif gain > 0:
                weakly_defended.append(f"{piece.symbol()} on {sq_name} (capture sequence wins +{gain} for opponent)")
        else:
            # Fallback (color is in check, so 'opponent to move' is illegal):
            # only flag the unambiguous case of attacked-and-undefended.
            if board.attackers(opp, sq) and not board.attackers(color, sq):
                hanging.append(f"{piece.symbol()} on {sq_name}")

    return {"hanging": hanging, "pinned": pinned, "weakly_defended": weakly_defended}


def _analyze_king_perimeter(board: chess.Board, color: chess.Color) -> dict:
    king_sq = board.king(color)
    if king_sq is None: return {"king_square": "None", "pawn_shield_squares": [], "enemy_pieces_attacking_perimeter":[]}

    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    forward_dir = 1 if color == chess.WHITE else -1

    pawn_shield = set()
    for f in [kf - 1, kf, kf + 1]:
        if 0 <= f <= 7:
            for r_offset in [0, 1, 2]:
                r = kr + (r_offset * forward_dir)
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield.add(chess.square_name(sq))

    enemy_attackers = set()
    adj_squares = [chess.square(f, r) for f in [kf-1, kf, kf+1] if 0<=f<=7 for r in [kr-1, kr, kr+1] if 0<=r<=7 and not (f==kf and r==kr)]
    for sq in adj_squares:
        for atk_sq in board.attackers(not color, sq):
            enemy_piece = board.piece_at(atk_sq)
            if enemy_piece: enemy_attackers.add(f"{enemy_piece.symbol()} on {chess.square_name(atk_sq)}")

    return {"king_square": chess.square_name(king_sq), "pawn_shield_squares": sorted(list(pawn_shield)), "enemy_pieces_attacking_perimeter": sorted(list(enemy_attackers))}


def _calc_material_score(board: chess.Board, color: chess.Color) -> int:
    return sum([len(board.pieces(p, color)) * v for p, v in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)]])


def _find_true_outposts(board: chess.Board, color: chess.Color) -> list:
    outposts = []
    valid_ranks = [3, 4, 5] if color == chess.WHITE else [2, 3, 4]
    forward_step = 1 if color == chess.WHITE else -1
    enemy_color = not color

    for sq in board.pieces(chess.KNIGHT, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)

        if r in valid_ranks:
            is_anchored = any(board.piece_at(d) and board.piece_at(d).piece_type == chess.PAWN for d in board.attackers(color, sq))
            if is_anchored:
                can_be_kicked = False
                enemy_pawns = list(board.pieces(chess.PAWN, enemy_color))
                for adj_f in [f - 1, f + 1]:
                    if 0 <= adj_f <= 7:
                        end_rank = 8 if color == chess.WHITE else -1
                        for check_r in range(r + forward_step, end_rank, forward_step):
                            if chess.square(adj_f, check_r) in enemy_pawns:
                                can_be_kicked = True; break
                if not can_be_kicked: outposts.append(chess.square_name(sq))
    return outposts


def _get_bad_bishops(board: chess.Board, color: chess.Color) -> list:
    bad_bishops =[]
    bishops = board.pieces(chess.BISHOP, color)
    pawns = board.pieces(chess.PAWN, color)
    forward_dir = 1 if color == chess.WHITE else -1

    for sq in bishops:
        is_light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 != 0
        blocked_central_pawns = 0

        for p_sq in pawns:
            f, r = chess.square_file(p_sq), chess.square_rank(p_sq)
            if 2 <= f <= 5:
                p_is_light = (f + r) % 2 != 0
                if p_is_light == is_light:
                    block_sq = chess.square(f, r + forward_dir)
                    if 0 <= r + forward_dir <= 7 and board.piece_at(block_sq):
                        blocked_central_pawns += 1

        if blocked_central_pawns >= 2:
            bad_bishops.append(chess.square_name(sq))
    return bad_bishops


def _are_rooks_connected(board: chess.Board, color: chess.Color) -> bool:
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) == 2:
        return rooks[1] in board.attacks(rooks[0])
    return False


# --- LOOSE PIECES, OVERLOAD, TENSION, TRAPPED ---

def _get_loose_pieces(board: chess.Board, color: chess.Color) -> list:
    """LPDO ('loose pieces drop off'): non-pawn pieces with ZERO defenders.
    Even when not attacked yet, these are the latent targets that make
    double attacks and forks possible."""
    loose = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if (piece and piece.color == color
                and piece.piece_type not in (chess.PAWN, chess.KING)
                and not board.attackers(color, sq)):
            loose.append(f"{piece.symbol()} on {chess.square_name(sq)}")
    return loose


def _get_overloaded_defenders(board: chess.Board, color: chess.Color) -> list:
    """Pieces of `color` that are the SOLE defender of 2+ attacked pieces —
    the precondition for removal-of-the-defender and deflection tactics."""
    opp = not color
    sole = {}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color or piece.piece_type == chess.KING:
            continue
        defenders = board.attackers(color, sq)
        if len(defenders) == 1 and board.attackers(opp, sq):
            sole.setdefault(next(iter(defenders)), []).append(sq)
    result = []
    for dsq, protected in sole.items():
        if len(protected) >= 2:
            names = " and ".join(f"{board.piece_at(p).symbol()} on {chess.square_name(p)}" for p in protected)
            result.append(f"{board.piece_at(dsq).symbol()} on {chess.square_name(dsq)} is the ONLY defender of {names}")
    return result


def _get_pawn_tension(board: chess.Board) -> list:
    """Pairs of pawns that mutually attack each other (e.g. d4<->e5).
    Deciding when to resolve, maintain, or push past tension is a core
    intermediate skill."""
    pairs = []
    for wsq in board.pieces(chess.PAWN, chess.WHITE):
        for tsq in board.attacks(wsq):
            target = board.piece_at(tsq)
            if target and target.piece_type == chess.PAWN and target.color == chess.BLACK:
                pairs.append(f"{chess.square_name(wsq)}<->{chess.square_name(tsq)}")
    return pairs


def _get_trapped_pieces(board: chess.Board, color: chess.Color) -> list:
    """Pieces under profitable attack with NO escape that doesn't lose material."""
    trapped = []
    own_view = board if board.turn == color else _flip_turn(board, color)
    opp_view = board if board.turn == (not color) else _flip_turn(board, not color)
    if own_view is None or opp_view is None:
        return []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if see_exchange_gain(opp_view, sq) <= 0:
            continue  # not under profitable attack
        has_safe_escape = False
        for move in own_view.legal_moves:
            if move.from_square != sq:
                continue
            victim = own_view.piece_at(move.to_square)
            captured_val = PIECE_VALUES[victim.piece_type] if victim else 0
            child = own_view.copy(stack=False)
            child.push(move)
            if captured_val - see_exchange_gain(child, move.to_square) >= 0:
                has_safe_escape = True
                break
        if not has_safe_escape:
            trapped.append(f"{piece.symbol()} on {chess.square_name(sq)} (attacked, no safe escape)")
    return trapped


# --- PAWN-PLAY CONCEPTS ---

def _get_backward_pawns(board: chess.Board, color: chess.Color) -> list:
    """Pawns whose neighbors have all advanced past them and whose stop square
    is controlled by an enemy pawn (isolated pawns are reported separately)."""
    backward = []
    opp = not color
    direction = 1 if color == chess.WHITE else -1
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        neighbors = [n for n in board.pieces(chess.PAWN, color)
                     if abs(chess.square_file(n) - f) == 1]
        if not neighbors:
            continue
        if any((chess.square_rank(n) - r) * direction <= 0 for n in neighbors):
            continue  # a neighbor level with or behind it can still support the advance
        stop_r = r + direction
        if not 0 <= stop_r <= 7:
            continue
        stop = chess.square(f, stop_r)
        if any(board.piece_at(a).piece_type == chess.PAWN for a in board.attackers(opp, stop)):
            backward.append(chess.square_name(sq))
    return backward


def _get_pawn_breaks(board: chess.Board, color: chess.Color) -> list:
    """Pawn-break levers: advances (one step, or two from home) to an empty
    square from which the pawn would attack an enemy pawn — the levers that
    open positions and create plans.

    Geometry alone ignores pins and checks (plan §10.1), so when `color` is to
    move the candidates are filtered through the legal moves. When the other
    side is to move they are only structural candidates, not available moves;
    the caller labels them that way."""
    breaks = []
    enemy_pawns = board.pieces(chess.PAWN, not color)
    direction = 1 if color == chess.WHITE else -1
    start_rank = 1 if color == chess.WHITE else 6
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        targets = []
        one_r = r + direction
        if 0 <= one_r <= 7 and not board.piece_at(chess.square(f, one_r)):
            targets.append(chess.square(f, one_r))
            two_r = r + 2 * direction
            if r == start_rank and 0 <= two_r <= 7 and not board.piece_at(chess.square(f, two_r)):
                targets.append(chess.square(f, two_r))
        for t in targets:
            if chess.SquareSet(chess.BB_PAWN_ATTACKS[color][t]) & enemy_pawns:
                breaks.append(chess.Move(sq, t))
    if board.turn == color:
        breaks = [m for m in breaks if board.is_legal(m)]
    return [f"{chess.square_name(m.from_square)}-{chess.square_name(m.to_square)}" for m in breaks]


# --- KING & FILE CONCEPTS ---

def _is_back_rank_vulnerable(board: chess.Board, color: chess.Color) -> bool:
    """King on its home rank with no luft (every flight square in front blocked
    by its OWN pawns), the opponent still has heavy pieces, AND those heavies
    have an entry route (a fully open file, or they already sit on the rank).
    Without a route the pattern is dormant — e.g. the starting position."""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    home = 0 if color == chess.WHITE else 7
    if chess.square_rank(king_sq) != home:
        return False
    opp = not color
    heavies = board.pieces(chess.ROOK, opp) | board.pieces(chess.QUEEN, opp)
    if not heavies:
        return False
    direction = 1 if color == chess.WHITE else -1
    kf = chess.square_file(king_sq)
    for df in (-1, 0, 1):
        f = kf + df
        if 0 <= f <= 7:
            piece = board.piece_at(chess.square(f, home + direction))
            if not (piece and piece.color == color and piece.piece_type == chess.PAWN):
                return False
    pawn_files = {chess.square_file(s)
                  for s in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)}
    has_open_file = len(pawn_files) < 8
    heavy_on_rank = any(chess.square_rank(s) == home for s in heavies)
    return has_open_file or heavy_on_rank


def _get_rooks_on_files(board: chess.Board, color: chess.Color,
                        open_files: list, half_open_files: list) -> tuple:
    files = "abcdefgh"
    on_open, on_half = [], []
    for sq in board.pieces(chess.ROOK, color):
        fname = files[chess.square_file(sq)]
        if fname in open_files:
            on_open.append(f"{chess.square_name(sq)} ({fname}-file)")
        elif fname in half_open_files:
            on_half.append(f"{chess.square_name(sq)} ({fname}-file)")
    return on_open, on_half


def _has_opposite_colored_bishops(board: chess.Board) -> bool:
    wb = list(board.pieces(chess.BISHOP, chess.WHITE))
    bb = list(board.pieces(chess.BISHOP, chess.BLACK))
    if len(wb) == 1 and len(bb) == 1:
        def is_light(s): return (chess.square_file(s) + chess.square_rank(s)) % 2 == 1
        return is_light(wb[0]) != is_light(bb[0])
    return False


def _get_mobility(board: chess.Board) -> dict:
    """Legal-move counts per side plus each side's least mobile piece — the
    honest way to talk about space and cramped positions."""
    result = {}
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        b = board if board.turn == color else _flip_turn(board, color)
        if b is None:
            continue
        counts, total = {}, 0
        for move in b.legal_moves:
            total += 1
            counts[move.from_square] = counts.get(move.from_square, 0) + 1
        result[f"{label}_total_legal_moves"] = total
        worst = None
        for sq in chess.SQUARES:
            piece = b.piece_at(sq)
            if piece and piece.color == color and piece.piece_type not in (chess.PAWN, chess.KING):
                c = counts.get(sq, 0)
                if worst is None or c < worst[1]:
                    worst = (f"{piece.symbol()} on {chess.square_name(sq)}", c)
        if worst:
            result[f"{label}_least_mobile_piece"] = f"{worst[0]} ({worst[1]} moves)"
    return result


def _observed_castling(board: chess.Board, color: chess.Color) -> bool:
    """True only when the board's own move history shows `color` castling."""
    if not board.move_stack:
        return False
    replay = board.root()
    for move in board.move_stack:
        if replay.turn == color and replay.is_castling(move):
            return True
        replay.push(move)
    return False


def _king_status(board: chess.Board, color: chess.Color) -> str:
    """Castling rights, king location, observed castling history, and an
    attack-based safety note, kept as separate facts (plan §10.1). A FEN alone
    cannot prove the king castled, and a castled-looking square is not proof
    of safety, so a g1/c1 king is only called castled when the move history
    shows it, and attacks on the king or its zone are reported, never hidden."""
    king_sq = board.king(color)
    if king_sq is None:
        return "no king on the board"
    if board.has_castling_rights(color):
        status = "can still castle"
    elif _observed_castling(board, color):
        status = f"castled (king on {chess.square_name(king_sq)})"
    elif king_sq in ((chess.G1, chess.C1) if color == chess.WHITE else (chess.G8, chess.C8)):
        status = f"on {chess.square_name(king_sq)} (castled position; castling rights gone)"
    else:
        status = "cannot castle (rights lost, king not on a castled square)"
    opp = not color
    if board.is_attacked_by(opp, king_sq):
        status += "; king is in check"
    elif any(board.is_attacked_by(opp, sq) for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])):
        status += "; enemy pieces attack the king's zone"
    return status


# --- GAME PHASE & FORCING MOVES ---

def get_game_phase(board: chess.Board) -> str:
    """Coarse phase label: opening / middlegame / endgame (heuristic)."""
    non_pawn_material = sum(
        PIECE_VALUES[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if non_pawn_material <= 13:
        return "endgame"
    if board.fullmove_number <= 10:
        return "opening"
    return "middlegame"


def get_forcing_moves(board: chess.Board) -> dict:
    """Checks and captures for the side to move — the calculation scaffold."""
    checks, captures = [], []
    for move in board.legal_moves:
        if board.gives_check(move):
            checks.append(board.san(move))
        elif board.is_capture(move):
            captures.append(board.san(move))
    return {"checks": checks[:12], "captures": captures[:12]}


# --- ENDGAME FEATURES (only computed in the endgame phase) ---

def _wrong_bishop_warning(board: chess.Board):
    """K + lone bishop + rook pawn(s) vs lone king, with the bishop unable to
    control the promotion corner: a theoretical draw if the defender's king
    reaches the corner."""
    def is_light(s): return (chess.square_file(s) + chess.square_rank(s)) % 2 == 1
    for color, label in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        bishops = list(board.pieces(chess.BISHOP, color))
        pawns = list(board.pieces(chess.PAWN, color))
        own_others = any(p.color == color and p.piece_type not in (chess.KING, chess.BISHOP, chess.PAWN)
                         for p in board.piece_map().values())
        opp_material = any(p.color != color and p.piece_type != chess.KING
                           for p in board.piece_map().values())
        if len(bishops) != 1 or not pawns or own_others or opp_material:
            continue
        pawn_files = {chess.square_file(s) for s in pawns}
        if not (pawn_files <= {0} or pawn_files <= {7}):
            continue
        corner = chess.square(next(iter(pawn_files)), 7 if color == chess.WHITE else 0)
        if is_light(bishops[0]) != is_light(corner):
            enemy_king = board.king(not color)
            dist = chess.square_distance(enemy_king, corner) if enemy_king is not None else 99
            return (f"{label} has the WRONG-COLORED BISHOP for the {chess.square_name(corner)} corner: "
                    f"this is a theoretical DRAW if the defending king reaches the corner "
                    f"(defending king is {dist} squares away).")
    return None


def _get_endgame_features(board: chess.Board) -> dict:
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    features = {}

    if wk is not None and bk is not None:
        same_file = chess.square_file(wk) == chess.square_file(bk) and abs(chess.square_rank(wk) - chess.square_rank(bk)) == 2
        same_rank = chess.square_rank(wk) == chess.square_rank(bk) and abs(chess.square_file(wk) - chess.square_file(bk)) == 2
        features["kings_in_direct_opposition"] = same_file or same_rank
        if same_file or same_rank:
            # The side NOT to move "has" the opposition.
            features["opposition_held_by"] = "black" if board.turn == chess.WHITE else "white"

    only_kings_and_pawns = all(p.piece_type in (chess.PAWN, chess.KING)
                               for p in board.piece_map().values())

    for color, label in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        passers = _get_passed_pawns(board, color)
        enemy_king = board.king(not color)
        own_king = board.king(color)
        passer_info = []
        for sq_name in passers:
            sq = chess.parse_square(sq_name)
            info = {"pawn": sq_name}
            if enemy_king is not None:
                info["enemy_king_distance"] = chess.square_distance(sq, enemy_king)
                info["is_outside_passer"] = abs(chess.square_file(sq) - chess.square_file(enemy_king)) >= 3
                if only_kings_and_pawns:
                    # Rule of the square: can the defending king catch the runner?
                    promo_rank = 7 if color == chess.WHITE else 0
                    promo_sq = chess.square(chess.square_file(sq), promo_rank)
                    steps = abs(promo_rank - chess.square_rank(sq))
                    on_start = chess.square_rank(sq) == (1 if color == chess.WHITE else 6)
                    if on_start:
                        steps -= 1  # double-step: treat as one rank further on
                    allowed = steps if board.turn != color else steps - 1
                    info["enemy_king_inside_the_square"] = chess.square_distance(enemy_king, promo_sq) <= allowed
            if own_king is not None:
                info["own_king_distance"] = chess.square_distance(sq, own_king)
            rooks_behind = [chess.square_name(r) for r in board.pieces(chess.ROOK, color)
                            if chess.square_file(r) == chess.square_file(sq)
                            and ((color == chess.WHITE and chess.square_rank(r) < chess.square_rank(sq))
                                 or (color == chess.BLACK and chess.square_rank(r) > chess.square_rank(sq)))]
            if rooks_behind:
                info["own_rook_behind_passer"] = rooks_behind
            passer_info.append(info)
        if passer_info:
            features[f"{label}_passed_pawn_details"] = passer_info

    warning = _wrong_bishop_warning(board)
    if warning:
        features["drawing_pattern_warning"] = warning

    return features


def extract_board_features(board: chess.Board) -> dict:
    """Rigorous extraction of structural features grounded in exact rules where
    possible (SEE-based piece safety) and labeled heuristics elsewhere."""
    features = {}
    files = "abcdefgh"

    features["game_phase"] = get_game_phase(board)
    features["side_to_move"] = "white" if board.turn == chess.WHITE else "black"

    w_data = _get_pawn_data(board, chess.WHITE)
    b_data = _get_pawn_data(board, chess.BLACK)

    open_files, half_open_w, half_open_b = [], [], []
    for f in files:
        w, b = w_data["file_counts"][f], b_data["file_counts"][f]
        if w == 0 and b == 0: open_files.append(f)
        elif w == 0 and b > 0: half_open_w.append(f)
        elif b == 0 and w > 0: half_open_b.append(f)

    features["pawn_structure"] = {
        "white_passed_pawns": _get_passed_pawns(board, chess.WHITE),
        "black_passed_pawns": _get_passed_pawns(board, chess.BLACK),
        "white_isolated_pawns": w_data["isolated"], "black_isolated_pawns": b_data["isolated"],
        "white_doubled_files": w_data["doubled_files"], "black_doubled_files": b_data["doubled_files"],
        "open_files": open_files, "half_open_for_white": half_open_w, "half_open_for_black": half_open_b
    }

    # New pawn-play concepts (emitted only when present, to keep the packet lean)
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        backward = _get_backward_pawns(board, color)
        if backward:
            features["pawn_structure"][f"{label}_backward_pawns"] = backward
        breaks = _get_pawn_breaks(board, color)
        if breaks:
            key = "available_pawn_breaks" if board.turn == color else "structural_break_candidates"
            features["pawn_structure"][f"{label}_{key}"] = breaks
    tension = _get_pawn_tension(board)
    if tension:
        features["pawn_structure"]["pawn_tension_points"] = tension

    # Pawn Majorities
    w_q = sum(w_data["file_counts"][f] for f in ['a', 'b', 'c'])
    b_q = sum(b_data["file_counts"][f] for f in ['a', 'b', 'c'])
    w_k = sum(w_data["file_counts"][f] for f in ['f', 'g', 'h'])
    b_k = sum(b_data["file_counts"][f] for f in ['f', 'g', 'h'])

    majorities = []
    if w_q > b_q: majorities.append(f"White Queenside ({w_q} vs {b_q})")
    elif b_q > w_q: majorities.append(f"Black Queenside ({b_q} vs {w_q})")

    if w_k > b_k: majorities.append(f"White Kingside ({w_k} vs {b_k})")
    elif b_k > w_k: majorities.append(f"Black Kingside ({b_k} vs {w_k})")
    features["pawn_structure"]["pawn_majorities"] = majorities

    # Piece Safety (Static Exchange Evaluation)
    w_safety = _get_piece_safety(board, chess.WHITE)
    b_safety = _get_piece_safety(board, chess.BLACK)

    features["piece_safety"] = {
        "hanging_white": w_safety["hanging"], "hanging_black": b_safety["hanging"],
        "pinned_white": w_safety["pinned"], "pinned_black": b_safety["pinned"],
        "weakly_defended_white": w_safety["weakly_defended"], "weakly_defended_black": b_safety["weakly_defended"]
    }

    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        loose = _get_loose_pieces(board, color)
        if loose:
            features["piece_safety"][f"loose_undefended_{label}"] = loose
        overloaded = _get_overloaded_defenders(board, color)
        if overloaded:
            features["piece_safety"][f"overloaded_defenders_{label}"] = overloaded
        trapped = _get_trapped_pieces(board, color)
        if trapped:
            features["piece_safety"][f"trapped_{label}"] = trapped

    features["king_safety"] = {
        "white": _analyze_king_perimeter(board, chess.WHITE),
        "black": _analyze_king_perimeter(board, chess.BLACK)
    }
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if _is_back_rank_vulnerable(board, color):
            features["king_safety"][label]["back_rank_vulnerable_no_luft"] = True

    # Material
    w_mat = _calc_material_score(board, chess.WHITE)
    b_mat = _calc_material_score(board, chess.BLACK)
    features["material"] = {
        "balance": "Even" if w_mat == b_mat else (f"White +{w_mat - b_mat}" if w_mat > b_mat else f"Black +{b_mat - w_mat}"),
        "white_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2,
        "black_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    }
    if _has_opposite_colored_bishops(board):
        features["material"]["opposite_colored_bishops"] = (
            "yes — single bishops on opposite color complexes (favors the attacker in "
            "middlegames; strong drawish tendency in pure endgames)")

    # Center Control
    center_sqs = [chess.D4, chess.E4, chess.D5, chess.E5]
    w_cnt = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_sqs)
    b_cnt = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_sqs)
    features["central_control"] = {"control_dominance": "White" if w_cnt > b_cnt else ("Black" if b_cnt > w_cnt else "Contested/Equal")}

    # Piece Activity
    features["piece_activity"] = {
        "white_knight_outposts": _find_true_outposts(board, chess.WHITE),
        "black_knight_outposts": _find_true_outposts(board, chess.BLACK),
        "white_rooks_connected": _are_rooks_connected(board, chess.WHITE),
        "black_rooks_connected": _are_rooks_connected(board, chess.BLACK),
        "white_rooks_on_7th": [chess.square_name(sq) for sq in board.pieces(chess.ROOK, chess.WHITE) if chess.square_rank(sq) == 6],
        "black_rooks_on_2nd": [chess.square_name(sq) for sq in board.pieces(chess.ROOK, chess.BLACK) if chess.square_rank(sq) == 1],
        "white_bad_bishops": _get_bad_bishops(board, chess.WHITE),
        "black_bad_bishops": _get_bad_bishops(board, chess.BLACK)
    }
    for color, label, own_half_open in ((chess.WHITE, "white", half_open_w),
                                        (chess.BLACK, "black", half_open_b)):
        on_open, on_half = _get_rooks_on_files(board, color, open_files, own_half_open)
        if on_open:
            features["piece_activity"][f"{label}_rooks_on_open_files"] = on_open
        if on_half:
            features["piece_activity"][f"{label}_rooks_on_half_open_files"] = on_half
    mobility = _get_mobility(board)
    if mobility:
        features["piece_activity"]["mobility"] = mobility

    # Development
    features["development"] = {
        "white_king_status": _king_status(board, chess.WHITE),
        "black_king_status": _king_status(board, chess.BLACK),
        "white_can_castle": board.has_castling_rights(chess.WHITE),
        "black_can_castle": board.has_castling_rights(chess.BLACK),
        "white_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE) if chess.square_rank(sq) == 0]),
        "black_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK) if chess.square_rank(sq) == 7])
    }

    # Forcing moves available to the side to move (calculation scaffold)
    features["forcing_moves_for_side_to_move"] = get_forcing_moves(board)

    if features["game_phase"] == "endgame":
        features["endgame"] = _get_endgame_features(board)

    return features
