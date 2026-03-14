import chess

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

def _get_piece_safety(board: chess.Board, color: chess.Color) -> dict:
    hanging = []
    pinned = []
    weakly_defended = []
    
    opp_color = not color
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            sq_name = chess.square_name(sq)
            
            if board.is_pinned(color, sq):
                pinned.append(f"{piece.symbol()} on {sq_name}")
            
            if piece.piece_type != chess.KING:
                attackers = list(board.attackers(opp_color, sq))
                defenders = list(board.attackers(color, sq))
                num_atk = len(attackers)
                num_def = len(defenders)
                
                if num_atk > 0 and num_def == 0:
                    hanging.append(f"{piece.symbol()} on {sq_name}")
                elif num_atk > 0:
                    is_weak = False
                    p_val = piece_values.get(piece.piece_type, 0)
                    lowest_atk_val = min(piece_values.get(board.piece_at(a).piece_type, 0) for a in attackers)
                    lowest_def_val = min(piece_values.get(board.piece_at(d).piece_type, 0) for d in defenders) if defenders else 0
                    
                    if num_atk > num_def: is_weak = True
                    elif lowest_atk_val < p_val: is_weak = True
                    elif lowest_def_val > lowest_atk_val: is_weak = True
                    elif num_atk == num_def and num_atk >= 2 and lowest_atk_val <= lowest_def_val: is_weak = True

                    if is_weak:
                        weakly_defended.append(f"{piece.symbol()} on {sq_name} (Atk: {num_atk}, Def: {num_def})")
                        
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

def extract_board_features(board: chess.Board) -> dict:
    """Rigorous extraction of structural features based on Soviet Chess Standards."""
    features = {}
    files = "abcdefgh"
    
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

    # Piece Safety
    w_safety = _get_piece_safety(board, chess.WHITE)
    b_safety = _get_piece_safety(board, chess.BLACK)
    
    features["piece_safety"] = {
        "hanging_white": w_safety["hanging"], "hanging_black": b_safety["hanging"],
        "pinned_white": w_safety["pinned"], "pinned_black": b_safety["pinned"],
        "weakly_defended_white": w_safety["weakly_defended"], "weakly_defended_black": b_safety["weakly_defended"]
    }

    features["king_safety"] = {
        "white": _analyze_king_perimeter(board, chess.WHITE),
        "black": _analyze_king_perimeter(board, chess.BLACK)
    }

    # Material
    w_mat = _calc_material_score(board, chess.WHITE)
    b_mat = _calc_material_score(board, chess.BLACK)
    features["material"] = {
        "balance": "Even" if w_mat == b_mat else (f"White +{w_mat - b_mat}" if w_mat > b_mat else f"Black +{b_mat - w_mat}"),
        "white_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2,
        "black_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    }

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

    # Development
    features["development"] = {
        "white_can_castle": board.has_castling_rights(chess.WHITE),
        "black_can_castle": board.has_castling_rights(chess.BLACK),
        "white_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE) if chess.square_rank(sq) == 0]),
        "black_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK) if chess.square_rank(sq) == 7])
    }
    
    return features