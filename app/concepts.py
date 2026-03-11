import chess

def extract_board_features(board: chess.Board) -> dict:
    """Rigorous extraction of structural features based on Soviet Chess Standards."""
    features = {
        "pawn_structure": {},
        "piece_safety": {"hanging_white":[], "hanging_black": [], "pinned_white": [], "pinned_black":[]}
    }
    
    files = "abcdefgh"
    
    # --- 1. PAWN STRUCTURE ---
    def get_pawn_data(color):
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

    w_data = get_pawn_data(chess.WHITE)
    b_data = get_pawn_data(chess.BLACK)
    
    open_files, half_open_w, half_open_b = [], [], []
    for f in files:
        w, b = w_data["file_counts"][f], b_data["file_counts"][f]
        if w == 0 and b == 0: open_files.append(f)
        elif w == 0 and b > 0: half_open_w.append(f)
        elif b == 0 and w > 0: half_open_b.append(f)
        
    passed_w, passed_b = [], []
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if not [osq for osq in board.pieces(chess.PAWN, chess.BLACK) if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) > r]:
            passed_w.append(chess.square_name(sq))
            
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if not [osq for osq in board.pieces(chess.PAWN, chess.WHITE) if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) < r]:
            passed_b.append(chess.square_name(sq))

    features["pawn_structure"] = {
        "white_passed_pawns": passed_w, "black_passed_pawns": passed_b,
        "white_isolated_pawns": w_data["isolated"], "black_isolated_pawns": b_data["isolated"],
        "white_doubled_files": w_data["doubled_files"], "black_doubled_files": b_data["doubled_files"],
        "open_files": open_files, "half_open_for_white": half_open_w, "half_open_for_black": half_open_b
    }

    # --- 2. PIECE SAFETY (Hanging, Pinned & Weakly Defended) ---
    features["piece_safety"] = {
        "hanging_white": [], "hanging_black": [], 
        "pinned_white": [], "pinned_black": [],
        "weakly_defended_white": [], "weakly_defended_black": []
    }
    
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

    for color, hanging_key, pinned_key, weak_key in [
        (chess.WHITE, "hanging_white", "pinned_white", "weakly_defended_white"), 
        (chess.BLACK, "hanging_black", "pinned_black", "weakly_defended_black")
    ]:
        opp_color = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color:
                sq_name = chess.square_name(sq)
                
                # Check Pins
                if board.is_pinned(color, sq):
                    features["piece_safety"][pinned_key].append(f"{piece.symbol()} on {sq_name}")
                
                # Check Hanging & Weakly Defended
                if piece.piece_type != chess.KING:
                    attackers = list(board.attackers(opp_color, sq))
                    defenders = list(board.attackers(color, sq))
                    num_atk = len(attackers)
                    num_def = len(defenders)
                    
                    if num_atk > 0 and num_def == 0:
                        features["piece_safety"][hanging_key].append(f"{piece.symbol()} on {sq_name}")
                    elif num_atk > num_def:
                        # Condition A: Outnumbered (Your logic)
                        features["piece_safety"][weak_key].append(f"{piece.symbol()} on {sq_name} (Attackers: {num_atk}, Defenders: {num_def})")
                    elif num_atk > 0:
                        # Condition B: Attacked by a lower-value piece (My logic)
                        p_val = piece_values.get(piece.piece_type, 0)
                        lowest_atk_val = min(piece_values.get(board.piece_at(a).piece_type, 0) for a in attackers)
                        if lowest_atk_val < p_val:
                            features["piece_safety"][weak_key].append(f"{piece.symbol()} on {sq_name} (Attacked by lower-value piece)")

    # --- 3. KING SAFETY PERIMETER ---
    def analyze_king_perimeter(color):
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

    features["king_safety"] = {"white": analyze_king_perimeter(chess.WHITE), "black": analyze_king_perimeter(chess.BLACK)}

    # --- 4. MATERIAL & CENTER CONTROL ---
    def calc_mat(color):
        return sum([len(board.pieces(p, color)) * v for p, v in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)]])
    
    w_mat, b_mat = calc_mat(chess.WHITE), calc_mat(chess.BLACK)
    features["material"] = {
        "balance": "Even" if w_mat == b_mat else (f"White +{w_mat - b_mat}" if w_mat > b_mat else f"Black +{b_mat - w_mat}"),
        "white_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2,
        "black_has_bishop_pair": len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    }

    center_sqs = [chess.D4, chess.E4, chess.D5, chess.E5]
    w_cnt = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_sqs)
    b_cnt = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_sqs)
    features["central_control"] = {"control_dominance": "White" if w_cnt > b_cnt else ("Black" if b_cnt > w_cnt else "Contested/Equal")}

    # --- 5. TRUE KNIGHT OUTPOSTS (The Soviet Standard) ---
    def find_true_outposts(color):
        outposts = []
        valid_ranks = [3, 4, 5] if color == chess.WHITE else [2, 3, 4]
        forward_step = 1 if color == chess.WHITE else -1
        enemy_color = not color

        for sq in board.pieces(chess.KNIGHT, color):
            r, f = chess.square_rank(sq), chess.square_file(sq)
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

    features["piece_activity"] = {"white_knight_outposts": find_true_outposts(chess.WHITE), "black_knight_outposts": find_true_outposts(chess.BLACK)}
    

    # --- 6. POSITIONAL: BAD BISHOPS ---
    def get_bad_bishops(color):
        bad_bishops =[]
        bishops = board.pieces(chess.BISHOP, color)
        pawns = board.pieces(chess.PAWN, color)
        
        forward_dir = 1 if color == chess.WHITE else -1
        
        for sq in bishops:
            is_light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 != 0
            blocked_central_pawns = 0
            
            for p_sq in pawns:
                f, r = chess.square_file(p_sq), chess.square_rank(p_sq)
                # Only care about central/semi-central pawns (c, d, e, f files)
                if 2 <= f <= 5:
                    p_is_light = (f + r) % 2 != 0
                    if p_is_light == is_light:
                        # Is the pawn blocked from moving forward?
                        block_sq = chess.square(f, r + forward_dir)
                        if 0 <= r + forward_dir <= 7 and board.piece_at(block_sq):
                            blocked_central_pawns += 1
                            
            # A bishop is structurally bad if 2 or more central pawns are blocked on its color
            if blocked_central_pawns >= 2:
                bad_bishops.append(chess.square_name(sq))
        return bad_bishops

    # --- 7. HEAVY PIECE COORDINATION ---
    def are_rooks_connected(color):
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) == 2:
            # Rooks are connected if they attack each other
            return rooks[1] in board.attacks(rooks[0])
        return False

    features["piece_activity"]["white_rooks_connected"] = are_rooks_connected(chess.WHITE)
    features["piece_activity"]["black_rooks_connected"] = are_rooks_connected(chess.BLACK)
    
    features["piece_activity"]["white_rooks_on_7th"] =[chess.square_name(sq) for sq in board.pieces(chess.ROOK, chess.WHITE) if chess.square_rank(sq) == 6]
    features["piece_activity"]["black_rooks_on_2nd"] =[chess.square_name(sq) for sq in board.pieces(chess.ROOK, chess.BLACK) if chess.square_rank(sq) == 1]

    features["piece_activity"]["white_bad_bishops"] = get_bad_bishops(chess.WHITE)
    features["piece_activity"]["black_bad_bishops"] = get_bad_bishops(chess.BLACK)

    # --- 8. PAWN MAJORITIES ---
    w_q = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) <= 2)
    b_q = sum(1 for sq in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(sq) <= 2)
    w_k = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) >= 5)
    b_k = sum(1 for sq in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(sq) >= 5)
    
    majorities =[]
    if w_q > b_q: majorities.append(f"White Queenside ({w_q} vs {b_q})")
    elif b_q > w_q: majorities.append(f"Black Queenside ({b_q} vs {w_q})")
    
    if w_k > b_k: majorities.append(f"White Kingside ({w_k} vs {b_k})")
    elif b_k > w_k: majorities.append(f"Black Kingside ({b_k} vs {w_k})")
    
    features["pawn_structure"]["pawn_majorities"] = majorities

    # --- 9. DEVELOPMENT PHASE ---
    features["development"] = {
        # 🐛 BUG FIX: Use has_castling_rights() instead of bitwise &
        "white_can_castle": board.has_castling_rights(chess.WHITE),
        "black_can_castle": board.has_castling_rights(chess.BLACK),
        "white_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE) if chess.square_rank(sq) == 0]),
        "black_undeveloped_minors": len([sq for sq in board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK) if chess.square_rank(sq) == 7])
    }
    
    return features