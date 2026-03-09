import os
import chess
import chess.engine
import subprocess
from functools import lru_cache
from app.ui import console

class EngineManager:
    def __init__(self, config):
        self.config = config
        self.stockfish = None
        self.maia = None

    def start(self):
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            threads = max(1, min(8, os.cpu_count() or 4))
            self.stockfish.configure({"Threads": threads, "Hash": 512})
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Stockfish not found at {self.config.stockfish_path}.")
            exit(1)

        try:
            rounded_elo = round(self.config.user_elo / 100) * 100
            maia_target = 1100 if rounded_elo <= 1300 else (1500 if rounded_elo <= 1700 else 1900)
            weights_file = os.path.join(self.config.maia_weights_dir, f"maia-{maia_target}.pb.gz")
            
            if os.path.exists(weights_file):
                self.maia = chess.engine.SimpleEngine.popen_uci(self.config.lc0_path, stderr=subprocess.DEVNULL)
                self.maia.configure({"WeightsFile": weights_file})
            else:
                self.maia = None
                console.print(f"[bold yellow]Warning: Maia weights not found at {weights_file}[/bold yellow]")
        except Exception as e:
            self.maia = None
            console.print(f"[bold red]Error initializing Maia:[/bold red] {e}")

    @lru_cache(maxsize=2000)
    def get_cached_eval(self, fen: str, time_limit: float):
        board = chess.Board(fen)
        info = self.stockfish.analyse(board, chess.engine.Limit(time=time_limit))
        return info["score"]

    def eval_move(self, board, move, time_limit, turn_color):
        board.push(move)
        score_obj = self.get_cached_eval(board.fen(), time_limit)
        board.pop()
        return score_obj.pov(turn_color).score(mate_score=100000)
        
    def quit(self):
        if self.stockfish: self.stockfish.quit()
        if self.maia: self.maia.quit()

def extract_board_features(board: chess.Board) -> dict:
    features = {
        "pawn_structure": {},
        "piece_safety": {"hanging_white": [], "hanging_black": [], "pinned_white": [], "pinned_black": []}
    }
    
    files = "abcdefgh"
    
    def get_pawn_data(color):
        counts = {f: 0 for f in files}
        pawns = list(board.pieces(chess.PAWN, color))
        for sq in pawns: counts[files[chess.square_file(sq)]] += 1
            
        islands = []
        current_island = []
        for f in files:
            if counts[f] > 0:
                current_island.append(f)
            else:
                if current_island:
                    islands.append(f"{current_island[0]}-{current_island[-1]}" if len(current_island) > 1 else current_island[0])
                    current_island = []
        if current_island:
            islands.append(f"{current_island[0]}-{current_island[-1]}" if len(current_island) > 1 else current_island[0])
                
        isolated = []
        for sq in pawns:
            f = chess.square_file(sq)
            left = counts[files[f - 1]] > 0 if f > 0 else False
            right = counts[files[f + 1]] > 0 if f < 7 else False
            if not left and not right:
                isolated.append(chess.square_name(sq))
                
        doubled = [f for f in files if counts[f] > 1]
                
        return {
            "total_pawns": len(pawns),
            "island_count": len(islands),
            "island_spans": islands,
            "isolated": isolated,
            "doubled_files": doubled,
            "file_counts": counts
        }

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
        blockers = [osq for osq in board.pieces(chess.PAWN, chess.BLACK) 
                    if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) > r]
        if not blockers: passed_w.append(chess.square_name(sq))
        
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blockers = [osq for osq in board.pieces(chess.PAWN, chess.WHITE) 
                    if abs(chess.square_file(osq) - f) <= 1 and chess.square_rank(osq) < r]
        if not blockers: passed_b.append(chess.square_name(sq))

    def analyze_king_perimeter(color):
        king_sq = board.king(color)
        if king_sq is None:
            return {"king_square": "None", "pawn_shield_squares":[], "perimeter_breaches":[]}
            
        king_name = chess.square_name(king_sq)
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        
        forward_dir = 1 if color == chess.WHITE else -1
        
        # 1. Map the Pawn Shield (Current rank + 2 ranks forward, adjacent files)
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
                            
        # 2. Map Perimeter Breaches (Enemy pieces attacking the King's immediate 8 squares)
        adjacent_squares =[
            chess.square(f, r)
            for f in[kf - 1, kf, kf + 1] if 0 <= f <= 7
            for r in[kr - 1, kr, kr + 1] if 0 <= r <= 7
            if not (f == kf and r == kr)
        ]
        
        enemy_attackers = set()
        for sq in adjacent_squares:
            attackers = board.attackers(not color, sq)
            for atk_sq in attackers:
                enemy_piece = board.piece_at(atk_sq)
                if enemy_piece:
                    enemy_attackers.add(f"{enemy_piece.symbol()} on {chess.square_name(atk_sq)}")

        return {
            "king_square": king_name,
            "pawn_shield_squares": sorted(list(pawn_shield)),
            "enemy_pieces_attacking_perimeter": sorted(list(enemy_attackers))
        }
        
    features["pawn_structure"] = {
        "white_total_pawns": w_data["total_pawns"],
        "black_total_pawns": b_data["total_pawns"],
        "white_pawn_islands_count": w_data["island_count"],
        "black_pawn_islands_count": b_data["island_count"],
        "white_pawn_islands_spans": w_data["island_spans"],
        "black_pawn_islands_spans": b_data["island_spans"],
        "white_doubled_files": w_data["doubled_files"],
        "black_doubled_files": b_data["doubled_files"],
        "white_isolated_pawns": w_data["isolated"],
        "black_isolated_pawns": b_data["isolated"],
        "white_passed_pawns": passed_w,
        "black_passed_pawns": passed_b,
        "open_files": open_files,
        "half_open_for_white": half_open_w,
        "half_open_for_black": half_open_b
    }
    
    for color, hanging_key, pinned_key in [(chess.WHITE, "hanging_white", "pinned_white"), 
                                           (chess.BLACK, "hanging_black", "pinned_black")]:
        opp_color = not color
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color:
                sq_name = chess.square_name(sq)
                if board.is_pinned(color, sq):
                    features["piece_safety"][pinned_key].append(f"{piece.symbol()} on {sq_name}")
                if piece.piece_type != chess.KING:
                    attackers = board.attackers(opp_color, sq)
                    defenders = board.attackers(color, sq)
                    if attackers and not defenders:
                        features["piece_safety"][hanging_key].append(f"{piece.symbol()} on {sq_name}")

    features["king_safety"] = {
        "white": analyze_king_perimeter(chess.WHITE),
        "black": analyze_king_perimeter(chess.BLACK)
    }

    # --- 1. ABSOLUTE MATERIAL COUNT & IMBALANCES ---
    def calculate_material(color):
        return (
            len(board.pieces(chess.PAWN, color)) * 1 +
            len(board.pieces(chess.KNIGHT, color)) * 3 +
            len(board.pieces(chess.BISHOP, color)) * 3 +
            len(board.pieces(chess.ROOK, color)) * 5 +
            len(board.pieces(chess.QUEEN, color)) * 9
        )
    
    white_mat = calculate_material(chess.WHITE)
    black_mat = calculate_material(chess.BLACK)
    
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))

    features["material"] = {
        "white_score": white_mat,
        "black_score": black_mat,
        "balance": "Even" if white_mat == black_mat else (f"White +{white_mat - black_mat}" if white_mat > black_mat else f"Black +{black_mat - white_mat}"),
        "white_has_bishop_pair": white_bishops >= 2,
        "black_has_bishop_pair": black_bishops >= 2
    }

    # --- 2. CENTRAL CONTROL & SPACE ---
    # We measure how many times each side attacks the 4 critical central squares.
    center_squares =[chess.D4, chess.E4, chess.D5, chess.E5]
    white_center_attacks = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_squares)
    black_center_attacks = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_squares)

    features["central_control"] = {
        "white_attacks_on_center": white_center_attacks,
        "black_attacks_on_center": black_center_attacks,
        "control_dominance": "White" if white_center_attacks > black_center_attacks else ("Black" if black_center_attacks > white_center_attacks else "Contested/Equal")
    }
    
    return features