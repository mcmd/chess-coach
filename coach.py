import io
import os
import json
import argparse
import shutil
import requests
import subprocess
import chess
import chess.pgn
import chess.engine
import re
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from dotenv import load_dotenv
from prompt_toolkit import prompt, HTML
from prompt_toolkit.history import FileHistory
from functools import lru_cache
from dataclasses import dataclass, field

load_dotenv()

console = Console()

# ======================================================================
# 1. SETUP, CONFIGURATION & ELO PARAMETERIZATION
# ======================================================================

parser = argparse.ArgumentParser(
    description="Gemini Chess Coach: An AI-powered, tool-augmented chess coach.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog="""
Examples:
  python coach.py                  # Runs the coach using the default root configuration and memory.
  python coach.py --profile=mcmd   # Loads (or creates) an isolated configuration and memory for 'mcmd'.
"""
)
parser.add_argument("--wipe_memory", action="store_true", help="Clear the chat memory (chess_coach_memory.json) before starting the coach.")
parser.add_argument(
    "--profile", 
    type=str, 
    help="Name of the profile to load.\n"
         "This isolates your Elo settings, chat memory, and command history.", 
    default=None
)
args = parser.parse_args()

ROOT_CONFIG_FILE = "config.json"
CONFIG_FILE = ROOT_CONFIG_FILE
HISTORY_FILE = "chess_coach_memory.json"
DOSSIER_FILE = "player_dossier.json" # NEW
CLI_HISTORY_FILE = "chess_cli_history.txt"

if not os.path.exists(ROOT_CONFIG_FILE):
    console.print(f"[bold red]Error:[/bold red] {ROOT_CONFIG_FILE} not found in the root directory. Please create it first.")
    exit(1)

with open(ROOT_CONFIG_FILE, "r") as f:
    root_config = json.load(f)

BASE_PROFILES_DIR = root_config.get("profiles_dir", "./profiles")

if args.profile:
    profile_dir = os.path.join(BASE_PROFILES_DIR, args.profile)
    os.makedirs(profile_dir, exist_ok=True)
    
    CONFIG_FILE = os.path.join(profile_dir, "config.json")
    HISTORY_FILE = os.path.join(profile_dir, "chess_coach_memory.json")
    DOSSIER_FILE = os.path.join(profile_dir, "player_dossier.json") # NEW
    CLI_HISTORY_FILE = os.path.join(profile_dir, "chess_cli_history.txt")
    
    if not os.path.exists(CONFIG_FILE):
        console.print(f"[yellow]Creating new config for profile '{args.profile}'.[/yellow]")
        shutil.copy(ROOT_CONFIG_FILE, CONFIG_FILE)

if args.wipe_memory and os.path.exists(HISTORY_FILE):
    os.remove(HISTORY_FILE)
    console.print(f"[bold red]Memory wiped:[/bold red] {HISTORY_FILE} has been cleared.")

with open(CONFIG_FILE, "r") as f:
    app_config = json.load(f)

USER_ELO = app_config.get("user_elo", 1100)
USER_LANGUAGE = app_config.get("language", "English") # NEW: Defaults to English if missing
STOCKFISH_PATH = app_config.get("stockfish_path")
LC0_PATH = app_config.get("lc0_path")
MAIA_DIR = app_config.get("maia_weights_dir", "./maia_weights")
GEMINI_MODEL = app_config.get("gemini_model", "gemini-3.1-pro-preview")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set.")
    exit(1)

client = genai.Client(api_key=api_key)

# ======================================================================
# 2. APPLICATION STATE & TIMELINE MANAGER (PHASE 2 UPGRADE)
# ======================================================================

@dataclass
class SessionState:
    """The Single Source of Truth for the local application."""
    board: chess.Board = field(default_factory=chess.Board)
    loaded_moves: list[chess.Move] = field(default_factory=list)
    move_index: int = 0
    sparring_mode: bool = False
    diagnostics: list[dict] = field(default_factory=list)
    board_visuals: dict = field(default_factory=dict)
    board_counter: int = 0

class TimelineManager:
    """Safely manages all physical board mutations and branching invariants."""
    def __init__(self, state: SessionState):
        self.state = state

    def reset(self):
        self.state.board.reset()
        self.state.loaded_moves = []
        self.state.move_index = 0
        self.state.diagnostics.clear()

    def load_game(self, game: chess.pgn.Game):
        if game and len(list(game.mainline_moves())) > 0:
            self.state.board = game.end().board()
            self.state.loaded_moves = list(game.mainline_moves())
            self.state.move_index = len(self.state.loaded_moves)

    def play_san(self, move_san: str) -> str:
        """Plays a move, handling timeline truncation if the user branches."""
        move = self.state.board.parse_san(move_san)
        if self.state.move_index < len(self.state.loaded_moves):
            self.state.loaded_moves = self.state.loaded_moves[:self.state.move_index]
        self.state.board.push(move)
        self.state.loaded_moves.append(move)
        self.state.move_index += 1
        return self.state.board.san(move)

    def step_forward(self) -> str:
        if self.state.move_index < len(self.state.loaded_moves):
            next_move = self.state.loaded_moves[self.state.move_index]
            move_san = self.state.board.san(next_move)
            self.state.board.push(next_move)
            self.state.move_index += 1
            return move_san
        raise IndexError("End of timeline.")

    def step_backward(self):
        if self.state.move_index > 0:
            self.state.board.pop()
            self.state.move_index -= 1
        else:
            raise IndexError("No moves to undo.")

# Instantiate the Single Source of Truth
app_state = SessionState()
timeline = TimelineManager(app_state)


# ======================================================================
# ENGINE INITIALIZATION (Now with optimized Threads and Hash!)
# ======================================================================
try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # Use max available threads (capped at 8) and 512MB RAM for vastly superior depth-per-second
    threads = max(1, min(8, os.cpu_count() or 4))
    stockfish_engine.configure({"Threads": threads, "Hash": 512})
except FileNotFoundError:
    console.print(f"[bold red]Error:[/bold red] Stockfish not found at {STOCKFISH_PATH}.")
    exit(1)

try:
    rounded_elo = round(USER_ELO / 100) * 100
    maia_target = 1100 if rounded_elo <= 1300 else (1500 if rounded_elo <= 1700 else 1900)
    weights_file = os.path.join(MAIA_DIR, f"maia-{maia_target}.pb.gz")
    
    if os.path.exists(weights_file):
        maia_engine = chess.engine.SimpleEngine.popen_uci(LC0_PATH, stderr=subprocess.DEVNULL)
        maia_engine.configure({"WeightsFile": weights_file})
    else:
        maia_engine = None
        console.print(f"[bold yellow]Warning: Maia weights not found at {weights_file}[/bold yellow]")
except Exception as e:
    maia_engine = None
    console.print(f"[bold red]Error initializing Maia:[/bold red] {e}")


# ======================================================================
# 2. THE TOOLS & EXPERT-REFINED BRILLIANCY SCANNER
# ======================================================================

@lru_cache(maxsize=2000)
def get_cached_eval(fen: str, time_limit: float):
    """Bounded LRU cache for engine evaluations to prevent memory leaks."""
    board = chess.Board(fen)
    info = stockfish_engine.analyse(board, chess.engine.Limit(time=time_limit))
    return info["score"]

def eval_move(board, move, time_limit, turn_color):
    """Helper for evaluating non-PV moves safely."""
    board.push(move)
    score_obj = get_cached_eval(board.fen(), time_limit)
    board.pop()
    return score_obj.pov(turn_color).score(mate_score=100000)

def extract_board_features(board: chess.Board) -> dict:
    """Fast, pure-python extraction of structural and safety features."""
    features = {
        "pawn_structure": {},
        "piece_safety": {"hanging_white": [], "hanging_black": [], "pinned_white": [], "pinned_black": []}
    }
    
    files = "abcdefgh"
    
    def get_pawn_data(color):
        counts = {f: 0 for f in files}
        pawns = list(board.pieces(chess.PAWN, color))
        for sq in pawns: counts[files[chess.square_file(sq)]] += 1
            
        # 1. Map exact island spans (e.g. ["a-d", "f-h"]) so the AI can't guess!
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
    
    # 2. Find Open and Half-Open Files
    open_files, half_open_w, half_open_b = [], [], []
    for f in files:
        w, b = w_data["file_counts"][f], b_data["file_counts"][f]
        if w == 0 and b == 0: open_files.append(f)
        elif w == 0 and b > 0: half_open_w.append(f)
        elif b == 0 and w > 0: half_open_b.append(f)
        
    # 3. Find Passed Pawns (NEW!)
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
    
    # --- Piece Safety ---
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

    return features

def execute_coach_move(move_san: str) -> dict:
    """Plays a chess move. ONLY allowed if SPARRING_MODE is True."""
    if not app_state.sparring_mode:
        return {"error": "Sparring mode is DISABLED. Tell the user to type /spar to enable it."}
    
    try:
        timeline.play_san(move_san)
        return {"success": True, "new_fen": app_state.board.fen(), "message": f"Successfully played {move_san}."}
    except ValueError as e:
        # THE LOGIC TUTOR PARADIGM
        legal_moves_str = ", ".join([app_state.board.san(m) for m in app_state.board.legal_moves])
        return {
            "error": f"Logic Tutor Intervention: Move '{move_san}' is ILLEGAL in this position.",
            "causal_trace": str(e),
            "pedagogical_hint": "The piece is either pinned, blocked, or your King is in check.",
            "required_action": f"To repair your plan, you MUST choose from these mathematically verified valid moves: [{legal_moves_str}]"
        }

def submit_diagnosis(opponent_last_move_delta: str, primary_task: str, error_diagnosis: str, 
                     weakest_reasoning_claim: str, lowest_confidence_score: float, preventative_habit: str) -> dict:
    """Submits a structured diagnosis of the user's mistake to the backend state."""
    
    # Claim-Level Behavioral Calibration (Anti-Hallucination)
    if float(lowest_confidence_score) < 0.90:
        error_diagnosis = "none" # Prevent logging low-confidence guesses to the Dossier
        
    diagnosis = {
        "opponent_last_move_delta": opponent_last_move_delta[:150], 
        "primary_task": primary_task,
        "error_diagnosis": error_diagnosis,
        "weakest_reasoning_claim": weakest_reasoning_claim[:150], 
        "lowest_confidence_score": float(lowest_confidence_score), 
        "preventative_habit": preventative_habit[:200] 
    }
    app_state.diagnostics.append(diagnosis)
    
    # Render the UI Panel immediately!
    console.print()
    console.print(Panel(
        f"[dim white]1. ∆ Opponent Move:[/dim white] [dim yellow]{diagnosis['opponent_last_move_delta']}[/dim yellow]\n"
        f"[dim white]2. Primary Task:[/dim white]    [dim cyan]{diagnosis['primary_task']}[/dim cyan]\n"
        f"[dim white]3. Error Diagnosis:[/dim white] [dim red]{diagnosis['error_diagnosis']}[/dim red]\n"
        f"[dim white]4. Weakest Claim:[/dim white]   [dim magenta]\"{diagnosis['weakest_reasoning_claim']}\" (Conf: {diagnosis['lowest_confidence_score']:.2f})[/dim magenta]\n"
        f"[dim white]5. Required Habit:[/dim white]  [dim green]{diagnosis['preventative_habit']}[/dim green]",
        title="[dim italic]Coach's Diagnostic Taxonomy[/dim italic]",
        border_style="dim blue",
        expand=False
    ))
    return {"status": "success", "message": "Diagnosis recorded in backend. Proceed with your final text response."}

def _safe_get_json(url: str, params: dict = None) -> dict:
    """Robust network request helper to prevent terminal hanging and encoding errors."""
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Network request failed: {e}"}

def scan_game_for_critical_moments(user_text: str) -> tuple[bool, str]:
    """
    Two-Pass Engine Scan:
    Pass A: Fast scan every ply (0.25s).
    Pass B: Deep confirm (1.0s) only if a brilliancy candidate is detected to prevent false tags.
    Returns: (was_scanned: bool, processed_pgn_text: str)
    """
    if "1." not in user_text:
        return False, user_text 
        
    try:
        pgn_io = io.StringIO(user_text)
        game = chess.pgn.read_game(pgn_io)
        if not game or len(list(game.mainline_moves())) < 3:
            return False, user_text
            
        console.print("\n[dim yellow]Running 2-Pass Brilliancy Scan...[/dim yellow]")
        
        annotated_games = []
        with console.status("[bold yellow]Stockfish & Maia are analyzing critical paths (Takes ~1 min)...[/bold yellow]", spinner="dots"):
            pgn_io = io.StringIO(user_text)
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None: break
                
                board = game.board()
                for node in game.mainline():
                    played_move = node.move
                    if played_move is None: continue # Safety check
                    
                    is_only_move = False
                    dodged_trap = False
                    maia_san = ""
                    
                    if stockfish_engine:
                        turn_color = board.turn
                        
                        # --- PASS A: FAST SCAN (0.25s) ---
                        info_a = stockfish_engine.analyse(board, chess.engine.Limit(time=0.25), multipv=3)
                        
                        if info_a and "pv" in info_a[0]:
                            best_move_a = info_a[0]["pv"][0]
                            best_score_a = info_a[0]["score"].pov(turn_color).score(mate_score=100000)
                            second_score_a = info_a[1]["score"].pov(turn_color).score(mate_score=100000) if len(info_a) > 1 and "pv" in info_a[1] else None
                            
                            candidate_only_move = False
                            candidate_trap = False
                            maia_move = None
                            
                            # Candidate check 1: Only Move
                            if played_move == best_move_a and second_score_a is not None:
                                if info_a[0]["score"].pov(turn_color).is_mate() and not info_a[1]["score"].pov(turn_color).is_mate() and info_a[0]["score"].pov(turn_color).mate() > 0:
                                    candidate_only_move = True
                                elif (best_score_a - second_score_a) >= 150:
                                    candidate_only_move = True
                                    
                            # Candidate check 2: Trap
                            if maia_engine:
                                maia_res = maia_engine.play(board, chess.engine.Limit(nodes=1))
                                maia_move = maia_res.move if maia_res else None
                                if maia_move and maia_move != played_move:
                                    candidate_trap = True
                            
                            # --- PASS B: DEEP CONFIRMATION (1.0s) ---
                            if candidate_only_move or candidate_trap:
                                info_b = stockfish_engine.analyse(board, chess.engine.Limit(time=1.0), multipv=3)
                                
                                if info_b and "pv" in info_b[0]:
                                    best_move_b = info_b[0]["pv"][0]
                                    best_score_b = info_b[0]["score"].pov(turn_color).score(mate_score=100000)
                                    second_score_b = info_b[1]["score"].pov(turn_color).score(mate_score=100000) if len(info_b) > 1 and "pv" in info_b[1] else None
                                    
                                    # 1. Confirm Only Move (Strict >= 200cp gap)
                                    if candidate_only_move and played_move == best_move_b and second_score_b is not None:
                                        if info_b[0]["score"].pov(turn_color).is_mate() and not info_b[1]["score"].pov(turn_color).is_mate() and info_b[0]["score"].pov(turn_color).mate() > 0:
                                            is_only_move = True
                                        elif (best_score_b - second_score_b) >= 200:
                                            is_only_move = True
                                                    
                                    # 2. Confirm Dodged Trap
                                    if candidate_trap and maia_move and maia_move != played_move:
                                        played_score = next((i["score"].pov(turn_color).score(mate_score=100000) for i in info_b if i.get("pv") and i["pv"][0] == played_move), None)
                                        if played_score is None: played_score = eval_move(board, played_move, 0.4, turn_color)
                                            
                                        maia_score = next((i["score"].pov(turn_color).score(mate_score=100000) for i in info_b if i.get("pv") and i["pv"][0] == maia_move), None)
                                        if maia_score is None: maia_score = eval_move(board, maia_move, 0.4, turn_color)
                                            
                                        # Thresholds: Played move within 70cp of best. Maia blunder >= 200cp worse.
                                        if played_score is not None and maia_score is not None and best_score_b is not None:
                                            if (best_score_b - played_score) <= 70 and (played_score - maia_score) >= 200:
                                                dodged_trap = True
                                                maia_san = board.san(maia_move)
                    
                    comment = node.comment if node.comment else ""
                    
                    # 1. FAANG FEN INVARIANT: Grab FEN BEFORE pushing the move!
                    fen_before = board.fen()
                    
                    board.push(played_move)
                    
                    # Apply tags
                    tag_string = ""
                    if is_only_move: tag_string += " [⭐ ONLY MOVE FOUND!]"
                    if dodged_trap: tag_string += f" [🛡️ DODGED HUMAN TRAP! Maia predicted {maia_san}]"
                            
                    # Inject basic evaluation if missing (Scoped strictly to 'comment')
                    if "[%eval" not in comment and stockfish_engine:
                        score_after = get_cached_eval(board.fen(), 0.25).white()
                        eval_str = f"#{score_after.mate()}" if score_after.is_mate() else f"{score_after.score(mate_score=100000) / 100.0:+.2f}"
                        tag_string += f" [%eval {eval_str}]"

                    # 3. ALWAYS inject the pre-move FEN so the AI never guesses
                    if "[%fen" not in comment:
                        tag_string += f' [%fen "{fen_before}"]'

                    if tag_string: node.comment = (comment + tag_string).strip()
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
                annotated_games.append(game.accept(exporter))
                
        if annotated_games:
            console.print("[green]Scan complete![/green]")
            return True, "\n\n".join(annotated_games)
            
    except Exception as e:
        console.print(f"[dim red]Game scan failed: {e}[/dim red]")
        
    return False, user_text

def analyze_position(fen: str, moves_to_apply: list[str] = None, depth: int = 20, pov: str = "white") -> dict:
    """Evaluates a position safely with input boundaries and Long-Round depth."""
    try:
        # SECURITY BOUNDARY: Cap depth to prevent infinite hangs (Default pushed to 20)
        safe_depth = max(1, min(int(depth), 25)) 
        
        board = chess.Board(fen)
        if moves_to_apply:
            for move_str in moves_to_apply[:10]: # Cap hypothetical moves
                board.push_san(move_str)
                
        infos = stockfish_engine.analyse(board, chess.engine.Limit(depth=safe_depth), multipv=2)
        
        top_lines = []
        for i, info in enumerate(infos):
            score = info["score"].white()
            eval_str = f"#{score.mate()}" if score.is_mate() else f"{score.score(mate_score=100000) / 100.0:+.2f}"
            pv_moves = []
            temp_board = board.copy()
            for move in info.get("pv", [])[:10]:  
                pv_moves.append(temp_board.san(move))
                temp_board.push(move)
            top_lines.append({"rank": i + 1, "evaluation": eval_str, "expected_engine_line": " ".join(pv_moves)})

        raw_board = board.unicode(empty_square="·")
        lines = raw_board.split('\n')
        labeled_board = []
        if pov.lower() == "black":
            lines.reverse() 
            for i, line in enumerate(lines):
                reversed_chars = " ".join(line.split(" ")[::-1])
                labeled_board.append(f"{i+1} │ {reversed_chars} │")
            labeled_board.append("  └" + "─" * 17 + "┘")
            labeled_board.append("    h g f e d c b a")
        else:
            for i, line in enumerate(lines):
                labeled_board.append(f"{8-i} │ {line} │")
            labeled_board.append("  └" + "─" * 17 + "┘")
            labeled_board.append("    a b c d e f g h")

        visual = "```text\n" + "\n".join(labeled_board) + "\n```"

        # MEMORY BOUNDARY
        if len(app_state.board_visuals) > 50:
            app_state.board_visuals.pop(next(iter(app_state.board_visuals)))
            
        app_state.board_counter += 1
        token = f"[[BOARD_{app_state.board_counter}]]"
        app_state.board_visuals[token] = visual

        # MATHEMATICAL GROUNDING: Inject Legal Moves to prevent Pointer Misbinding
        legal_moves_str = ", ".join([board.san(m) for m in board.legal_moves])

        return {
            "top_stockfish_lines": top_lines, 
            "board_token_to_insert": token,
            "mathematically_legal_moves": legal_moves_str,
            "positional_and_safety_features": extract_board_features(board)
        }
    except Exception as e: return {"error": str(e)}

def predict_human_move(fen: str) -> dict:
    if not maia_engine: return {"error": "Maia engine not loaded."}
    try:
        board = chess.Board(fen)
        result = maia_engine.play(board, chess.engine.Limit(nodes=1))
        if not result.move: return {"predicted_human_move": "None"}
            
        maia_move_san = board.san(result.move)
        board.push(result.move)
        
        info = stockfish_engine.analyse(board, chess.engine.Limit(depth=20))
        score = info["score"].white()
        eval_str = f"#{score.mate()}" if score.is_mate() else f"{score.score(mate_score=100000) / 100.0:+.2f}"
        
        refutation_moves = []
        temp_board = board.copy()
        for move in info.get("pv", [])[:10]: 
            refutation_moves.append(temp_board.san(move))
            temp_board.push(move)
            
        return {
            "predicted_human_move": maia_move_san,
            "resulting_evaluation_if_played": eval_str,
            "stockfish_refutation_line": " ".join(refutation_moves)
        }
    except Exception as e: return {"error": str(e)}

def get_opening_repertoire(fen: str) -> dict:
    try:
        master_res = _safe_get_json("https://explorer.lichess.ovh/masters", {"fen": fen})
        if "error" in master_res: return master_res
        
        opening_name = master_res.get("opening", {}).get("name")
        eco_code = master_res.get("opening", {}).get("eco")
        
        master_moves = []
        for move in master_res.get("moves", [])[:4]:
            total = move["white"] + move["draws"] + move["black"]
            if total > 0:
                master_moves.append({
                    "move": move["san"], "white_win_pct": round((move["white"] / total) * 100, 1),
                    "draw_pct": round((move["draws"] / total) * 100, 1), "black_win_pct": round((move["black"] / total) * 100, 1)
                })

        brackets = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]
        valid_brackets = [str(b) for b in brackets if abs(b - USER_ELO) <= 250]
        ratings_param = ",".join(valid_brackets) if valid_brackets else "1000,1200"
        
        amateur_res = _safe_get_json("https://explorer.lichess.ovh/lichess", {
            "variant": "standard", "speeds": "blitz,rapid", "ratings": ratings_param, "fen": fen
        })
        if "error" in amateur_res: return amateur_res
        
        amateur_moves = []
        for move in amateur_res.get("moves", [])[:4]:
            total = move["white"] + move["draws"] + move["black"]
            if total > 0:
                amateur_moves.append({
                    "move": move["san"], "total_games": total, "white_win_pct": round((move["white"] / total) * 100, 1),
                    "black_win_pct": round((move["black"] / total) * 100, 1)
                })
            
        return {
            "theoretical_opening_name": opening_name, "eco_code": eco_code,
            "top_master_moves": master_moves, "typical_amateur_moves": amateur_moves, "amateur_brackets_queried": ratings_param
        }
    except Exception as e: return {"error": str(e)}

def fetch_lichess_game_by_id(game_id: str) -> dict:
    try:
        url = f"https://lichess.org/game/export/{game_id}"
        params = {"evals": "true", "clocks": "true", "accuracy": "true", "opening": "true", "literate": "true", "tags": "true"}
        
        r = requests.get(url, params=params, headers={"Accept": "application/x-chess-pgn"}, timeout=15)
        if r.status_code == 200:
            console.print("[dim yellow]↳ Running Brilliancy Scanner on downloaded Lichess game...[/dim yellow]")
            was_scanned, scanned_pgn = scan_game_for_critical_moments(r.text)
            
            # --- SYNC BACKEND STATE ---
            try:
                parsed_game = chess.pgn.read_game(io.StringIO(scanned_pgn))
                timeline.load_game(parsed_game)
            except Exception: pass
            # --------------------------
            
            if was_scanned:
                sys_msg = "\n\n[System Info: I have mathematically scanned the PGN above and added Brilliancy tags. Please use them for your analysis.]"
                return {"pgn": scanned_pgn + sys_msg}
            return {"pgn": scanned_pgn}
        elif r.status_code == 404: return {"error": f"Game with ID '{game_id}' not found."}
        else: return {"error": f"Lichess API returned status {r.status_code}."}
    except Exception as e: return {"error": str(e)}

def fetch_chessgames_com_game(game_id: str) -> dict:
    try:
        url = f"https://www.chessgames.com/njs/api/game/viewPGN/{game_id}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/plain, application/x-chess-pgn, */*"}
        
        r = requests.get(url, params={"raw": "true"}, headers=headers, timeout=15)
        if r.status_code == 200:
            pgn_text = r.text.strip()
            if not pgn_text or not pgn_text.startswith("[Event"): return {"error": "Received malformed data."}
            
            console.print("[dim yellow]↳ Running Brilliancy Scanner on downloaded Master game...[/dim yellow]")
            was_scanned, scanned_pgn = scan_game_for_critical_moments(pgn_text)
            
            # --- SYNC BACKEND STATE ---
            try:
                parsed_game = chess.pgn.read_game(io.StringIO(scanned_pgn))
                timeline.load_game(parsed_game)
            except Exception: pass
            # --------------------------
            
            if was_scanned:
                sys_msg = "\n\n[System Info: I have mathematically scanned the PGN above and added Brilliancy tags. Please use them for your analysis.]"
                return {"pgn": scanned_pgn + sys_msg}
            return {"pgn": scanned_pgn}
        elif r.status_code == 404: return {"error": f"Game with ID '{game_id}' not found."}
        else: return {"error": f"Chessgames API returned status {r.status_code}."}
    except Exception as e: return {"error": str(e)}

def query_endgame_tablebase(fen: str) -> dict:
    try:
        res = _safe_get_json("https://tablebase.lichess.ovh/standard", {"fen": fen})
        if "error" in res: return res
        return {"status": res.get("category"), "dtm": res.get("dtm"), "best_move": res.get("moves", [{}])[0].get("san", "None")}
    except Exception as e: return {"error": str(e)}

# ======================================================================
# 3. GEMINI AGENT & MEMORY SETUP
# ======================================================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                data = json.load(f)
                contents = []
                for item in data:
                    parts =[types.Part.from_text(text=p["text"]) for p in item["parts"]]
                    contents.append(types.Content(role=item["role"], parts=parts))
                return contents
            except json.JSONDecodeError: return []
    return []

def save_history(chat_session):
    serializable =[]
    for message in chat_session.get_history():
        parts =[{"text": p.text} for p in message.parts if getattr(p, 'text', None)]
        if parts: serializable.append({"role": message.role, "parts": parts})
    with open(HISTORY_FILE, "w") as f: json.dump(serializable, f)

def create_chat_session():
    """Builds a fresh chat session by hot-reloading the latest Dossier from disk."""
    dossier_context = "No historical data available yet. This is your first session."
    if os.path.exists(DOSSIER_FILE):
        try:
            with open(DOSSIER_FILE, "r") as f:
                d = json.load(f)
                if d.get("total_blunders_analyzed", 0) > 0:
                    sorted_errors = sorted(d["error_frequencies"].items(), key=lambda item: item[1], reverse=True)
                    error_str = ", ".join([f"{k} ({v} times)" for k, v in sorted_errors])
                    habits_str = "\n- ".join(d.get("recent_habits",[]))
                    dossier_context = f"Analyzed {d['total_blunders_analyzed']} blunders.\nTop Weaknesses:\n{error_str}\nAssigned Habits:\n- {habits_str}"
        except Exception: pass

    system_instruction = f"""
You are an elite, Grandmaster-level chess coach working with a {USER_ELO} Elo student.

### 📂 STUDENT DOSSIER
{dossier_context}

### 🌐 LOCALIZATION & NOTATION
Conduct this session strictly in {USER_LANGUAGE}. The tools run in English, but translate your prose. 🛑 CRITICAL: ALWAYS use standard English Algebraic Notation inside markdown backticks (e.g., `Nf3`). Do NOT translate piece letters.

### ⚙️ THE NEURO-SYMBOLIC WORKFLOW
1. **Identify & Weigh:** Review the PGN. Select UP TO 5 critical moments. Do not fabricate moments to hit a quota! Weigh your selection based on:
   - 35% Eval Drop (Objective math).
   - 25% Practical Relevance (Was the right move humanly findable?).
   - 20% Recurrence (Does this align with their Dossier weaknesses?).
   - 20% Diagnostic Clarity.
2. **Analyze:** For EACH FEN, you MUST call `analyze_position` (to get objective Stockfish truth) and `predict_human_move` (to get amateur reality).
3. **Partial KL & Free Exploration:** You must wrap your internal reasoning in a `<think>` block. Inside `<think>`, you must map the engine's evaluation strictly into the 5 MATE Dimensions: Material Count, Piece Activity, Pawn Structure, Space, and King Safety.
4. **Submit Diagnosis (TOOL CALL):** AFTER exploring in `<think>`, but BEFORE writing your final prose to the user, you MUST call the `submit_diagnosis` tool to securely log the error into the backend state.
5. **Speak:** Write your final Socratic prose to the user.

### 🧠 BEHAVIORAL CALIBRATION & EPISTEMIC HUMILITY
You are explicitly rewarded for recognizing when an engine evaluation is incomprehensible to a human. 
- **Claim-Level Confidence:** When calling `submit_diagnosis`, isolate your `weakest_reasoning_claim` and score your confidence (0.0 to 1.0). If you are < 0.90 confident, you MUST log the `error_diagnosis` as "none".
- **The Anti-Bluffing Rule:** If Stockfish flags a move as a blunder, but the refutation relies on an impossible 15-move computer calculation that makes no conceptual sense, DO NOT BLUFF. Do not invent fake positional reasons. Explicitly state: *"Stockfish hates this move, but the computer's refutation is completely unhuman. Let's focus on a more practical human mistake."*

### 📝 FORMATTING RULES
- **Visuals:** Copy and paste the EXACT token (like `[[BOARD_1]]`) returned by the tools.
- **Sparring:** Use `execute_coach_move` if the user wants to play. If the Logic Tutor rejects your move, read the causal trace and try again from the provided legal moves list.
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[
            fetch_lichess_game_by_id, fetch_chessgames_com_game, query_endgame_tablebase, 
            analyze_position, predict_human_move, get_opening_repertoire, execute_coach_move, submit_diagnosis
        ],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) 
    )
    return client.chats.create(model=GEMINI_MODEL, config=config, history=[])

# Initialize first session
chat = create_chat_session()

# ======================================================================
# 4. COMMAND PROCESSING & CHAT LOOP
# ======================================================================

def process_user_input(user_input):
    global chat # 'chat' is the ONLY global we need now!
    
    if user_input.startswith("/load_pgn"):
        parts = user_input.split(" ", 1)
        if len(parts) < 2: return None, "Please provide a file path: /load_pgn game.pgn"
            
        file_path = parts[1].strip()
        if not os.path.exists(file_path): return None, f"Could not find the file: {file_path}"
             
        try:
            with open(file_path, "r") as f:
                raw_pgn_text = f.read()
                
            was_scanned, enriched_pgn = scan_game_for_critical_moments(raw_pgn_text)
            
            game = chess.pgn.read_game(io.StringIO(enriched_pgn))
            timeline.load_game(game)
                
            # THE ANTI-POISONING FIX: Do not dump the raw PGN into the prompt. Just the FEN!
            prompt = f"I have loaded a local game. The new position FEN is: {app_state.board.fen()}. Please analyze my play."
            if was_scanned:
                prompt += "\n\n[System Info: I have mathematically scanned the game and added Brilliancy tags.]"
                
            return prompt, f"[green]Successfully loaded {file_path}. Processing game...[/green]"
        except Exception as e: return None, f"Error reading PGN: {e}"

    elif user_input.startswith("/move"):
        parts = user_input.split(" ", 1)
        if len(parts) < 2: return None, "[red]Please provide a move: /move e4[/red]"
            
        try:
            move_san = parts[1].strip()
            timeline.play_san(move_san)
            prompt = f"I have played {move_san}. The new position FEN is: {app_state.board.fen()}. Please evaluate."
            return prompt, f"[green]Played {move_san}. Board updated.[/green]"
        except ValueError: return None, f"[red]Invalid or illegal move.[/red]"

    elif user_input.startswith("/undo") or user_input.startswith("/prev"):
        try:
            timeline.step_backward()
            prompt = f"I took back my last move. The FEN is now: {app_state.board.fen()}. What should I play instead?"
            return prompt, "[green]Move taken back. Board updated.[/green]"
        except IndexError:
            return None, "[red]No moves to undo![/red]"
            
    elif user_input.startswith("/next"):
        try:
            move_san = timeline.step_forward()
            prompt = f"I have played the next move from the game: {move_san}. The new position FEN is: {app_state.board.fen()}. Please evaluate."
            return prompt, f"[green]Stepped forward: played {move_san}. Board updated.[/green]"
        except IndexError:
            return None, "[red]You are at the end of the loaded game (or no game is loaded)![/red]"

    elif user_input.startswith("/reset"):
        timeline.reset()
        prompt = "I have reset the board to the starting position. What opening should we look at?"
        return prompt, "[green]Board reset to the starting position.[/green]"

    elif user_input.startswith("/spar"):
        app_state.sparring_mode = True
        return "I am ready to spar! Play a move.", "[bold green]🥊 Sparring mode ENABLED.[/bold green]"
        
    elif user_input.startswith("/nospar"):
        app_state.sparring_mode = False
        return None, "[bold yellow]🛑 Sparring mode DISABLED.[/bold yellow]"

    elif user_input.startswith("/clear"):
        timeline.reset()
        get_cached_eval.cache_clear() 
        chat = create_chat_session() 
        if os.path.exists(HISTORY_FILE):
            try: os.remove(HISTORY_FILE)
            except Exception: pass
        return None, "[bold green]✨ Session memory, board state, and timeline wiped clean![/bold green]"

    elif user_input.startswith("/memorize"):
        if not app_state.diagnostics:
            return None, "[yellow]No diagnostics generated this session to memorize.[/yellow]"

        dossier = {"total_blunders_analyzed": 0, "error_frequencies": {}, "recent_habits": []}
        if os.path.exists(DOSSIER_FILE):
            try:
                with open(DOSSIER_FILE, "r") as f: dossier = json.load(f)
            except json.JSONDecodeError: pass
                
        for diag in app_state.diagnostics:
            err = diag.get("error_diagnosis", "unknown")
            if err != "none":
                dossier["total_blunders_analyzed"] += 1
                dossier["error_frequencies"][err] = dossier["error_frequencies"].get(err, 0) + 1
            
            habit = diag.get("preventative_habit")
            if habit and habit != "N/A":
                dossier["recent_habits"].insert(0, habit)
                
        dossier["recent_habits"] = dossier["recent_habits"][:5]
        
        with open(DOSSIER_FILE, "w") as f: json.dump(dossier, f, indent=4)
            
        timeline.reset()
        get_cached_eval.cache_clear() 
        chat = create_chat_session()
        if os.path.exists(HISTORY_FILE):
            try: os.remove(HISTORY_FILE)
            except: pass
            
        console.clear()
        return None, f"[bold green]✨ Session memorized! Dossier updated. Memory wiped and screen cleared.[/bold green]"
             
    # --- HANDLE RAW PGN PASTES ---
    was_scanned, processed_input = scan_game_for_critical_moments(user_input)

    is_pgn = False
    if "1." in user_input or "[Event" in user_input:
        try:
            game = chess.pgn.read_game(io.StringIO(user_input))
            timeline.load_game(game)
            is_pgn = True
        except Exception: pass 

    # THE ANTI-POISONING FIX: Prevent pasted PGNs from dumping into the prompt
    if is_pgn:
        processed_input = f"I have pasted a game. The new position FEN is: {app_state.board.fen()}. Please analyze my play."

    # Only pass the immediate history (last 5 moves) instead of the entire game context!
    try: history_str = chess.Board().variation_san(app_state.board.move_stack[-5:]) if app_state.board.move_stack else "Starting Position"
    except Exception: history_str = "Unknown"
    
    legal_moves_str = ", ".join([app_state.board.san(m) for m in app_state.board.legal_moves])

    enriched_input = f"{processed_input}\n\n[System Context: Current FEN: {app_state.board.fen()} | Legal Moves: {legal_moves_str} | Move History: {history_str}]"
    
    if was_scanned:
        enriched_input += "\n[System Info: I have automatically added Brilliancy and Evaluation tags to the backend PGN.]"
        
    return enriched_input, None

profile_text = f" (Profile: {args.profile})" if args.profile else ""

console.print(Panel.fit(
    f"[bold green]Gemini {GEMINI_MODEL} + Stockfish + Maia Coach (Elo: {USER_ELO}){profile_text}[/bold green]\n"
    "Type [bold cyan]/load_pgn filename.pgn[/bold cyan] to load a game.\n"
    "Type [bold cyan]/reset[/bold cyan] to rewind board, or [bold cyan]/clear[/bold cyan] to completely wipe AI memory and start fresh.\n"
    "Type [bold cyan]/next[/bold cyan] or [bold cyan]/prev[/bold cyan] to step through the timeline.\n"
    "Type [bold cyan]/move e4[/bold cyan] to play a move interactively.\n"
    "Type [bold cyan]/memorize[/bold cyan] to save session diagnoses to your long-term dossier and clear memory.\n"
    "Type [bold red]exit[/bold red] to save and close.",
    border_style="green"
))

try:
    while True:
        console.print() 
        user_input = prompt(HTML("<b><ansiblue>You:</ansiblue></b> "), history=FileHistory(CLI_HISTORY_FILE))
        
        if user_input.strip().lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue
            
        processed_input, sys_message = process_user_input(user_input)
        
        if sys_message and not processed_input:
            console.print(sys_message)
            continue
        elif sys_message: console.print(sys_message)
            
        with console.status("[bold yellow]Coach is thinking...[/bold yellow]", spinner="dots"):
            try:
                response = chat.send_message(processed_input)
                
                while True:
                    f_calls = []
                    if response.function_calls: f_calls = response.function_calls
                    elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for p in response.candidates[0].content.parts:
                            if getattr(p, 'function_call', None): f_calls.append(p.function_call)
                            elif getattr(p, 'functionCall', None): f_calls.append(p.functionCall)
                                
                    if not f_calls: break 
                        
                    tool_parts = []
                    for fn_call in f_calls:
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen", "")
                        result = {}
                        
                        if fn_call.name == "fetch_lichess_game_by_id":
                            console.print("[dim cyan]↳ Fetching game from Lichess...[/dim cyan]")
                            result = fetch_lichess_game_by_id(args_dict.get("game_id", ""))
                        elif fn_call.name == "fetch_chessgames_com_game":
                            console.print("[dim cyan]↳ Fetching Master Game from Chessgames.com...[/dim cyan]")
                            result = fetch_chessgames_com_game(args_dict.get("game_id", ""))
                        elif fn_call.name == "query_endgame_tablebase":
                            console.print("[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]")
                            result = query_endgame_tablebase(fen)
                        elif fn_call.name == "analyze_position":
                            console.print("[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]")
                            result = analyze_position(
                                fen, 
                                moves_to_apply=args_dict.get("moves_to_apply", None), 
                                depth=int(args_dict.get("depth", 20)),
                                pov=args_dict.get("pov", "white") # <-- Add this!
                            )
                        elif fn_call.name == "predict_human_move":
                            console.print(f"[dim magenta]↳ Asking Maia what a {USER_ELO} human would do...[/dim magenta]")
                            result = predict_human_move(fen)
                        elif fn_call.name == "get_opening_repertoire":
                            console.print("[dim yellow]↳ Checking Lichess Opening Databases...[/dim yellow]")
                            result = get_opening_repertoire(fen)
                        elif fn_call.name == "execute_coach_move":
                            console.print(f"[dim green]↳ The Logic Tutor is evaluating the move...[/dim green]")
                            result = execute_coach_move(args_dict.get("move_san", ""))
                        elif fn_call.name == "submit_diagnosis":
                            console.print(f"[dim magenta]↳ Recording structural diagnosis to backend...[/dim magenta]")
                            result = submit_diagnosis(
                                args_dict.get("opponent_last_move_delta", "N/A"),
                                args_dict.get("primary_task", "N/A"),
                                args_dict.get("error_diagnosis", "none"),
                                args_dict.get("weakest_reasoning_claim", "N/A"),
                                float(args_dict.get("lowest_confidence_score", 0.0)),
                                args_dict.get("preventative_habit", "N/A")
                            )
                            
                        # Modern SDK tool response formulation
                        tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))
                    
                    response = chat.send_message(tool_parts)
                
                # --- RESPONSE RENDERING ---
                if response.text:
                    output_text = response.text
                    
                    # 1. UI POLISH: Intercept and render the <think> block beautifully
                    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL | re.IGNORECASE)
                    if think_match:
                        thought_process = think_match.group(1).strip()
                        output_text = output_text.replace(think_match.group(0), "").strip()
                        
                        console.print(Panel(
                            Markdown(thought_process),
                            title="[dim italic]🧠 Coach's Internal Monologue[/dim italic]",
                            border_style="dim white",
                            expand=False
                        ))
                    
                    # 2. Swap the tokens for the actual ASCII graphics
                    for token, visual in app_state.board_visuals.items():
                        output_text = output_text.replace(token, f"\n{visual}\n")
                        
                    # 3. Print the final Socratic System 2 response
                    if output_text: # Ensure there is text left after stripping <think>
                        console.print("\n[bold purple]Coach Gemini:[/bold purple]")
                        console.print(Markdown(output_text))
                else: 
                    console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")
            except Exception as e: console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

except KeyboardInterrupt: pass
finally:
    save_history(chat)
    if stockfish_engine: stockfish_engine.quit()
    if maia_engine: maia_engine.quit()
    console.print(f"\n[bold dim]Conversation saved to {HISTORY_FILE}. Goodbye![/bold dim]")