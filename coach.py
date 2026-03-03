import io
import os
import json
import argparse
import shutil
import requests
import subprocess
import chess.pgn
import chess.engine
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from dotenv import load_dotenv
from prompt_toolkit import prompt, HTML
from prompt_toolkit.history import FileHistory

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
STOCKFISH_PATH = app_config.get("stockfish_path")
LC0_PATH = app_config.get("lc0_path")
MAIA_DIR = app_config.get("maia_weights_dir", "./maia_weights")
GEMINI_MODEL = app_config.get("gemini_model", "gemini-3.1-pro-preview")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set.")
    exit(1)

client = genai.Client(api_key=api_key)

current_board = chess.Board()

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

engine_cache = {}

def eval_move(board, move, time_limit, turn_color):
    """Helper for evaluating non-PV moves safely with LRU caching."""
    board.push(move)
    fen = board.fen()
    key = (fen, time_limit)
    if key not in engine_cache:
        engine_cache[key] = stockfish_engine.analyse(board, chess.engine.Limit(time=time_limit))
    
    score = engine_cache[key]["score"].pov(turn_color).score(mate_score=100000)
    board.pop()
    return score

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
                    board.push(played_move)
                    
                    # Apply tags
                    tag_string = ""
                    if is_only_move: tag_string += " [⭐ ONLY MOVE FOUND!]"
                    if dodged_trap: tag_string += f" [🛡️ DODGED HUMAN TRAP! Maia predicted {maia_san}]"
                            
                    # Inject basic evaluation if missing
                    if "[%eval" not in user_text and stockfish_engine:
                        fen_after = board.fen()
                        if (fen_after, 0.25) not in engine_cache:
                            engine_cache[(fen_after, 0.25)] = stockfish_engine.analyse(board, chess.engine.Limit(time=0.25))
                        score_after = engine_cache[(fen_after, 0.25)]["score"].white()
                        
                        eval_str = f"#{score_after.mate()}" if score_after.is_mate() else f"{score_after.score(mate_score=100000) / 100.0:+.2f}"
                        tag_string += f" [%eval {eval_str}]"

                    # --- NEW: ALWAYS INJECT THE EXACT FEN SO THE AI NEVER HAS TO GUESS! ---
                    if "[%fen" not in comment:
                        tag_string += f' [%fen "{board.fen()}"]'

                    if tag_string: node.comment = (comment + tag_string).strip()
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
                annotated_games.append(game.accept(exporter))
                
        if annotated_games:
            console.print("[green]Scan complete![/green]")
            return True, "\n\n".join(annotated_games)
            
    except Exception as e:
        console.print(f"[dim red]Game scan failed: {e}[/dim red]")
        
    return False, user_text

def analyze_position(fen: str, moves_to_apply: list[str] = None, depth: int = 15, pov: str = "white") -> dict:
    """
    Evaluates a position using Stockfish. 
    'moves_to_apply' applies hypothetical moves first.
    'pov' determines board orientation ('white' or 'black').
    """
    try:
        board = chess.Board(fen)
        if moves_to_apply:
            for move_str in moves_to_apply: board.push_san(move_str)
                
        infos = stockfish_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
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

        # FORCE-FEED THE VISUAL BOARD
        raw_board = board.unicode(empty_square="·", invert_color=True)
        lines = raw_board.split('\n')
        
        labeled_board = []
        
        # --- NEW: POV FLIPPING LOGIC ---
        if pov.lower() == "black":
            lines.reverse() # Flips ranks 1-8 to 8-1
            for i, line in enumerate(lines):
                # Reverse the characters in the line (flips files a-h to h-a)
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

        return {"top_stockfish_lines": top_lines, "mandatory_board_visual": f"\n{visual}\n"}
    except Exception as e: return {"error": str(e)}

def predict_human_move(fen: str) -> dict:
    if not maia_engine: return {"error": "Maia engine not loaded."}
    try:
        board = chess.Board(fen)
        result = maia_engine.play(board, chess.engine.Limit(nodes=1))
        if not result.move: return {"predicted_human_move": "None"}
            
        maia_move_san = board.san(result.move)
        board.push(result.move)
        
        info = stockfish_engine.analyse(board, chess.engine.Limit(depth=15))
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

system_instruction = f"""
You are an elite, Grandmaster-level chess coach. You are currently working one-on-one with a student rated {USER_ELO} Elo. 
Your goal is not just to give the student the engine's top move, but to improve their understanding of chess concepts, calculation, and human psychology.

### 🛑 CRITICAL RULE: NO TOOL HALLUCINATIONS
You are STRICTLY FORBIDDEN from guessing, inferring, or fabricating engine evaluations, Maia predictions, or opening statistics. 
If you say "Stockfish says" or "Maia predicts", you MUST have actually executed the `analyze_position` or `predict_human_move` tools for that exact position during this conversation turn. 

### ⚙️ MANDATORY GAME ANALYSIS WORKFLOW
When the user asks you to analyze a downloaded game or a sequence of moves, you CANNOT just read the PGN and immediately reply. You MUST execute this exact step-by-step loop BEFORE generating your final text response:
1. **Fetch:** Use the appropriate tool to download the game.
2. **Identify:** Read the PGN silently and isolate 4 to 5 critical moments. You MUST strive for a balance of negative and positive reinforcement:
   - **Mistakes:** Identify critical blunders or mistakes (look for sudden drops in `[%eval]`).
   - **Brilliant Saves:** Look specifically for moves explicitly tagged with `[⭐ ONLY MOVE FOUND!]` or `[🛡️ DODGED HUMAN TRAP!]` in the PGN. If the student navigated a minefield or successfully avoided a tempting human blunder, heavily praise them!
3. **Extract FENs:** To analyze a specific move, you need the mathematically exact FEN of the board *before* that move was played. Do NOT calculate this or use tools to reconstruct it! The system secretly injects the true FEN into the PGN comments after every single move (e.g., `[%fen "..."]`). Simply copy the exact FEN string from the comment of the move *immediately preceding* the blunder/moment! (For White's Move 1, use the starting FEN: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`).
4. **Analyze:** For EACH FEN generated in Step 3, you MUST call:
   - `analyze_position` (to get objective Stockfish truth).
   - `predict_human_move` (to see what a {USER_ELO} player would actually do).
    - `get_opening_repertoire` (if it is the opening phase, to compare Grandmaster theory against amateur reality).
5. **Respond:** ONLY AFTER you have gathered real data using these tools are you allowed to synthesize your final textual response to the student.

### ♟️ COACHING METHODOLOGY & PEDAGOGY
1. **The Socratic Method:** Do not spoon-feed answers. Ask leading questions. If the student makes a mistake, ask them: "What did your opponent's last move threaten?" Make them think.
2. **Objective vs. Practical Reality:** Synthesize objective truth (Stockfish) and human probability (Maia / Lichess database). 
   - If Stockfish suggests a move that is only good because of an impossible-to-see 5-move+ tactic, warn the student.
   - The `predict_human_move` tool now returns BOTH the human impulse AND the Stockfish refutation of that impulse! If Maia predicts the student or opponent will play a tempting blunder, explicitly warn them: "I know [Maia's Move] looks tempting, but if you play it, the evaluation drops to [Evaluation] because of this refutation: [Stockfish Refutation Line]."
3. **Speak in Concepts:** At {USER_ELO} Elo, a simple, solid plan is better than a complex engine sequence. Prioritize explaining concepts like King Safety, Piece Activity, controlling the center, and identifying undefended pieces.
4. **Grounding in Theory:** If `get_opening_repertoire` provides a `theoretical_opening_name`, explicitly state it. Use your deep internal knowledge of chess theory to briefly explain the classical Grandmaster plans. Then, compare the `top_master_moves` against the `typical_amateur_moves` to show the student how their opponents are likely to deviate from GM theory and how to punish it!
5. **Handling Hypotheticals:** If the user asks "What if I play [Move]?", "Why can't I play [Move]?", or suggests an alternative idea, IMMEDIATELY use the `analyze_position` tool and pass their proposed move sequence into the `moves_to_apply` parameter. Use the returned `expected_engine_line` to prove exactly how their idea would be mathematically punished or rewarded!
6. **Time Management (Clock Analysis):** If the provided PGN contains clock timestamps (e.g., `[%clk 0:10:00]`), you MUST evaluate the student's time management on their critical mistakes. 
   - Calculate the estimated time spent by comparing the clock on their blunder to the clock on their *previous* turn (factor in any time increment if noted in the headers). 
   - **"Hope Chess" (Impulse Blunders):** If they make a critical blunder or miss a tactic in under 10 seconds, explicitly call them out! Tell them: "You only spent X seconds on this move. You played 'Hope Chess' instead of calculating."
   - **"Freezing" (Overthinking):** If they spend over 60 seconds on a completely forced move (like escaping a check or making an obvious recapture), warn them about "freezing" and hallucinating phantom threats.
   - ALWAYS weave the exact time they spent into your explanation.
7. **Positive Reinforcement:** 
   - If the student found a move tagged with `[⭐ ONLY MOVE FOUND!]`, explicitly praise them for navigating a tactical minefield where any other move would lose.
   - If a move is tagged with `[🛡️ DODGED HUMAN TRAP!]`, explicitly praise them for resisting temptation! Explain *why* the natural human impulse (the move Maia predicted) was a dangerous trap, and how the student's actual move brilliantly bypassed it.
   - 🛑 CRITICAL FORMATTING RULE: The tags `[⭐ ONLY MOVE FOUND!]` and `[🛡️ DODGED HUMAN TRAP!]` are secret backend metadata for your eyes only. NEVER literally print these bracketed tags or emojis in your final text response to the student! Instead, translate the concept into natural, conversational human language (e.g., "You found the absolute only move on the board to save the game!", or "Maia predicted most players would play X, but you successfully dodged the trap!").
8. **Make Annotations Provable:** For every "avoid X" or "X is a blunder" claim you make, you MUST include a 3-to-5 move refutation line derived from your engine tool results to prove *why*. Do not just say a move loses; mathematically prove it with the Stockfish continuation.

### 📝 FORMATTING & RULES OF ENGAGEMENT
- **Current State:** The system automatically appends the current board state (FEN) to every message the user sends. You ALWAYS know the current board state.
- **Handling Follow-ups & Past Moves:** If the user asks a follow-up question about a specific past move, DO NOT guess the board state! Find that move in the PGN, grab the `[%fen "..."]` tag from the move *prior* to it, and pass it to `analyze_position` to get the true visual board before replying.
- **Endgames:** If there are 7 or fewer pieces on the board, immediately use `query_endgame_tablebase`.
- **MANDATORY VISUALIZATIONS & LINES:** The `analyze_position` tool will now automatically return a `mandatory_board_visual` string and an `expected_engine_line`. 
  - 🛑 CRITICAL: NEVER attempt to draw, type, or format an ASCII chess board yourself! You are highly prone to hallucinating piece locations and colors (e.g., putting White pieces on Rank 8).
  - You MUST copy and paste the exact `mandatory_board_visual` string into your chat response whenever you analyze a position. NEVER discuss a position without showing the board! DO NOT wrap the board in markdown code blocks.
  - Use the `expected_engine_line` to definitively prove *why* a move is bad by showing the student how the opponent would mathematically punish them!
  - **Board POV:** The `analyze_position` tool accepts a `pov` argument ("white" or "black"). You MUST set `pov="black"` if the student played the Black pieces in the analyzed game, so the board renders correctly from their perspective. Otherwise, default to "white".
- **CHESS NOTATION FORMATTING (STRICT UI REQUIREMENT):**
  - You MUST wrap EVERY single chess square, individual move, and move sequence in markdown backticks to render them as `inline code` (e.g., `e4`, `Nf3`, `1. d4 d5`).
  - DO NOT wrap plain English piece names in backticks! Write "your King" or "the Pawn", NEVER "your `King`" or "the `Pawn`". ONLY use backticks for formal algebraic notation and squares.
  - Our terminal UI relies entirely on these backticks to generate the visual highlight boxes. 
- Break up your text into easily readable paragraphs or bullet points.
- Even if you are using the Socratic approach with the user, NEVER use the word "Socratic" in your output.
"""

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    tools=[
        fetch_lichess_game_by_id, 
        fetch_chessgames_com_game, query_endgame_tablebase, 
        analyze_position, predict_human_move, get_opening_repertoire
    ],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) 
)

chat = client.chats.create(
    model=GEMINI_MODEL,
    config=config,
    history=load_history()
)

# ======================================================================
# 4. COMMAND PROCESSING & CHAT LOOP
# ======================================================================

def process_user_input(user_input):
    global current_board
    
    if user_input.startswith("/load_pgn"):
        parts = user_input.split(" ", 1)
        if len(parts) < 2: return None, "Please provide a file path: /load_pgn game.pgn"
            
        file_path = parts[1].strip()
        if not os.path.exists(file_path): return None, f"Could not find the file: {file_path}"
             
        try:
            with open(file_path, "r") as f:
                raw_pgn_text = f.read()
                
            # RUN SCANNER
            was_scanned, enriched_pgn = scan_game_for_critical_moments(raw_pgn_text)
            
            game = chess.pgn.read_game(io.StringIO(enriched_pgn))
            if game: current_board = game.end().board()
                
            # Safely inject the system prompt outside the PGN text
            prompt = f"I have loaded a local game. Here is the full PGN:\n\n{enriched_pgn}\n\nPlease analyze my play and my time management."
            if was_scanned:
                prompt += "\n\n[System Info: I have mathematically scanned the PGN above and added Brilliancy tags. Please use them for your analysis.]"
                
            return prompt, f"[green]Successfully loaded {file_path}. Processing game...[/green]"
        except Exception as e: return None, f"Error reading PGN: {e}"

    elif user_input.startswith("/move"):
        parts = user_input.split(" ", 1)
        if len(parts) < 2: return None, "[red]Please provide a move: /move e4[/red]"
            
        move_str = parts[1].strip()
        try:
            current_board.push_san(move_str)
            prompt = f"I have played {move_str}. The new position FEN is: {current_board.fen()}. Please evaluate."
            return prompt, f"[green]Played {move_str}. Board updated.[/green]"
        except ValueError: return None, f"[red]Invalid or illegal move: {move_str}[/red]"
             
    # --- IF THE USER PASTED A RAW PGN INTO THE TERMINAL ---
    was_scanned, processed_input = scan_game_for_critical_moments(user_input)

    try: history_str = chess.Board().variation_san(current_board.move_stack) if current_board.move_stack else "Starting Position"
    except Exception: history_str = "Unknown"

    enriched_input = f"{processed_input}\n\n[System Context for Coach: Current FEN: {current_board.fen()} | Move History: {history_str}]"
    
    if was_scanned:
        enriched_input += "\n[System Info: If the text above was a PGN, I have automatically added Brilliancy and Evaluation tags to it.]"
        
    return enriched_input, None

profile_text = f" (Profile: {args.profile})" if args.profile else ""

console.print(Panel.fit(
    f"[bold green]Gemini {GEMINI_MODEL} + Stockfish + Maia Coach (Elo: {USER_ELO}){profile_text}[/bold green]\n"
    "Type [bold cyan]/load_pgn filename.pgn[/bold cyan] to load a game.\n"
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
                                depth=int(args_dict.get("depth", 15)),
                                pov=args_dict.get("pov", "white") # <-- Add this!
                            )
                        elif fn_call.name == "predict_human_move":
                            console.print(f"[dim magenta]↳ Asking Maia what a {USER_ELO} human would do...[/dim magenta]")
                            result = predict_human_move(fen)
                        elif fn_call.name == "get_opening_repertoire":
                            console.print("[dim yellow]↳ Checking Lichess Opening Databases...[/dim yellow]")
                            result = get_opening_repertoire(fen)
                            
                        tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))
                    
                    response = chat.send_message(tool_parts)
                if response.text:
                    console.print("\n[bold purple]Coach Gemini:[/bold purple]")
                    console.print(Markdown(response.text))
                else: console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")

            except Exception as e: console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

except KeyboardInterrupt: pass
finally:
    save_history(chat)
    if stockfish_engine: stockfish_engine.quit()
    if maia_engine: maia_engine.quit()
    console.print(f"\n[bold dim]Conversation saved to {HISTORY_FILE}. Goodbye![/bold dim]")