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
from stockfish import Stockfish
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
         "This isolates your Elo settings, chat memory, and command history.\n"
         "If the profile does not exist, it will be created using the root config.json as a template.", 
    default=None
)
args = parser.parse_args()

# Default root paths
ROOT_CONFIG_FILE = "config.json"
CONFIG_FILE = ROOT_CONFIG_FILE
HISTORY_FILE = "chess_coach_memory.json"
CLI_HISTORY_FILE = "chess_cli_history.txt"

if not os.path.exists(ROOT_CONFIG_FILE):
    console.print(f"[bold red]Error:[/bold red] {ROOT_CONFIG_FILE} not found in the root directory. Please create it first.")
    exit(1)

# Read the root config first to establish the base profiles directory
with open(ROOT_CONFIG_FILE, "r") as f:
    root_config = json.load(f)

# Allow users to customize where profiles are stored via the root config
BASE_PROFILES_DIR = root_config.get("profiles_dir", "./profiles")

# Handle Profile Logic
if args.profile:
    profile_dir = os.path.join(BASE_PROFILES_DIR, args.profile)
    # os.makedirs safely creates intermediate directories (like 'profiles') if they don't exist
    os.makedirs(profile_dir, exist_ok=True)
    
    CONFIG_FILE = os.path.join(profile_dir, "config.json")
    HISTORY_FILE = os.path.join(profile_dir, "chess_coach_memory.json")
    CLI_HISTORY_FILE = os.path.join(profile_dir, "chess_cli_history.txt")
    
    # If the profile doesn't have a config yet, copy the root one
    if not os.path.exists(CONFIG_FILE):
        console.print(f"[yellow]Creating new config for profile '{args.profile}' from root config.[/yellow]")
        shutil.copy(ROOT_CONFIG_FILE, CONFIG_FILE)
if args.wipe_memory and os.path.exists(HISTORY_FILE):
    os.remove(HISTORY_FILE)
    console.print(f"[bold red]Memory wiped:[/bold red] {HISTORY_FILE} has been cleared.")

# Now read the final, active config (either root or profile-specific)
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

# We keep a persistent board state for conversational back-and-forth
current_board = chess.Board()

# Initialize Stockfish globally using python-chess (Better than the stockfish wrapper!)
try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
except FileNotFoundError:
    console.print(f"[bold red]Error:[/bold red] Stockfish not found at {STOCKFISH_PATH}.")
    exit(1)

# ======================================================================
# NEW: Initialize Maia globally to eliminate latency!
# ======================================================================
try:
    rounded_elo = round(USER_ELO / 100) * 100
    if rounded_elo <= 1300:
        maia_target = 1100
    elif rounded_elo <= 1700:
        maia_target = 1500
    else:
        maia_target = 1900

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
# 2. THE TOOLS (STOCKFISH, MAIA, LICHESS EXPLORER)
# ======================================================================

def scan_game_for_critical_moments(user_text: str) -> str:
    """Scans the PGN and injects custom tags for Only Moves and Dodged Traps."""
    if "1." not in user_text:
        return user_text 
        
    try:
        pgn_io = io.StringIO(user_text)
        game = chess.pgn.read_game(pgn_io)
        if not game or len(list(game.mainline_moves())) < 3:
            return user_text
            
        console.print("\n[dim yellow]Scanning game(s) for brilliant moments and blunders...[/dim yellow]")
        
        annotated_games = []
        with console.status("[bold yellow]Stockfish & Maia are analyzing critical paths...[/bold yellow]", spinner="dots"):
            pgn_io = io.StringIO(user_text)
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None: break
                
                board = game.board()
                for node in game.mainline():
                    played_move = node.move
                    is_only_move = False
                    dodged_trap = False
                    maia_san = ""
                    
                    if stockfish_engine:
                        turn_color = board.turn
                        # Baseline the position BEFORE the move
                        info = stockfish_engine.analyse(board, chess.engine.Limit(time=0.1), multipv=2)
                        
                        if info and len(info) > 0 and "pv" in info[0]:
                            best_move = info[0]["pv"][0]
                            best_score = info[0]["score"].pov(turn_color).score(mate_score=100000)
                            
                            if best_score is not None:
                                # 1. ONLY MOVE CHECK
                                if len(info) > 1 and played_move == best_move and "pv" in info[1]:
                                    second_score = info[1]["score"].pov(turn_color).score(mate_score=100000)
                                    if second_score is not None and (best_score - second_score) >= 150: # 1.5 pawn gap
                                        is_only_move = True
                                
                                # 2. DODGED HUMAN TRAP CHECK (Completely Independent!)
                                if maia_engine:
                                    maia_res = maia_engine.play(board, chess.engine.Limit(nodes=1))
                                    maia_move = maia_res.move if maia_res else None
                                    
                                    if maia_move and maia_move != played_move:
                                        # Fast Eval: How good was the move you actually played?
                                        if played_move == best_move:
                                            played_score = best_score
                                        elif len(info) > 1 and "pv" in info[1] and played_move == info[1]["pv"][0]:
                                            played_score = info[1]["score"].pov(turn_color).score(mate_score=100000)
                                        else:
                                            board.push(played_move)
                                            info_played = stockfish_engine.analyse(board, chess.engine.Limit(time=0.05))
                                            played_score = info_played["score"].pov(turn_color).score(mate_score=100000)
                                            board.pop()
                                            
                                        # Fast Eval: How bad was the move Maia predicted?
                                        if maia_move == best_move:
                                            maia_score = best_score
                                        elif len(info) > 1 and "pv" in info[1] and maia_move == info[1]["pv"][0]:
                                            maia_score = info[1]["score"].pov(turn_color).score(mate_score=100000)
                                        else:
                                            board.push(maia_move)
                                            info_maia = stockfish_engine.analyse(board, chess.engine.Limit(time=0.05))
                                            maia_score = info_maia["score"].pov(turn_color).score(mate_score=100000)
                                            board.pop()
                                            
                                        if played_score is not None and maia_score is not None:
                                            # If your move was solid (within 1 pawn of the absolute best move)
                                            # AND Maia's predicted move was a blunder (at least 1.5 pawns worse than what you played)
                                            if (best_score - played_score) <= 100 and (played_score - maia_score) >= 150:
                                                dodged_trap = True
                                                maia_san = board.san(maia_move)
                    
                    comment = node.comment if node.comment else ""
                    board.push(played_move)
                    
                    # Apply the tags
                    tag_string = ""
                    if is_only_move:
                        tag_string += " [⭐ ONLY MOVE FOUND!]"
                    if dodged_trap:
                        tag_string += f" [🛡️ DODGED HUMAN TRAP! Maia predicted {maia_san}]"
                            
                    # Inject basic evaluation if missing
                    if "[%eval" not in user_text and stockfish_engine:
                        info_after = stockfish_engine.analyse(board, chess.engine.Limit(time=0.1))
                        score_after = info_after["score"].white()
                        if score_after.is_mate(): eval_str = f"#{score_after.mate()}"
                        else: eval_str = f"{score_after.score(mate_score=100000) / 100.0:+.2f}"
                        tag_string += f" [%eval {eval_str}]"

                    if tag_string:
                        node.comment = (comment + tag_string).strip()
                
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
                annotated_games.append(game.accept(exporter))
                
        if annotated_games:
            console.print("[green]Scan complete![/green]")
            annotated_text = "\n\n".join(annotated_games)
            return f"{user_text}\n\n[System Info: I have scanned the PGN and added Brilliancy tags below. Please use THIS version of the PGN for your analysis:]\n\n{annotated_text}"
            
    except Exception as e:
        console.print(f"[dim red]Game scan failed: {e}[/dim red]")
        
    return user_text

def get_fen_from_moves(starting_fen: str, moves: list[str]) -> dict:
    """
    Applies a list of algebraic moves (e.g., ["e4", "c5", "Nf3"]) to the starting position 
    and returns the resulting FEN string. Use this when the user suggests a sequence of moves 
    and you need the exact FEN to analyze it.
    """
    try:
        temp_board = chess.Board(starting_fen)
        for move_str in moves:
            temp_board.push_san(move_str)
        return {"fen": temp_board.fen()}
    except ValueError as e:
        return {"error": f"Invalid move sequence: {e}. Starting FEN was: {starting_fen}"}

def analyze_position(fen: str, moves_to_apply: list[str] = None, depth: int = 15) -> dict:
    """
    Evaluates a position using Stockfish. 
    If 'moves_to_apply' is provided, it applies those hypothetical moves first before evaluating.
    Returns the objective evaluation, the deep continuation line (PV), and a visual board.
    """
    try:
        board = chess.Board(fen)
        
        # NEW: Apply hypothetical moves if the user is asking "What if I play..."
        if moves_to_apply:
            for move_str in moves_to_apply:
                board.push_san(move_str)
                
        # multipv=2 tells Stockfish to calculate the top 2 best lines simultaneously
        infos = stockfish_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
        
        top_lines = []
        for i, info in enumerate(infos):
            # Format the evaluation score cleanly
            score = info["score"].white()
            if score.is_mate(): eval_str = f"Mate in {score.mate()}"
            else: eval_str = f"{score.score(mate_score=100000) / 100.0:+.2f}"
            
            # Extract the Principal Variation (PV) list of moves
            pv_moves = []
            temp_board = board.copy()
            for move in info.get("pv", [])[:5]:  
                pv_moves.append(temp_board.san(move))
                temp_board.push(move)
                
            top_lines.append({
                "rank": i + 1,
                "evaluation": eval_str,
                "expected_engine_line": " ".join(pv_moves) 
            })

        # FORCE-FEED THE VISUAL BOARD (Now shows the board AFTER the hypothetical moves!)
        raw_board = board.unicode(empty_square="·", invert_color=True)
        lines = raw_board.split('\n')
        labeled_board = [f"{8-i} │ {line} │  " for i, line in enumerate(lines)]
        labeled_board.append("\u00A0\u00A0└" + "─" * 17 + "┘  ")
        labeled_board.append("\u00A0\u00A0\u00A0\u00A0a b c d e f g h  ")
        visual = "\n".join(labeled_board)

        return {
            "top_stockfish_lines": top_lines,
            "mandatory_board_visual": f"\n\n{visual}\n\n"
        }
    except Exception as e: 
        return {"error": str(e)}

def predict_human_move(fen: str) -> dict:
    """
    Predicts the most likely move a human player would make using Maia.
    THEN, temporarily plays that move and asks Stockfish how to mathematically punish it!
    """
    # Assuming maia_engine and stockfish_engine are initialized globally
    if not maia_engine: return {"error": "Maia engine not loaded."}
    
    try:
        board = chess.Board(fen)
        
        # 1. Ask Maia for the human impulse (1 node only)
        result = maia_engine.play(board, chess.engine.Limit(nodes=1))
        if not result.move:
            return {"predicted_human_move": "None"}
            
        maia_move_san = board.san(result.move)
        
        # 2. Apply the human move to the board
        board.push(result.move)
        
        # 3. Ask Stockfish: "What is the objective reality of this new position?"
        info = stockfish_engine.analyse(board, chess.engine.Limit(depth=15))
        
        # Format the resulting evaluation
        score = info["score"].white()
        if score.is_mate(): eval_str = f"Mate in {score.mate()}"
        else: eval_str = f"{score.score(mate_score=100000) / 100.0:+.2f}"
        
        # Extract Stockfish's punishment/continuation line
        refutation_moves = []
        temp_board = board.copy()
        for move in info.get("pv", [])[:4]: # Get the next 4 moves of the punishment
            refutation_moves.append(temp_board.san(move))
            temp_board.push(move)
            
        return {
            "predicted_human_move": maia_move_san,
            "resulting_evaluation_if_played": eval_str,
            "stockfish_refutation_line": " ".join(refutation_moves)
        }
        
    except Exception as e: 
        return {"error": str(e)}

def get_opening_repertoire(fen: str) -> dict:
    """
    Looks up the current FEN in BOTH the Lichess Master database and the Amateur database.
    Use this to compare Grandmaster theory against what players at the user's Elo actually play.
    """
    try:
        # 1. Fetch Master (Grandmaster) Data
        master_url = f"https://explorer.lichess.ovh/masters?fen={fen}"
        master_res = requests.get(master_url).json()
        
        # Extract the Memory Trigger from the Master DB
        opening_info = master_res.get("opening", {})
        opening_name = opening_info.get("name")
        eco_code = opening_info.get("eco")
        
        master_moves = []
        for move in master_res.get("moves", [])[:4]:
            total = move["white"] + move["draws"] + move["black"]
            if total > 0:
                master_moves.append({
                    "move": move["san"],
                    "white_win_pct": round((move["white"] / total) * 100, 1),
                    "draw_pct": round((move["draws"] / total) * 100, 1),
                    "black_win_pct": round((move["black"] / total) * 100, 1)
                })

        # 2. Fetch Amateur Data (Dynamic to User Elo)
        brackets = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]
        valid_brackets = [str(b) for b in brackets if abs(b - USER_ELO) <= 250]
        ratings_param = ",".join(valid_brackets) if valid_brackets else "1000,1200"
        
        amateur_url = f"https://explorer.lichess.ovh/lichess?variant=standard&speeds=blitz,rapid&ratings={ratings_param}&fen={fen}"
        amateur_res = requests.get(amateur_url).json()
        
        amateur_moves = []
        for move in amateur_res.get("moves", [])[:4]:
            total = move["white"] + move["draws"] + move["black"]
            if total > 0:
                amateur_moves.append({
                    "move": move["san"],
                    "total_games": total,
                    "white_win_pct": round((move["white"] / total) * 100, 1),
                    "black_win_pct": round((move["black"] / total) * 100, 1)
                })
            
        return {
            "theoretical_opening_name": opening_name,
            "eco_code": eco_code,
            "top_master_moves": master_moves,
            "typical_amateur_moves": amateur_moves,
            "amateur_brackets_queried": ratings_param
        }
        
    except Exception as e:
        return {"error": str(e)}

def fetch_lichess_game_by_id(game_id: str) -> dict:
    """
    Fetches a specific game from Lichess by its ID, retrieving the maximum 
    available information including computer analysis, evaluations, and clocks.
    
    Args:
        game_id: The 8 or 12 character Lichess game ID (e.g., 'Qa7FJNk2').
    """
    try:
        url = f"https://lichess.org/game/export/{game_id}"
        
        # Request all possible analytical data Lichess might have stored for the game
        params = {
            "evals": "true",     # Include computer evaluations
            "clocks": "true",    # Include clock times
            "accuracy": "true",  # Include accuracy metrics (if available)
            "opening": "true",   # Include opening name and ECO code
            "literate": "true",  # Insert textual annotations like "Blunder" or "Mistake"
            "tags": "true"       # Include standard PGN headers (players, elo, etc.)
        }
        
        # Setting Accept to application/x-chess-pgn gets the fully annotated PGN text
        headers = {
            "Accept": "application/x-chess-pgn"
        }
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            console.print("[dim yellow]↳ Running Brilliancy Scanner on downloaded Lichess game...[/dim yellow]")
            scanned_pgn = scan_game_for_critical_moments(response.text)
            return {"pgn": scanned_pgn}
        elif response.status_code == 404:
            return {"error": f"Game with ID '{game_id}' not found."}
        else:
            return {"error": f"Lichess API returned status {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

def fetch_chessgames_com_game(game_id: str) -> dict:
    """
    Fetches a specific master game's PGN from chessgames.com using its numeric game ID.
    
    Args:
        game_id: The numeric ID of the game (e.g., '1111195').
    """
    try:
        url = f"https://www.chessgames.com/njs/api/game/viewPGN/{game_id}?raw=true"
        
        # It is good practice to include a User-Agent so the website doesn't block the Python request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GeminiChessCoach/1.0",
            "Accept": "text/plain, application/x-chess-pgn, */*"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            pgn_text = response.text.strip()
            
            # Basic validation to ensure we didn't get an empty or error page
            if not pgn_text or not pgn_text.startswith("[Event"):
                return {"error": "Received malformed data. Are you sure this is a valid game ID?"}
                
            return {"pgn": pgn_text}
            
        elif response.status_code == 404:
            return {"error": f"Game with ID '{game_id}' not found on chessgames.com."}
        else:
            return {"error": f"Chessgames.com returned status {response.status_code}."}
            
    except Exception as e:
        return {"error": str(e)}

def query_endgame_tablebase(fen: str) -> dict:
    """
    Queries the Syzygy 7-piece endgame tablebase. 
    ONLY use this tool if there are 7 or fewer pieces total on the board!
    """
    try:
        url = f"http://tablebase.lichess.ovh/standard?fen={fen}"
        response = requests.get(url).json()
        
        return {
            "status": response.get("category"), # 'win', 'loss', or 'draw'
            "dtm": response.get("dtm"),         # Distance to mate (number of moves)
            "best_move": response.get("moves", [{}])[0].get("san", "None")
        }
    except Exception as e:
        return {"error": str(e)}

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

# Dynamic prompt injection using the configured Elo
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
3. **Generate:** For EACH critical moment, you MUST use the `get_fen_from_moves` tool to calculate the exact FEN of the board.
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
   - IMPORTANT FORMATTING: Do NOT literally print the JSON keys like "stockfish_refutation_line" in your response. Weave the moves naturally into your explanation.
3. **Speak in Concepts:** At {USER_ELO} Elo, a simple, solid plan is better than a complex engine sequence. Prioritize explaining concepts like King Safety, Piece Activity, controlling the center, and identifying undefended pieces.
4. **Grounding in Theory:** If `get_opening_repertoire` provides a `theoretical_opening_name`, explicitly state it. Use your deep internal knowledge of chess theory to briefly explain the classical Grandmaster plans. Then, compare the `top_master_moves` against the `typical_amateur_moves` to show the student how their opponents are likely to deviate from GM theory and how to punish it!
5. **Handling Hypotheticals:** If the user asks "What if I play [Move]?", "Why can't I play [Move]?", or suggests an alternative idea, IMMEDIATELY use the `analyze_position` tool and pass their proposed move sequence into the `moves_to_apply` parameter. Use the returned `expected_engine_line` to prove exactly how their idea would be mathematically punished or rewarded!
6. **Time Management (Clock Analysis):** If the provided PGN contains clock timestamps (e.g., `[%clk 0:10:00]`), you MUST evaluate the student's time management on their critical mistakes. 
   - Calculate the estimated time spent by comparing the clock on their blunder to the clock on their *previous* turn (factor in any time increment if noted in the headers). 
   - **"Hope Chess" (Impulse Blunders):** If they make a critical blunder or miss a tactic in under 10 seconds, explicitly call them out! Tell them: "You only spent X seconds on this move. You played 'Hope Chess' instead of calculating."
   - **"Freezing" (Overthinking):** If they spend over 60 seconds on a completely forced move (like escaping a check or making an obvious recapture), warn them about "freezing" and hallucinating phantom threats.
   - ALWAYS weave the exact time they spent into your explanation (e.g., "You played this blunder in only 6 seconds!").
7. **Positive Reinforcement:** 
   - If the student found a move tagged with `[⭐ ONLY MOVE FOUND!]`, explicitly praise them for navigating a tactical minefield where any other move would lose.
   - If a move is tagged with `[🛡️ DODGED HUMAN TRAP!]`, explicitly praise them for resisting temptation! Explain *why* the natural human impulse (the move Maia predicted) was a dangerous trap, and how the student's actual move brilliantly bypassed it.
   - 🛑 CRITICAL FORMATTING RULE: The tags `[⭐ ONLY MOVE FOUND!]` and `[🛡️ DODGED HUMAN TRAP!]` are secret backend metadata for your eyes only. NEVER literally print these bracketed tags or emojis in your final text response to the student! Instead, translate the concept into natural, conversational human language (e.g., "You found the absolute only move on the board to save the game!", or "Maia predicted most players would play X, but you successfully dodged the trap!").

### 📝 FORMATTING & RULES OF ENGAGEMENT
- **Current State:** The system automatically appends the current board state (FEN) to every message the user sends. You ALWAYS know the current board state.
- **Endgames:** If there are 7 or fewer pieces on the board, immediately use `query_endgame_tablebase`.
- **MANDATORY VISUALIZATIONS & LINES:** The `analyze_position` tool will now automatically return a `mandatory_board_visual` string and an `expected_engine_line`. 
  - You MUST copy and paste the exact `mandatory_board_visual` string into your chat response whenever you analyze a position. NEVER discuss a position without showing the board! DO NOT wrap the board in markdown code blocks.
  - Use the `expected_engine_line` to definitively prove *why* a move is bad by showing the student how the opponent would mathematically punish them!
  - IMPORTANT FORMATTING: Do NOT literally print the JSON key "expected_engine_line" in your response. Weave the moves naturally into your explanation (e.g., "The engine calculates the following continuation:", or "Stockfish's optimal line is:").
- **CHESS NOTATION FORMATTING (STRICT UI REQUIREMENT):**
  - You MUST wrap EVERY single chess square, individual move, and move sequence in markdown backticks to render them as `inline code` (e.g., `e4`, `Nf3`, `1. d4 d5`).
  - DO NOT wrap plain English piece names in backticks! Write "your King" or "the Pawn", NEVER "your `King`" or "the `Pawn`". ONLY use backticks for formal algebraic notation and squares.
  - Our terminal UI relies entirely on these backticks to generate the visual highlight boxes. 
  - NEVER use standard bold (`**`) or plain text for anything related to chess notation.
  - ✅ DO THIS: Because your Knight is on `c3`, it no longer defends the `b2` pawn. 
  - ❌ NEVER DO THIS: Because your `Knight` is on `c3`, it no longer defends the `b2` `Pawn`.
  - ❌ NEVER DO THIS: Because your Knight is on **c3**, it no longer defends the **b2** pawn.
- Break up your text into easily readable paragraphs or bullet points.
- Even if you are using the Socratic approach with the user, NEVER use the word "Socratic" in your output (e.g. Don't say "Socratic Question:", just say "Question").
"""

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    tools=[
        get_fen_from_moves, fetch_lichess_game_by_id, 
        fetch_chessgames_com_game,
        query_endgame_tablebase, analyze_position, predict_human_move, get_opening_repertoire
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
        if len(parts) < 2:
            return None, "Please provide a file path: /load_pgn game.pgn"
            
        file_path = parts[1].strip()
        if not os.path.exists(file_path):
             return None, f"Could not find the file: {file_path}"
             
        try:
            with open(file_path, "r") as f:
                raw_pgn_text = f.read()
                
            # RUN SCANNER
            enriched_pgn = scan_game_for_critical_moments(raw_pgn_text)
            
            # Parse it with python-chess just to update our current_board state
            # (We use StringIO to safely parse the newly generated string)
            game = chess.pgn.read_game(io.StringIO(enriched_pgn))
            if game:
                current_board = game.end().board()
                
            prompt = f"I have loaded a local game. Here is the full PGN:\n\n{enriched_pgn}\n\nPlease analyze my play and my time management."
            return prompt, f"[green]Successfully loaded {file_path}. Processing game...[/green]"
        except Exception as e:
             return None, f"Error reading PGN: {e}"

    elif user_input.startswith("/move"):
        parts = user_input.split(" ", 1)
        if len(parts) < 2:
            return None, "[red]Please provide a move: /move e4[/red]"
            
        move_str = parts[1].strip()
        try:
            current_board.push_san(move_str)
            prompt = f"I have played {move_str}. The new position FEN is: {current_board.fen()}. Please evaluate."
            return prompt, f"[green]Played {move_str}. Board updated.[/green]"
        except ValueError:
            return None, f"[red]Invalid or illegal move: {move_str}[/red]"
             
    # --- NEW: IF THE USER PASTED A RAW PGN INTO THE TERMINAL ---
    # We pass their raw input through the scanner before handing it to the LLM.
    processed_input = scan_game_for_critical_moments(user_input)

    # Generate readable algebraic move history for live sparring
    try:
        history_str = chess.Board().variation_san(current_board.move_stack) if current_board.move_stack else "Starting Position"
    except Exception:
        history_str = "Unknown"

    enriched_input = (
        f"{processed_input}\n\n"
        f"[System Context for Coach: Current FEN: {current_board.fen()} | Move History: {history_str}]"
    )
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
        # Print a blank line using rich for spacing
        console.print() 
        
        # Use prompt_toolkit for input, with cross-session up-arrow history!
        user_input = prompt(
            HTML("<b><ansiblue>You:</ansiblue></b> "), 
            history=FileHistory(CLI_HISTORY_FILE)
        )
        
        if user_input.strip().lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue
            
        processed_input, sys_message = process_user_input(user_input)
        
        if sys_message and not processed_input:
            console.print(sys_message)
            continue
        elif sys_message:
            console.print(sys_message)
            
        with console.status("[bold yellow]Coach is thinking...[/bold yellow]", spinner="dots"):
            try:
                response = chat.send_message(processed_input)
                
                # Parallel Function Calling Handling Loop
                while True:
                    f_calls = []
                    # Safely extract all function calls from the response
                    if response.function_calls:
                        f_calls = response.function_calls
                    elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for p in response.candidates[0].content.parts:
                            if getattr(p, 'function_call', None): f_calls.append(p.function_call)
                            elif getattr(p, 'functionCall', None): f_calls.append(p.functionCall)
                                
                    # If no tools were requested, break the loop and print the text
                    if not f_calls: break 
                        
                    tool_parts = []
                    
                    # Iterate through EVERY tool the AI asked for in this turn
                    for fn_call in f_calls:
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen", "")
                        result = {}
                        
                        if fn_call.name == "get_fen_from_moves":
                            console.print("[dim cyan]↳ Calculating FEN from move sequence...[/dim cyan]")
                            moves = args_dict.get("moves", [])
                            starting_fen = args_dict.get("starting_fen", fen)
                            result = get_fen_from_moves(starting_fen, moves)

                        elif fn_call.name == "fetch_lichess_game_by_id":
                            console.print("[dim cyan]↳ Fetching game from Lichess...[/dim cyan]")
                            game_id = args_dict.get("game_id", "")
                            result = fetch_lichess_game_by_id(game_id)
                            
                        elif fn_call.name == "fetch_chessgames_com_game":
                            console.print("[dim cyan]↳ Fetching Master Game from Chessgames.com...[/dim cyan]")
                            game_id = args_dict.get("game_id", "")
                            result = fetch_chessgames_com_game(game_id)

                        elif fn_call.name == "query_endgame_tablebase":
                            console.print("[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]")
                            result = query_endgame_tablebase(fen)

                        elif fn_call.name == "analyze_position":
                            console.print("[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]")
                            moves = args_dict.get("moves_to_apply", None)
                            result = analyze_position(fen, moves_to_apply=moves, depth=int(args_dict.get("depth", 15)))
                        
                        elif fn_call.name == "predict_human_move":
                            console.print(f"[dim magenta]↳ Asking Maia what a {USER_ELO} human would do...[/dim magenta]")
                            result = predict_human_move(fen)
                            
                        elif fn_call.name == "get_opening_repertoire":
                            console.print("[dim yellow]↳ Checking Lichess Opening Databases...[/dim yellow]")
                            result = get_opening_repertoire(fen)
                            
                        # Package the specific tool result into a Part
                        tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))
                    
                    # Return ALL tool execution results to Gemini simultaneously
                    response = chat.send_message(tool_parts)
                if response.text:
                    console.print("\n[bold purple]Coach Gemini:[/bold purple]")
                    console.print(Markdown(response.text))
                else:
                    console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")

            except Exception as e:
                console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

except KeyboardInterrupt: 
    pass
finally:
    save_history(chat)
    
    # Gracefully shut down the background engine threads so the terminal doesn't hang!
    if stockfish_engine:
        stockfish_engine.quit()
    if maia_engine:
        maia_engine.quit()
        
    console.print(f"\n[bold dim]Conversation saved to {HISTORY_FILE}. Goodbye![/bold dim]")