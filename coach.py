import os
import json
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

CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    console.print(f"[bold red]Error:[/bold red] {CONFIG_FILE} not found. Please create it.")
    exit(1)

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

# Initialize Stockfish
try:
    stockfish = Stockfish(path=STOCKFISH_PATH)
except FileNotFoundError:
    console.print(f"[bold red]Error:[/bold red] Stockfish not found at {STOCKFISH_PATH}.")
    exit(1)

# ======================================================================
# 2. THE TOOLS (STOCKFISH, MAIA, LICHESS EXPLORER)
# ======================================================================

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

def analyze_position(fen: str, depth: int = 15) -> dict:
    """Evaluates a position using Stockfish to find the objective best move."""
    try:
        if not stockfish.is_fen_valid(fen): return {"error": "Invalid FEN"}
        stockfish.set_fen_position(fen)
        stockfish.set_depth(depth)
        return {
            "evaluation": stockfish.get_evaluation(),
            "top_moves": stockfish.get_top_moves(3)
        }
    except Exception as e: return {"error": str(e)}

def predict_human_move(fen: str) -> dict:
    """
    Predicts the most likely move a human player would make using the Maia neural network.
    Dynamically loads the correct weights based on the user's configured Elo.
    """
    try:
        # Round the user's Elo to the nearest 100, clamped between 1100 and 1900
        # Specifically targeting the downloaded weights: 1100, 1500, 1900
        rounded_elo = round(USER_ELO / 100) * 100
        if rounded_elo <= 1300:
            maia_target = 1100
        elif rounded_elo <= 1700:
            maia_target = 1500
        else:
            maia_target = 1900

        weights_file = os.path.join(MAIA_DIR, f"maia-{maia_target}.pb.gz")
        
        if not os.path.exists(weights_file):
            return {"error": f"Weights file {weights_file} not found."}

        # Initialize Lc0 and hide its startup text by pointing stderr to DEVNULL
        engine = chess.engine.SimpleEngine.popen_uci(
            LC0_PATH,
            stderr=subprocess.DEVNULL
        )
        engine.configure({"WeightsFile": weights_file})
        
        board = chess.Board(fen)
        # CRITICAL: Restrict Maia to exactly 1 node to get raw human intuition, not deep calculation
        result = engine.play(board, chess.engine.Limit(nodes=1))
        predicted_move = result.move.uci() if result.move else "None"
        engine.quit()
        
        return {"predicted_human_move": predicted_move, "maia_model_used": maia_target}
    except Exception as e: return {"error": str(e)}

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

def print_board(fen: str) -> dict:
    """
    Generates a visual text representation of the chess board.
    Use this if the user asks to "see the board" or "show the position".
    """
    try:
        board = chess.Board(fen)
        # Using unicode for nice chess piece characters
        visual = board.unicode(empty_square="·")
        return {"board_visual": f"\n{visual}\n"}
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
            return {"pgn": response.text}
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

HISTORY_FILE = "chess_coach_memory.json"

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
2. **Identify:** Read the PGN silently and isolate 2 to 3 critical moments (e.g., blunders, missed tactics, or opening traps).
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
   - If Maia predicts the opponent will blunder or play sub-optimally, tell the student: "At your level, your opponent is highly likely to play [Move] here. How would we set a trap for that?"
3. **Speak in Concepts:** At {USER_ELO} Elo, a simple, solid plan is better than a complex engine sequence. Prioritize explaining concepts like King Safety, Piece Activity, controlling the center, and identifying undefended pieces.
4. **Grounding in Theory:** If `get_opening_repertoire` provides a `theoretical_opening_name`, explicitly state it. Use your deep internal knowledge of chess theory to briefly explain the classical Grandmaster plans. Then, compare the `top_master_moves` against the `typical_amateur_moves` to show the student how their opponents are likely to deviate from GM theory and how to punish it!

### 📝 FORMATTING & RULES OF ENGAGEMENT
- **Current State:** The system automatically appends the current board state (FEN) to every message the user sends. You ALWAYS know the current board state.
- **Endgames:** If there are 7 or fewer pieces on the board, immediately use `query_endgame_tablebase`.
- **CHESS NOTATION FORMATTING (STRICT UI REQUIREMENT):**
  - You MUST wrap EVERY single chess square, piece name, individual move, and move sequence in markdown backticks to render them as `inline code`. 
  - Our terminal UI relies entirely on these backticks to generate the visual highlight boxes. 
  - NEVER use standard bold (`**`) or plain text for anything related to chess notation.
  - ✅ DO THIS: Because your Knight is on `c3`, it no longer defends the `b2` pawn. 
  - ❌ NEVER DO THIS: Because your Knight is on **c3**, it no longer defends the **b2** pawn.
  - ❌ NEVER DO THIS: Because your Knight is on c3, it no longer defends the b2 pawn.
  - ✅ DO THIS: The "Grandmaster" Antidote: `5. Na4!`
  - ❌ NEVER DO THIS: The "Grandmaster" Antidote: **5. Na4!**
  - ✅ DO THIS: `1. d4 d5 2. Nc3 Nf6`
  - ❌ NEVER DO THIS: 1. d4 d5 2. Nc3 Nf6
- Break up your text into easily readable paragraphs or bullet points.
- Even if you are using the Socratic approach with the user, NEVER use the word "Socratic" in your output (e.g. Don't say "Socratic Question:", just say "Question").
"""

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    tools=[
        get_fen_from_moves, print_board, fetch_lichess_game_by_id, 
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
                game = chess.pgn.read_game(f)
                if game is None:
                    return None, "Could not parse PGN file."
                
                current_board = game.end().board()
                fen = current_board.fen()
                
                prompt = f"I have loaded a game. The final position FEN is: {fen}. Please analyze this position."
                return prompt, f"[green]Successfully loaded {file_path}. Processing position...[/green]"
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
             
    # If it's a normal message, secretly append the current FEN so Gemini knows what the board looks like right now
    enriched_input = f"{user_input}\n\n[System Info: The current board FEN is {current_board.fen()}]"
    return enriched_input, None

console.print(Panel.fit(
    f"[bold green]Gemini {GEMINI_MODEL} + Stockfish + Maia Coach (Elo: {USER_ELO})[/bold green]\n"
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
            history=FileHistory("chess_cli_history.txt")
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
                
                # Manual Function Calling Handling Loop
                while True:
                    fn_call = None
                    if response.function_calls: fn_call = response.function_calls[0]
                    elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for p in response.candidates[0].content.parts:
                            if getattr(p, 'function_call', None): fn_call = p.function_call; break
                            if getattr(p, 'functionCall', None): fn_call = p.functionCall; break
                                
                    if not fn_call: break 
                        
                    args_dict = dict(fn_call.args) if fn_call.args else {}
                    fen = args_dict.get("fen", "")
                    
                    if fn_call.name == "get_fen_from_moves":
                        console.print("[dim cyan]↳ Calculating FEN from move sequence...[/dim cyan]")
                        moves = args_dict.get("moves", [])
                        starting_fen = args_dict.get("starting_fen", fen)
                        result = get_fen_from_moves(starting_fen, moves)

                    elif fn_call.name == "print_board":
                        console.print("[dim cyan]↳ Generating board visual...[/dim cyan]")
                        result = print_board(fen)

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
                        result = analyze_position(fen, int(args_dict.get("depth", 15)))
                    
                    elif fn_call.name == "predict_human_move":
                        console.print(f"[dim magenta]↳ Asking Maia what a {USER_ELO} human would do...[/dim magenta]")
                        result = predict_human_move(fen)
                        
                    elif fn_call.name == "get_opening_repertoire":
                        console.print("[dim yellow]↳ Querying Master & Amateur Opening Databases...[/dim yellow]")
                        result = get_opening_repertoire(fen)
                        
                    else: break
                    
                    # Return tool execution result to Gemini
                    tool_part = types.Part.from_function_response(name=fn_call.name, response={"result": result})
                    response = chat.send_message(tool_part)

                if response.text:
                    console.print("\n[bold purple]Coach Gemini:[/bold purple]")
                    console.print(Markdown(response.text))
                else:
                    console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")

            except Exception as e:
                console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

except KeyboardInterrupt: pass
finally:
    save_history(chat)
    console.print("\n[bold dim]Conversation saved. Goodbye![/bold dim]")