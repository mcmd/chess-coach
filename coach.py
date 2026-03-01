import os
import json
import requests
import chess.pgn
import chess
from google import genai
from google.genai import types
from stockfish import Stockfish
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()

console = Console()

# ======================================================================
# 1. SETUP & CONFIGURATION
# ======================================================================

STOCKFISH_PATH = "/usr/bin/stockfish" # Updated to your Linux/WSL path

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    console.print("[bold red]Error:[/bold red] GEMINI_API_KEY environment variable not set.")
    exit(1)

client = genai.Client(api_key=api_key)

# We keep a persistent board state for conversational back-and-forth
current_board = chess.Board()

# ======================================================================
# 2. TOOLS
# ======================================================================

try:
    stockfish = Stockfish(path=STOCKFISH_PATH)
except FileNotFoundError:
    console.print(f"[bold red]Error:[/bold red] Stockfish not found at {STOCKFISH_PATH}.")
    exit(1)

def get_fen_from_moves(moves: list[str]) -> dict:
    """
    Applies a list of algebraic moves (e.g., ["e4", "c5", "Nf3"]) to the starting position 
    and returns the resulting FEN string. Use this when the user suggests a sequence of moves 
    and you need the exact FEN to analyze it.
    """
    temp_board = chess.Board()
    try:
        for move_str in moves:
            temp_board.push_san(move_str)
        return {"fen": temp_board.fen()}
    except ValueError as e:
        return {"error": f"Invalid move sequence: {e}. Current FEN: {temp_board.fen()}"}

def analyze_position(fen: str, depth: int = 15) -> dict:
    """
    Evaluates a chess position using the Stockfish engine.
    
    Args:
        fen: The FEN string of the current board position.
        depth: The depth to which Stockfish should calculate (default 15).
    """
    try:
        if not stockfish.is_fen_valid(fen):
            return {"error": "The provided FEN string is invalid."}
            
        stockfish.set_fen_position(fen)
        stockfish.set_depth(depth)
        
        evaluation = stockfish.get_evaluation() 
        top_moves = stockfish.get_top_moves(3) 
        
        return {
            "evaluation": evaluation,
            "top_moves": top_moves
        }
    except Exception as e:
        return {"error": f"Error running Stockfish: {str(e)}"}

def get_opening_explorer_data(fen: str) -> dict:
    """
    Looks up the current FEN in the Lichess Masters database to see standard opening moves.
    Use this when the user is in the first 15 moves of the game.
    """
    try:
        url = f"https://explorer.lichess.ovh/masters?fen={fen}"
        response = requests.get(url).json()
        
        # Extract the opening name and the top 4 most popular master moves
        opening_name = response.get("opening", {}).get("name", "Unknown Opening") if response.get("opening") else "Unknown Opening"
        top_moves = []
        for move in response.get("moves",[])[:4]:
            top_moves.append({
                "move": move["san"],
                "white_wins": move["white"],
                "draws": move["draws"],
                "black_wins": move["black"]
            })
            
        return {"opening_name": opening_name, "top_master_moves": top_moves}
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
                history_contents =[]
                for item in data:
                    parts =[types.Part.from_text(text=p["text"]) for p in item["parts"]]
                    history_contents.append(types.Content(role=item["role"], parts=parts))
                return history_contents
            except json.JSONDecodeError:
                return[]
    return[]

def save_history(chat_session):
    """Saves only the textual conversation, filtering out raw tool data to keep memory clean."""
    serializable_history =[]
    
    for message in chat_session.get_history():
        parts =[]
        if message.parts:
            for part in message.parts:
                if getattr(part, 'text', None):
                    parts.append({"text": part.text})
        if parts:
            serializable_history.append({"role": message.role, "parts": parts})
            
    with open(HISTORY_FILE, "w") as f:
        json.dump(serializable_history, f)

system_instruction = """
You are a Grandmaster-level chess coach. You help the user analyze their games. 

You have six tools available:
1. `get_fen_from_moves`: If the user says "what if I play e4 c5 Nf3", you MUST use this tool to generate the correct FEN string first. Do NOT try to guess the FEN in your head.
2. `analyze_position`: Use this to get the objective Stockfish evaluation by passing it the FEN string.
3. `get_opening_explorer_data`: Use this to query the Lichess Masters database for standard opening moves if the position is in the opening phase (roughly the first 10-15 moves).
4. `print_board`: Use this to generate a visual representation of the chess board if the user asks to "see the board", "show the position", etc. Make sure you output the result directly.
5. `fetch_lichess_game_by_id`: Use this to fetch a specific game's details (PGN, evaluation, clock times) from Lichess using its game ID.
6. `query_endgame_tablebase`: Use this ONLY when there are 7 or fewer pieces remaining on the board to get perfect endgame play (win/draw/loss and distance to mate).

When discussing a specific position, you must ALWAYS use either Stockfish, the Opening Explorer, or the Endgame Tablebase to get objective data. 
Once you have the data, translate it into plain, conversational English. 
Explain WHY a move is good in terms of chess principles. 
Do NOT just read the raw numbers back to the user. Be encouraging but honest!
"""

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    tools=[get_fen_from_moves, analyze_position, get_opening_explorer_data, print_board, fetch_lichess_game_by_id, query_endgame_tablebase],
    # explicitly DISABLE automatic tool calling so we can handle it flawlessly ourselves
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) 
)

# Initialize with Gemini 3.1 Pro Preview
chat = client.chats.create(
    model='gemini-3.1-pro-preview',
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
             
    # If it's a normal message, secretly append the current FEN so Gemini knows what the board looks like right now
    enriched_input = f"{user_input}\n\n[System Info: The current board FEN is {current_board.fen()}]"
    return enriched_input, None

console.print(Panel.fit(
    "[bold green]Gemini 3.1 + Stockfish Chess Coach[/bold green]\n"
    "Type [bold cyan]/load_pgn filename.pgn[/bold cyan] to load a game.\n"
    "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to save and close.",
    border_style="green"
))

try:
    while True:
        user_input = console.input("\n[bold blue]You:[/bold blue] ")
        
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        processed_input, sys_message = process_user_input(user_input)
        
        if sys_message and not processed_input:
            console.print(sys_message)
            continue
        elif sys_message:
            console.print(sys_message)
            
        with console.status("[bold yellow]Coach is thinking...[/bold yellow]", spinner="dots"):
            try:
                response = chat.send_message(processed_input)
                
                # --- MANUAL FUNCTION CALLING LOOP ---
                while True:
                    fn_call = None
                    
                    if response.function_calls:
                        fn_call = response.function_calls[0]
                    elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for p in response.candidates[0].content.parts:
                            if getattr(p, 'function_call', None): 
                                fn_call = p.function_call; break
                            if getattr(p, 'functionCall', None): 
                                fn_call = p.functionCall; break
                                
                    if not fn_call:
                        break # Exit the loop if Gemini isn't asking to use a tool
                        
                    if fn_call.name == "get_fen_from_moves":
                        console.print("[dim cyan]↳ Calculating FEN from move sequence...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        moves = args_dict.get("moves", [])
                        
                        result = get_fen_from_moves(moves)
                            
                        tool_part = types.Part.from_function_response(
                            name="get_fen_from_moves",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)
                        
                    elif fn_call.name == "analyze_position":
                        console.print("[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen")
                        depth = args_dict.get("depth", 15)
                        
                        if not fen:
                            result = {"error": "Missing FEN string in tool call."}
                        else:
                            result = analyze_position(fen, int(depth))
                        
                        tool_part = types.Part.from_function_response(
                            name="analyze_position",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)
                        
                    elif fn_call.name == "get_opening_explorer_data":
                        console.print("[dim cyan]↳ Querying Lichess Opening Explorer...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen")
                        
                        if not fen:
                            result = {"error": "Missing FEN string in tool call."}
                        else:
                            result = get_opening_explorer_data(fen)
                            
                        tool_part = types.Part.from_function_response(
                            name="get_opening_explorer_data",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)
                        
                    elif fn_call.name == "print_board":
                        console.print("[dim cyan]↳ Generating board visual...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen")
                        
                        if not fen:
                            result = {"error": "Missing FEN string in tool call."}
                        else:
                            result = print_board(fen)
                            
                        tool_part = types.Part.from_function_response(
                            name="print_board",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)

                    elif fn_call.name == "fetch_lichess_game_by_id":
                        console.print("[dim cyan]↳ Fetching game from Lichess...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        game_id = args_dict.get("game_id")
                        
                        if not game_id:
                            result = {"error": "Missing game_id in tool call."}
                        else:
                            result = fetch_lichess_game_by_id(game_id)
                            
                        tool_part = types.Part.from_function_response(
                            name="fetch_lichess_game_by_id",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)
                        
                    elif fn_call.name == "query_endgame_tablebase":
                        console.print("[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]")
                        
                        args_dict = dict(fn_call.args) if fn_call.args else {}
                        fen = args_dict.get("fen")
                        
                        if not fen:
                            result = {"error": "Missing FEN string in tool call."}
                        else:
                            result = query_endgame_tablebase(fen)
                            
                        tool_part = types.Part.from_function_response(
                            name="query_endgame_tablebase",
                            response={"result": result}
                        )
                        response = chat.send_message(tool_part)
                        
                    else:
                        break
                # --- END OF LOOP ---

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
    console.print("\n[bold dim]Conversation saved. Goodbye![/bold dim]")