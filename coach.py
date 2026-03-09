import io
import os
import re
import chess.pgn
import chess
from google import genai
from google.genai import types
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import prompt, HTML
from prompt_toolkit.history import FileHistory
from dotenv import load_dotenv
from app.ui import console

from app.config import load_config
from app.state import SessionState, TimelineManager
from app.engines import EngineManager
from app.scanner import scan_game_for_critical_moments
from app.memory import load_history, save_history, update_dossier
from app.tools import CoachTools
from app.prompts import create_chat_session

load_dotenv()

def main():
    config = load_config()
    client = genai.Client(api_key=config.api_key)
    
    # --- 1. DEPENDENCY INJECTION ---
    app_state = SessionState()
    timeline = TimelineManager(app_state)
    
    engines = EngineManager(config)
    engines.start()
    
    coach_tools = CoachTools(app_state, timeline, engines)
    chat = create_chat_session(client, config, coach_tools, load_history(config.history_file))

    def process_user_input(user_input):
        nonlocal chat 
        
        if user_input.startswith("/load_pgn"):
            parts = user_input.split(" ", 1)
            if len(parts) < 2: return None, "Please provide a file path: /load_pgn game.pgn"
                
            file_path = parts[1].strip()
            if not os.path.exists(file_path): return None, f"Could not find the file: {file_path}"
                 
            try:
                with open(file_path, "r") as f:
                    raw_pgn_text = f.read()
                    
                was_scanned, enriched_pgn = scan_game_for_critical_moments(raw_pgn_text, engines)
                
                game = chess.pgn.read_game(io.StringIO(enriched_pgn))
                timeline.load_game(game)
                    
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
                move_san, time_spent_str = timeline.step_forward()
                prompt = f"I have played the next move from the game: {move_san}.{time_spent_str} The new position FEN is: {app_state.board.fen()}. Please evaluate."
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
            engines.get_cached_eval.cache_clear() 
            chat = create_chat_session(client, config, coach_tools, []) 
            if os.path.exists(config.history_file):
                try: os.remove(config.history_file)
                except Exception: pass
            return None, "[bold green]✨ Session memory, board state, and timeline wiped clean![/bold green]"

        elif user_input.startswith("/memorize"):
            if not app_state.diagnostics:
                return None, "[yellow]No diagnostics generated this session to memorize.[/yellow]"
            
            update_dossier(app_state.diagnostics, config.dossier_file)
            timeline.reset()
            engines.get_cached_eval.cache_clear() 
            chat = create_chat_session(client, config, coach_tools, [])
            if os.path.exists(config.history_file):
                try: os.remove(config.history_file)
                except: pass
                
            console.clear()
            return None, f"[bold green]✨ Session memorized! Dossier updated. Memory wiped and screen cleared.[/bold green]"
             
        # --- HANDLE RAW PGN PASTES ---
        was_scanned, processed_input = scan_game_for_critical_moments(user_input, engines)

        is_pgn = False
        if "1." in user_input or "[Event" in user_input:
            try:
                game = chess.pgn.read_game(io.StringIO(user_input))
                timeline.load_game(game)
                is_pgn = True
            except Exception: pass 

        if is_pgn:
            processed_input = f"I have pasted a game. The new position FEN is: {app_state.board.fen()}. Please analyze my play."

        # 🐛 THE HISTORY BUG FIX: Properly recreate the board state from 5 moves ago to generate SAN
        try: 
            if app_state.board.move_stack:
                temp_board = app_state.board.copy()
                num_moves = min(5, len(temp_board.move_stack))
                show_moves = temp_board.move_stack[-num_moves:]
                for _ in range(num_moves): temp_board.pop()
                history_str = temp_board.variation_san(show_moves)
            else:
                history_str = "Starting Position"
        except Exception: history_str = "Unknown"
        
        legal_moves_str = ", ".join([app_state.board.san(m) for m in app_state.board.legal_moves])
        enriched_input = f"{processed_input}\n\n[System Context: Current FEN: {app_state.board.fen()} | Legal Moves: {legal_moves_str} | Move History: {history_str}]"
        if was_scanned: enriched_input += "\n[System Info: I have automatically added Brilliancy and Evaluation tags to the backend PGN.]"
        return enriched_input, None

    profile_text = f" (Profile: {config.profile})" if config.profile else ""
    console.print(Panel.fit(f"[bold green]Gemini Coach + Logic Tutor (Elo: {config.user_elo}){profile_text}[/bold green]\nType [bold red]exit[/bold red] to save and close.", border_style="green"))

    # --- MAIN CHAT LOOP ---
    try:
        while True:
            console.print() 
            user_input = prompt(HTML("<b><ansiblue>You:</ansiblue></b> "), history=FileHistory(config.cli_history_file))
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
                    
                    # 🐛 TEXT-SWALLOWING FIX: ACCUMULATE TEXT ACROSS ALL TOOL TURNS
                    full_response_text = ""
                    
                    while True:
                        # 🐛 1. SMALL BUG FIX: Safely extract text to silence the SDK warning
                        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            for part in response.candidates[0].content.parts:
                                if getattr(part, 'text', None):
                                    full_response_text += part.text + "\n"

                        # CHECK FOR TOOLS
                        f_calls = []
                        if response.function_calls: f_calls = response.function_calls
                        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            for p in response.candidates[0].content.parts:
                                if getattr(p, 'function_call', None): f_calls.append(p.function_call)
                                elif getattr(p, 'functionCall', None): f_calls.append(p.functionCall)
                                    
                        # IF NO TOOLS, WE ARE DONE
                        if not f_calls: break 
                            
                        # EXECUTE TOOLS
                        tool_parts = []
                        for fn_call in f_calls:
                            args_dict = dict(fn_call.args) if fn_call.args else {}
                            
                            # 🐛 MEDIUM BUG FIX: Restore the beautiful UI tool messages!
                            tool_msgs = {
                                "fetch_lichess_game_by_id": "[dim cyan]↳ Fetching game from Lichess...[/dim cyan]",
                                "fetch_chessgames_com_game": "[dim cyan]↳ Fetching Master Game from Chessgames.com...[/dim cyan]",
                                "query_endgame_tablebase": "[dim cyan]↳ Querying Syzygy Endgame Tablebase...[/dim cyan]",
                                "analyze_position": "[dim cyan]↳ Interfacing with local Stockfish...[/dim cyan]",
                                "predict_human_move": f"[dim magenta]↳ Asking Maia what a {config.user_elo} human would do...[/dim magenta]",
                                "get_opening_repertoire": "[dim yellow]↳ Checking Lichess Opening Databases...[/dim yellow]",
                                "execute_coach_move": "[dim green]↳ The Logic Tutor is evaluating the move...[/dim green]",
                                "submit_diagnosis": "[dim magenta]↳ Recording structural diagnosis to backend...[/dim magenta]"
                            }
                            
                            if fn_call.name in tool_msgs:
                                console.print(tool_msgs[fn_call.name])
                            else:
                                console.print(f"[dim cyan]↳ Executing {fn_call.name}...[/dim cyan]")

                            if hasattr(coach_tools, fn_call.name):
                                method = getattr(coach_tools, fn_call.name)
                                result = method(**args_dict)
                            else:
                                result = {"error": f"Tool {fn_call.name} not found."}
                                
                            tool_parts.append(types.Part.from_function_response(name=fn_call.name, response={"result": result}))
                        
                        response = chat.send_message(tool_parts)
                        
                    # --- CLEAN RESPONSE RENDERING ---
                    if full_response_text:
                        output_text = full_response_text.strip()
                        
                        # Find ALL <think> blocks via finditer to capture everything
                        think_matches = list(re.finditer(r"<think>(.*?)</think>", output_text, re.DOTALL | re.IGNORECASE))
                        if think_matches:
                            thought_process = "\n\n".join([m.group(1).strip() for m in think_matches])
                            output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL | re.IGNORECASE).strip()
                            console.print(Panel(Markdown(thought_process), title="[dim italic]🧠 Coach's Internal Monologue[/dim italic]", border_style="dim white", expand=False))
                        
                        # 🐛 BIG BUG FIX: The scrubber no longer deletes '...' chess notation!
                        output_text = re.sub(r"^(?:(?:Output|Done|End|Yes|I will output now)\.\s*)+", "", output_text, flags=re.IGNORECASE).strip()

                        for token, visual in app_state.board_visuals.items():
                            output_text = output_text.replace(token, f"\n{visual}\n")
                            
                        if output_text: 
                            console.print("\n[bold purple]Coach Gemini:[/bold purple]")
                            console.print(Markdown(output_text))
                    else: 
                        console.print("\n[bold red]Error: Received no text response from the model.[/bold red]")
                        
                except Exception as e: console.print(f"\n[bold red]An API error occurred:[/bold red] {e}")

    except KeyboardInterrupt: pass
    finally:
        save_history(chat, config.history_file)
        engines.quit()
        console.print(f"\n[bold dim]Conversation saved to {config.history_file}. Goodbye![/bold dim]")

if __name__ == "__main__":
    main()