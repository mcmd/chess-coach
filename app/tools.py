import uuid
import chess
import requests
import io
import chess.pgn
import chess.engine
from rich.panel import Panel
from app.ui import console
from app.scanner import scan_game_for_critical_moments
from app.concepts import extract_board_features

class CoachTools:
    def __init__(self, state, timeline, engines):
        self.state = state
        self.timeline = timeline
        self.engines = engines

    def execute_coach_move(self, move_san: str) -> dict:
        if not self.state.sparring_mode:
            return {"error": "Sparring mode is DISABLED. Tell the user to type /spar to enable it."}
        
        try:
            self.timeline.play_san(move_san)
            return {"success": True, "new_fen": self.state.board.fen(), "message": f"Successfully played {move_san}."}
        except ValueError as e:
            legal_moves_str = ", ".join([self.state.board.san(m) for m in self.state.board.legal_moves])
            return {
                "error": f"Logic Tutor Intervention: Move '{move_san}' is ILLEGAL in this position.",
                "causal_trace": str(e),
                "pedagogical_hint": "The piece is either pinned, blocked, or your King is in check.",
                "required_action": f"To repair your plan, you MUST choose from these mathematically verified valid moves: [{legal_moves_str}]"
            }

    def submit_diagnosis(self, opponent_last_move_delta: str, primary_task: str, error_diagnosis: str, 
                         weakest_reasoning_claim: str, lowest_confidence_score: float, preventative_habit: str) -> dict:
        
        # 💥 3A. BIG BUG FIX (Backend): Prevent dossier bloat for the same position
        current_fen = self.state.board.fen()
        if any(d.get("fen") == current_fen for d in self.state.diagnostics):
            return {"success": True, "message": "Diagnosis already logged for this position. Do not call this tool again."}

        if float(lowest_confidence_score) < 0.90: error_diagnosis = "none"
            
        diagnosis = {
            "fen": current_fen, # Save the FEN to track duplicates
            "opponent_last_move_delta": opponent_last_move_delta[:150], 
            "primary_task": primary_task,
            "error_diagnosis": error_diagnosis,
            "weakest_reasoning_claim": weakest_reasoning_claim[:150], 
            "lowest_confidence_score": float(lowest_confidence_score), 
            "preventative_habit": preventative_habit[:200] 
        }
        self.state.diagnostics.append(diagnosis)
        
        console.print()
        console.print(Panel(
            f"[dim white]1. ∆ Opponent Move:[/dim white] [dim yellow]{diagnosis['opponent_last_move_delta']}[/dim yellow]\n"
            f"[dim white]2. Primary Task:[/dim white]    [dim cyan]{diagnosis['primary_task']}[/dim cyan]\n"
            f"[dim white]3. Error Diagnosis:[/dim white] [dim red]{diagnosis['error_diagnosis']}[/dim red]\n"
            f"[dim white]4. Weakest Claim:[/dim white]   [dim magenta]\"{diagnosis['weakest_reasoning_claim']}\" (Conf: {diagnosis['lowest_confidence_score']:.2f})[/dim magenta]\n"
            f"[dim white]5. Required Habit:[/dim white]  [dim green]{diagnosis['preventative_habit']}[/dim green]",
            title="[dim italic]Coach's Diagnostic Taxonomy[/dim italic]",
            border_style="dim blue", expand=False
        ))
        
        return {"success": True}
        
    def analyze_position(self, fen: str, moves_to_apply: list[str] = None, depth: int = 20, pov: str = "white") -> dict:
        try:
            safe_depth = max(1, min(int(depth), 25)) 
            board = chess.Board(fen)
            
            # --- HELPER: Generate Unique Visual Tokens ---
            def get_visual_token(b):
                raw = b.unicode(empty_square="·")
                lines = raw.split('\n')
                labeled = []
                if pov.lower() == "black":
                    lines.reverse() 
                    for i, line in enumerate(lines):
                        reversed_chars = " ".join(line.split(" ")[::-1])
                        labeled.append(f"{i+1} │ {reversed_chars} │")
                    labeled.append("  └" + "─" * 17 + "┘")
                    labeled.append("    h g f e d c b a")
                else:
                    for i, line in enumerate(lines):
                        labeled.append(f"{8-i} │ {line} │")
                    labeled.append("  └" + "─" * 17 + "┘")
                    labeled.append("    a b c d e f g h")
                
                visual = "```text\n" + "\n".join(labeled) + "\n```"
                if len(self.state.board_visuals) > 50:
                    self.state.board_visuals.pop(next(iter(self.state.board_visuals)))
                
                # Your brilliant UUID addition!
                render_id = str(uuid.uuid4())[:8] 
                tkn = f"[[BOARD_{render_id}]]"
                self.state.board_visuals[tkn] = visual
                return tkn

            # 📸 1. Snapshot BEFORE moves (The Past)
            token_before = get_visual_token(board)
            
            token_after = None
            if moves_to_apply:
                for move_str in moves_to_apply[:10]: 
                    board.push_san(move_str)
                # 📸 2. Snapshot AFTER moves (The Future)
                token_after = get_visual_token(board)
                
            infos = self.engines.stockfish.analyse(board, chess.engine.Limit(depth=safe_depth), multipv=2)
            
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

            legal_moves_str = ", ".join([board.san(m) for m in board.legal_moves])
            
            # --- 3. EXTRACT FEATURES ---
            features = extract_board_features(board)

            return {
                "top_stockfish_lines": top_lines, 
                "board_token_before_moves": token_before, # <-- Hand the AI BOTH tokens!
                "board_token_after_moves": token_after,
                "mathematically_legal_moves": legal_moves_str,
                "positional_and_safety_features": features
            }
        except Exception as e: return {"error": str(e)}

    def predict_human_move(self, fen: str) -> dict:
        if not self.engines.maia: return {"error": "Maia engine not loaded."}
        try:
            board = chess.Board(fen)
            result = self.engines.maia.play(board, chess.engine.Limit(nodes=1))
            if not result.move: return {"predicted_human_move": "None"}
                
            maia_move_san = board.san(result.move)
            board.push(result.move)
            
            info = self.engines.stockfish.analyse(board, chess.engine.Limit(depth=25))
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

    def get_opening_repertoire(self, fen: str) -> dict:
        def _safe_get_json(url: str, params: dict = None) -> dict:
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"error": f"Network request failed: {e}"}

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
            valid_brackets = [str(b) for b in brackets if abs(b - self.engines.config.user_elo) <= 250]
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

    def fetch_lichess_game_by_id(self, game_id: str) -> dict:
        try:
            # 🐛 2B. BONUS FIX: Lichess IDs are 8 chars. Strip the trailing player perspective characters.
            game_id = game_id[:8] 
            url = f"https://lichess.org/game/export/{game_id}"
            params = {"evals": "true", "clocks": "true", "accuracy": "true", "opening": "true", "literate": "true", "tags": "true"}
            
            r = requests.get(url, params=params, headers={"Accept": "application/x-chess-pgn"}, timeout=15)
            if r.status_code == 200:
                console.print("[dim yellow]↳ Running Brilliancy Scanner on downloaded Lichess game...[/dim yellow]")
                was_scanned, scanned_pgn = scan_game_for_critical_moments(r.text, self.engines)
                
                try:
                    parsed_game = chess.pgn.read_game(io.StringIO(scanned_pgn))
                    self.timeline.load_game(parsed_game)
                except Exception: pass
                
                if was_scanned:
                    sys_msg = "\n\n[System Info: I have mathematically scanned the PGN above and added Brilliancy tags. Please use them for your analysis.]"
                    return {"pgn": scanned_pgn + sys_msg}
                return {"pgn": scanned_pgn}
            elif r.status_code == 404: return {"error": f"Game with ID '{game_id}' not found."}
            else: return {"error": f"Lichess API returned status {r.status_code}."}
        except Exception as e: return {"error": str(e)}

    def fetch_chessgames_com_game(self, game_id: str) -> dict:
        try:
            url = f"https://www.chessgames.com/njs/api/game/viewPGN/{game_id}"
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/plain, application/x-chess-pgn, */*"}
            
            r = requests.get(url, params={"raw": "true"}, headers=headers, timeout=15)
            if r.status_code == 200:
                pgn_text = r.text.strip()
                if not pgn_text or not pgn_text.startswith("[Event"): return {"error": "Received malformed data."}
                
                console.print("[dim yellow]↳ Running Brilliancy Scanner on downloaded Master game...[/dim yellow]")
                was_scanned, scanned_pgn = scan_game_for_critical_moments(pgn_text, self.engines)
                
                try:
                    parsed_game = chess.pgn.read_game(io.StringIO(scanned_pgn))
                    self.timeline.load_game(parsed_game)
                except Exception: pass
                
                if was_scanned:
                    sys_msg = "\n\n[System Info: I have mathematically scanned the PGN above and added Brilliancy tags. Please use them for your analysis.]"
                    return {"pgn": scanned_pgn + sys_msg}
                return {"pgn": scanned_pgn}
            elif r.status_code == 404: return {"error": f"Game with ID '{game_id}' not found."}
            else: return {"error": f"Chessgames API returned status {r.status_code}."}
        except Exception as e: return {"error": str(e)}

    def query_endgame_tablebase(self, fen: str) -> dict:
        def _safe_get_json(url: str, params: dict = None) -> dict:
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"error": f"Network request failed: {e}"}
        try:
            res = _safe_get_json("https://tablebase.lichess.ovh/standard", {"fen": fen})
            if "error" in res: return res
            return {"status": res.get("category"), "dtm": res.get("dtm"), "best_move": res.get("moves", [{}])[0].get("san", "None")}
        except Exception as e: return {"error": str(e)}

    def get_tool_list(self):
        """Returns the bound methods to be passed to the Gemini API."""
        return [
            self.fetch_lichess_game_by_id, self.fetch_chessgames_com_game, self.query_endgame_tablebase, 
            self.analyze_position, self.predict_human_move, self.get_opening_repertoire, 
            self.execute_coach_move, self.submit_diagnosis
        ]