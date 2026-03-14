import io
import chess
import chess.pgn
import chess.engine
from app.ui import console

def scan_game_for_critical_moments(user_text: str, engines) -> tuple[bool, str]:
    if "1." not in user_text:
        return False, user_text 
        
    try:
        pgn_io = io.StringIO(user_text)
        game = chess.pgn.read_game(pgn_io)
        if not game or len(list(game.mainline_moves())) < 3:
            return False, user_text
            
        console.print("\n[dim yellow]Running 2-Pass Brilliancy Scan...[/dim yellow]")
        
        annotated_games = []
        
        # 💫 UI FIX: Use a static print instead of a nested spinner to stop flickering
        console.print("[bold yellow]⏳ Stockfish & Maia are analyzing critical paths (Takes ~1 min)...[/bold yellow]")
        
        # REMOVED: The 'with console.status' context manager
        pgn_io = io.StringIO(user_text)
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None: break
            
            board = game.board()
            for node in game.mainline():
                played_move = node.move
                if played_move is None: continue 
                
                is_only_move = False
                dodged_trap = False
                maia_san = ""
                
                if engines.stockfish:
                    turn_color = board.turn
                    
                    info_a = engines.stockfish.analyse(board, chess.engine.Limit(depth=15), multipv=3)
                    
                    if info_a and "pv" in info_a[0]:
                        best_move_a = info_a[0]["pv"][0]
                        best_score_a = info_a[0]["score"].pov(turn_color).score(mate_score=100000)
                        second_score_a = info_a[1]["score"].pov(turn_color).score(mate_score=100000) if len(info_a) > 1 and "pv" in info_a[1] else None
                        
                        candidate_only_move = False
                        candidate_trap = False
                        maia_move = None
                        
                        if played_move == best_move_a and second_score_a is not None:
                            if info_a[0]["score"].pov(turn_color).is_mate() and not info_a[1]["score"].pov(turn_color).is_mate() and info_a[0]["score"].pov(turn_color).mate() > 0:
                                candidate_only_move = True
                            elif (best_score_a - second_score_a) >= 150:
                                candidate_only_move = True
                                
                        if engines.maia:
                            from maia2 import inference
                            move_probs, _ = inference.inference_each(
                                engines.maia, 
                                engines.maia_prepared, 
                                board.fen(), 
                                elo_self=engines.config.user_elo, 
                                elo_oppo=engines.config.user_elo
                            )
                            if move_probs:
                                best_move_str = max(move_probs, key=move_probs.get)
                                maia_move = chess.Move.from_uci(best_move_str)
                                if maia_move and maia_move != played_move:
                                    candidate_trap = True
                        
                        if candidate_only_move or candidate_trap:
                            info_b = engines.stockfish.analyse(board, chess.engine.Limit(depth=18), multipv=3)
                            
                            if info_b and "pv" in info_b[0]:
                                best_move_b = info_b[0]["pv"][0]
                                best_score_b = info_b[0]["score"].pov(turn_color).score(mate_score=100000)
                                second_score_b = info_b[1]["score"].pov(turn_color).score(mate_score=100000) if len(info_b) > 1 and "pv" in info_b[1] else None
                                
                                if candidate_only_move and played_move == best_move_b and second_score_b is not None:
                                    if info_b[0]["score"].pov(turn_color).is_mate() and not info_b[1]["score"].pov(turn_color).is_mate() and info_b[0]["score"].pov(turn_color).mate() > 0:
                                        is_only_move = True
                                    elif (best_score_b - second_score_b) >= 200:
                                        is_only_move = True
                                                
                                if candidate_trap and maia_move and maia_move != played_move:
                                    played_score = next((i["score"].pov(turn_color).score(mate_score=100000) for i in info_b if i.get("pv") and i["pv"][0] == played_move), None)
                                    if played_score is None: played_score = engines.eval_move(board, played_move, 0.4, turn_color)
                                        
                                    maia_score = next((i["score"].pov(turn_color).score(mate_score=100000) for i in info_b if i.get("pv") and i["pv"][0] == maia_move), None)
                                    if maia_score is None: maia_score = engines.eval_move(board, maia_move, 0.4, turn_color)
                                        
                                    if played_score is not None and maia_score is not None and best_score_b is not None:
                                        if (best_score_b - played_score) <= 70 and (played_score - maia_score) >= 200:
                                            dodged_trap = True
                                            maia_san = board.san(maia_move)
                
                comment = node.comment if node.comment else ""
                
                fen_before = board.fen()
                
                # 🐛 IDIOT-PROOF FEN FIX: Make the temporal state explicitly readable to the LLM
                move_label = f"{board.fullmove_number}. {board.san(played_move)}" if board.turn == chess.WHITE else f"{board.fullmove_number}... {board.san(played_move)}"
                
                # 🐛 CAPTURE BLINDNESS FIX: Detect the exact piece BEFORE pushing the move
                capture_tag = ""
                if board.is_capture(played_move):
                    if board.is_en_passant(played_move):
                        capture_tag = " [Captured: pawn]"
                    else:
                        cap_piece = board.piece_at(played_move.to_square)
                        if cap_piece:
                            capture_tag = f" [Captured: {chess.piece_name(cap_piece.piece_type)}]"
                
                board.push(played_move)
                
                tag_string = capture_tag
                if is_only_move: tag_string += " [⭐ ONLY MOVE FOUND!]"
                if dodged_trap: tag_string += f" [🛡️ DODGED HUMAN TRAP! Maia predicted {maia_san}]"
                        
                if "[%eval" not in comment and engines.stockfish:
                    score_after = engines.get_cached_eval(board.fen(), 0.25).white()
                    eval_str = f"#{score_after.mate()}" if score_after.is_mate() else f"{score_after.score(mate_score=100000) / 100.0:+.2f}"
                    tag_string += f" [Eval AFTER {move_label}: {eval_str}]"

                if "Board BEFORE" not in comment:
                    tag_string += f' [Board BEFORE {move_label}: "{fen_before}"]'

                if tag_string: node.comment = (comment + tag_string).strip()
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
            annotated_games.append(game.accept(exporter))
                
        if annotated_games:
            console.print("[green]Scan complete![/green]")
            return True, "\n\n".join(annotated_games)
            
    except Exception as e:
        console.print(f"[dim red]Game scan failed: {e}[/dim red]")
        
    return False, user_text