import io
import random
import re

import chess
import chess.engine
import chess.pgn
import requests
from rich.panel import Panel

from app.concepts import extract_board_features
from app.memory import load_lesson_metadata, save_lesson_metadata
from app.motifs import detect_motifs_in_line
from app.pawn_story import analyze_pawn_structure_evolution
from app.render import render_board_token
from app.review import review_game, attach_board_tokens, cp_to_win_pct, MATE_CP
from app.ui import console

# The closed mistake taxonomy. Diagnoses outside this list are rejected so the
# long-term dossier can actually aggregate recurring weaknesses.
ERROR_TAXONOMY = [
    "hanging_piece",
    "missed_tactic",
    "missed_fork",
    "missed_pin_or_skewer",
    "missed_mate",
    "ignored_threat",
    "failed_recapture",
    "bad_trade",
    "king_safety",
    "premature_attack",
    "development_neglect",
    "pawn_structure",
    "endgame_technique",
    "opening_theory",
    "time_management",
    "calculation_error",
]


def _score_to_str(score: chess.engine.PovScore) -> str:
    white = score.white()
    if white.is_mate():
        return f"#{white.mate()}"
    return f"{white.score(mate_score=MATE_CP) / 100.0:+.2f}"


from app.notation import legal_san_mentions as _legal_san_mentions


def _valid_diagnosis_override(reason: str, recommended: str, chosen: str) -> bool:
    if not reason:
        return False
    text = reason.strip().casefold()
    if len(text) < 24:
        return False
    boilerplate = (
        "matches backend recommendation",
        "backend recommendation",
        "same as recommended",
        "override",
        "n/a",
    )
    if any(phrase in text for phrase in boilerplate):
        return False
    if recommended.replace("_", " ") in text and chosen.replace("_", " ") not in text:
        return False
    return True


class CoachTools:
    def __init__(self, state, timeline, engines, config, on_progress=None):
        self.state = state
        self.timeline = timeline
        self.engines = engines
        self.config = config
        self.on_progress = on_progress  # optional callable(done, total) for review progress
        self.state.lesson_metadata = load_lesson_metadata(getattr(config, "lesson_file", None))

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------

    def analyze_position(self, fen: str, moves_to_apply: list[str] = None, depth: int = 18, pov: str = "white") -> dict:
        """
        The PRIMARY analysis tool: returns a complete, backend-verified coaching
        packet for a position — engine lines, piece-safety (exact exchange
        evaluation), opponent threats, human-likely moves (Maia), tactical
        motifs, game phase, and structural features.

        Args:
            fen: The starting FEN string. Copy it EXACTLY from backend data; never construct one yourself.
            moves_to_apply: Optional SAN moves (e.g. ["Rd1", "Rxb2"]) to play from the FEN BEFORE analyzing.
                            Use this to verify any hypothetical line — NEVER calculate tactics mentally.
            depth: Engine search depth (default 18, clamped 12-24, always wall-clock capped).
            pov: "white" or "black" — orient the rendered board to the student's side.
        """
        try:
            safe_depth = max(12, min(int(depth), 24))
            board = chess.Board(fen)
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid FEN: {e}"}

        invert = self.config.dark_terminal
        token_before = render_board_token(self.state, board, pov, invert)
        token_after = None

        if moves_to_apply:
            for move_str in moves_to_apply[:10]:
                try:
                    board.push_san(move_str)
                except ValueError:
                    legal = ", ".join(board.san(m) for m in board.legal_moves)
                    return {"error": f"Move '{move_str}' is illegal at that point in the sequence.",
                            "position_reached": board.fen(),
                            "legal_moves_there": legal}
            token_after = render_board_token(self.state, board, pov, invert)

        try:
            infos = self.engines.top_lines(board, multipv=3, depth=safe_depth)
        except Exception as e:
            return {"error": f"Engine analysis failed: {e}"}

        top_lines = []
        best_pv_san = []
        for i, info in enumerate(infos):
            pv = info.get("pv") or []
            pv_san, temp = [], board.copy(stack=False)
            for move in pv[:8]:
                pv_san.append(temp.san(move))
                temp.push(move)
            if i == 0:
                best_pv_san = pv_san
            top_lines.append({"rank": i + 1, "evaluation": _score_to_str(info["score"]),
                              "expected_engine_line": " ".join(pv_san)})

        features = extract_board_features(board)
        threats = self.engines.threat_probe(board)
        motifs = detect_motifs_in_line(board, best_pv_san) if best_pv_san else []

        maia_summary = None
        maia = self.engines.maia_predict(board.fen())
        if maia:
            maia_summary = {"top_human_moves": maia["top_moves"],
                            "win_probability_for_side_to_move": maia["win_probability_for_side_to_move"]}

        return {
            "top_stockfish_lines": top_lines,
            "tactical_motifs_in_best_line": motifs,
            "threats_if_side_to_move_passes": threats,
            "human_tendency_at_student_elo": maia_summary,
            "board_token_before_moves": token_before,
            "board_token_after_moves": token_after,
            "mathematically_legal_moves": ", ".join(board.san(m) for m in board.legal_moves),
            "positional_and_safety_features": features,
        }

    def evaluate_student_answer(self, fen: str, student_move_san: str) -> dict:
        """
        Grade the student's proposed move. You MUST call this whenever the student
        answers a "find the move / what's the threat" question with a concrete move.
        The backend counts attempts: the correct answer (and engine line) is only
        included once `reveal_allowed` is true — when the student got it right,
        or after 2 failed attempts.

        Args:
            fen: The FEN of the position the question was asked about (copy it exactly).
            student_move_san: The move the student proposed, in SAN (e.g. "Nf3").
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}

        attempts = self.state.socratic_attempts
        mentioned = _legal_san_mentions(board, student_move_san)
        if len(mentioned) > 1:
            return {
                "verdict": "needs_commitment",
                "candidate_moves": mentioned,
                "message": "The student listed several legal candidates. Ask which one they choose before grading.",
                "reveal_allowed": False,
            }
        try:
            move = board.parse_san(student_move_san)
        except ValueError:
            attempts[fen] = attempts.get(fen, 0) + 1
            n = attempts[fen]
            return {"verdict": "illegal",
                    "explanation": f"'{student_move_san}' is not a legal move here.",
                    "attempts_so_far": n, "reveal_allowed": n >= 2,
                    "legal_moves": ", ".join(board.san(m) for m in board.legal_moves)}

        try:
            infos = self.engines.top_lines(board, multipv=5, depth=16, time_cap=3.0)
        except Exception as e:
            return {"error": f"Engine analysis failed: {e}"}

        pov = board.turn
        ranked = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            ranked.append((pv[0], info["score"].pov(pov).score(mate_score=MATE_CP), pv))
        if not ranked:
            return {"error": "Engine returned no lines for this position."}

        best_move, best_cp, best_pv = ranked[0]
        student_cp = None
        student_rank = None
        for idx, (mv, cp, _) in enumerate(ranked):
            if mv == move:
                student_cp, student_rank = cp, idx + 1
                break
        if student_cp is None:
            after = board.copy(stack=False)
            after.push(move)
            try:
                score, _ = self.engines.quick_eval(after.fen(), 1.0)
                student_cp = score.pov(pov).score(mate_score=MATE_CP)
            except Exception as e:
                return {"error": f"Engine analysis failed: {e}"}

        drop = max(0.0, cp_to_win_pct(best_cp) - cp_to_win_pct(student_cp))
        if move == best_move or drop <= 0.5:
            verdict = "best"
        elif drop <= 2:
            verdict = "excellent_alternative"
        elif drop <= 5:
            verdict = "reasonable_but_imprecise"
        elif drop <= 12:
            verdict = "inaccurate"
        else:
            verdict = "wrong"

        # In tablebase range, category truth beats centipawns: a "slow" move
        # that keeps the proven result is correct; one that downgrades it is wrong.
        tablebase_verdict = None
        if len(board.piece_map()) <= 7 and getattr(self.config, "use_online_databases", True):
            from app import external
            tb = external.tablebase_probe(fen)
            if tb:
                pos_cat = external.normalize_category(tb.get("category"))
                entry = next((m for m in (tb.get("moves") or [])
                              if m.get("uci") == move.uci() or m.get("san") == board.san(move)), None)
                if pos_cat and entry:
                    move_cat = external.invert_category(external.normalize_category(entry.get("category")))
                    if move_cat == pos_cat:
                        verdict = "best"
                        tablebase_verdict = f"tablebase-verified: this move KEEPS the proven {pos_cat}"
                    else:
                        verdict = "wrong"
                        tablebase_verdict = (f"tablebase-verified: this move turns a proven {pos_cat} "
                                             f"into a proven {move_cat}")

        correct = verdict in ("best", "excellent_alternative")
        attempts[fen] = attempts.get(fen, 0) + 1
        n = attempts[fen]
        reveal = correct or n >= 2

        result = {
            "verdict": verdict,
            "student_move": board.san(move),
            "student_move_evaluation": f"{(student_cp if pov == chess.WHITE else -student_cp) / 100.0:+.2f}",
            "win_pct_lost_vs_best": round(drop, 1),
            "rank_among_engine_top5": student_rank,
            "attempts_so_far": n,
            "reveal_allowed": reveal,
        }
        if tablebase_verdict:
            result["tablebase_verdict"] = tablebase_verdict
        maia = self.engines.maia_predict(fen)
        if maia:
            prob = maia["raw_probs"].get(move.uci())
            if prob is not None and prob >= 0.15:
                result["human_popularity"] = f"~{prob:.0%} of similar-rated players choose this move"

        if reveal:
            pv_san, temp = [], board.copy(stack=False)
            for m in best_pv[:8]:
                pv_san.append(temp.san(m))
                temp.push(m)
            result["best_move"] = board.san(best_move)
            result["best_line"] = " ".join(pv_san)
            record = self._agenda_record_for_fen(fen)
            if record and getattr(record, "explanation_primitives", None):
                result["explanation_primitives"] = record.explanation_primitives
                result["recommended_diagnosis"] = getattr(record, "recommended_diagnosis", None)
                result["lesson_objective"] = getattr(record, "lesson_objective", None)
                result["next_step"] = ("Ask the student to explain WHY the move works before logging diagnosis; "
                                       "then teach from these concrete primitives. Vivid phrasing is fine, "
                                       "but immediately unpack it with the move, square, target, checked king, "
                                       "captured piece, or attacked piece.")
        else:
            result["note"] = "Answer withheld by backend. Give ONE hint and ask the student to try once more."
            result["hint_facts"] = self._hint_facts(board, best_move)
        self._persist_lesson_progress(fen, "awaiting_diagnosis" if reveal else "awaiting_answer")
        return result

    def _agenda_record_for_fen(self, fen: str):
        if not self.state.review:
            return None
        return next((r for r in self.state.review.agenda if r.fen_before == fen), None)

    def _hint_facts(self, board: chess.Board, best_move: chess.Move) -> dict:
        """Verified facts about the withheld best move, so hints are never invented."""
        piece = board.piece_at(best_move.from_square)
        mover = board.turn
        after = board.copy(stack=False)
        after.push(best_move)
        attacked = []
        for sq in after.attacks(best_move.to_square):
            target = after.piece_at(sq)
            if target and target.color != mover and target.piece_type != chess.PAWN:
                attacked.append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(sq)}")
        return {
            "moving_piece": chess.piece_name(piece.piece_type) if piece else None,
            "is_a_capture": board.is_capture(best_move),
            "gives_check": board.gives_check(best_move),
            "enemy_pieces_it_attacks_after_moving": attacked[:4],
            "usage": "These are VERIFIED facts about the withheld move. Build your hint from "
                     "AT MOST ONE of them — never assert properties of a move that are not listed here.",
        }

    def evaluate_threat_answer(self, fen: str, threat_move_san: str) -> dict:
        """
        Grade the student's answer to "what is the opponent threatening?".
        Use this when the student names a THREAT (a move the OPPONENT would play
        next), as opposed to a move of their own (use evaluate_student_answer
        for those). Attempts are counted; the real threat is only included once
        `reveal_allowed` is true.

        Args:
            fen: The FEN of the position the question was asked about (copy it exactly).
            threat_move_san: The opponent move the student believes is the threat (e.g. "Qh4").
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        if board.is_check():
            return {"note": "The side to move is in check — the question here is how to MEET the check, "
                            "not what is threatened next."}

        probe = board.copy(stack=False)
        probe.push(chess.Move.null())
        attempts = self.state.socratic_attempts
        key = f"threat:{fen}"

        try:
            move = probe.parse_san(threat_move_san)
        except ValueError:
            attempts[key] = attempts.get(key, 0) + 1
            n = attempts[key]
            return {"verdict": "not_a_legal_opponent_move",
                    "explanation": f"'{threat_move_san}' is not a legal reply for the opponent here.",
                    "attempts_so_far": n, "reveal_allowed": n >= 2}

        student_threat = probe.san(move)
        try:
            threats = self.engines.threat_probe(board)
        except Exception:
            threats = []
        main_moves = [t["threat_line"].split()[0] for t in threats if t.get("threat_line")]

        if student_threat in main_moves:
            attempts[key] = attempts.get(key, 0) + 1
            matched = next(t for t in threats if t["threat_line"].split()[0] == student_threat)
            self._persist_lesson_progress(fen, "awaiting_diagnosis")
            return {"verdict": "correct_threat", "student_threat": student_threat,
                    "threat_severity": matched.get("threat"),
                    "threat_line": matched.get("threat_line"),
                    "attempts_so_far": attempts[key], "reveal_allowed": True}

        # Not the engine's main threat — is it a real threat at all?
        is_real = False
        student_gain_cp = None
        try:
            current, _ = self.engines.quick_eval(fen)
            current_cp = current.pov(probe.turn).score(mate_score=MATE_CP)
            after = probe.copy(stack=False)
            after.push(move)
            score, _ = self.engines.quick_eval(after.fen())
            student_gain_cp = score.pov(probe.turn).score(mate_score=MATE_CP) - current_cp
            is_real = student_gain_cp >= 120
        except Exception:
            pass

        attempts[key] = attempts.get(key, 0) + 1
        n = attempts[key]
        result = {"student_threat": student_threat, "attempts_so_far": n,
                  "verdict": "real_but_secondary_threat" if is_real else "not_a_real_threat",
                  "reveal_allowed": n >= 2}
        # Verified material for explaining the verdict — the model was observed
        # inventing geometric reasons ("the e-pawn is blocked") when handed a
        # bare classification. Explanations must come from THIS data.
        result["comparison"] = {
            "your_named_threat_gain_for_opponent": (
                f"{student_gain_cp / 100.0:+.2f}" if student_gain_cp is not None else "unknown"),
            "engine_main_threat": threats[0].get("threat") if threats else "none found",
            "note": "Explain why the named move is secondary or not a threat ONLY from these "
                    "numbers and lines. NEVER invent a geometric reason (blocked pawns, pins, "
                    "defenders) that no tool stated — call analyze_position if you need geometry.",
        }
        if not threats:
            if not is_real:
                result["verdict"] = "no_significant_threat_exists"
            result["reveal_allowed"] = True
            result["note"] = "The engine finds no significant threat if the side to move passes."
        if result["reveal_allowed"] and threats:
            result["actual_threats"] = threats
        elif not result["reveal_allowed"]:
            result["note"] = "Withheld. Tell the student this is not the main threat and let them look again."
        self._persist_lesson_progress(fen, "awaiting_diagnosis" if result["reveal_allowed"] else "awaiting_answer")
        return result

    # NOTE: the old predict_human_move tool was retired: analyze_position already
    # includes Maia's tendencies automatically, the review consults Maia on every
    # move, and the digest pre-computes the anticipated wrong answer.

    # ------------------------------------------------------------------
    # SPARRING
    # ------------------------------------------------------------------

    def play_sparring_reply(self) -> dict:
        """
        Plays the coach's reply move during sparring. The BACKEND chooses a
        human-realistic move (Maia at the student's level) — you do not pick the
        move yourself. Call this whenever it is your turn in a sparring game.
        """
        if not self.state.sparring_mode:
            return {"error": "Sparring mode is DISABLED. Tell the user to type /spar to enable it."}
        board = self.state.board
        if board.is_game_over():
            return {"game_over": True, "result": board.result(),
                    "message": "The game is over — no reply to play."}

        san = None
        maia = self.engines.maia_predict(board.fen())
        if maia and maia["top_moves"]:
            moves = maia["top_moves"]
            weights = [m["probability"] for m in moves]
            san = random.choices([m["move"] for m in moves], weights=weights, k=1)[0]
        else:
            try:
                infos = self.engines.top_lines(board, multipv=3, depth=8, time_cap=1.0)
                candidates = [i["pv"][0] for i in infos if i.get("pv")]
                if candidates:
                    san = board.san(random.choice(candidates))
            except Exception:
                pass
        if san is None:
            return {"error": "No engine available to select a sparring move."}

        played = self.timeline.play_san(san)
        result = {"coach_played": played, "new_fen": self.state.board.fen(),
                  "selected_by": "Maia (human-realistic at student Elo)" if maia else "Stockfish (shallow)"}
        try:
            score, _ = self.engines.quick_eval(self.state.board.fen())
            result["evaluation_after"] = _score_to_str(score)
        except Exception:
            pass
        if self.state.board.is_game_over():
            result["game_over"] = True
            result["result"] = self.state.board.result()
        return result

    def execute_coach_move(self, move_san: str) -> dict:
        """
        Plays ONE specific move on the global board during sparring. Use ONLY when
        the user explicitly dictates a move for you to play. For your own sparring
        replies, call `play_sparring_reply` instead (the backend picks the move).

        Args:
            move_san: The move in Standard Algebraic Notation (e.g., "Nf3", "O-O").
        """
        if not self.state.sparring_mode:
            return {"error": "Sparring mode is DISABLED. Tell the user to type /spar to enable it."}

        try:
            self.timeline.play_san(move_san)
            return {"success": True, "new_fen": self.state.board.fen(), "message": f"Successfully played {move_san}."}
        except ValueError as e:
            legal_moves_str = ", ".join(self.state.board.san(m) for m in self.state.board.legal_moves)
            return {
                "error": f"Logic Tutor Intervention: Move '{move_san}' is ILLEGAL in this position.",
                "causal_trace": str(e),
                "pedagogical_hint": "The piece is either pinned, blocked, or your King is in check.",
                "required_action": f"To repair your plan, you MUST choose from these mathematically verified valid moves: [{legal_moves_str}]"
            }

    # ------------------------------------------------------------------
    # DIAGNOSIS & MEMORY
    # ------------------------------------------------------------------

    def submit_diagnosis(self, fen: str, error_category: str, error_explanation: str,
                         preventative_habit: str, confidence: float = 1.0,
                         override_reason: str = None) -> dict:
        """
        Logs a diagnosed mistake to the student's long-term dossier. Call this ONCE
        per distinct mistake, after the student has worked through it.

        Pick the category that names the DECISION ERROR (why the wrong move was
        chosen), not the topic you happened to discuss. If the review digest
        suggests a category for this moment, use it unless the conversation
        revealed a different root cause.

        Args:
            fen: The exact FEN of the position BEFORE the mistaken move.
            error_category: One of:
                hanging_piece (left material to a simple capture),
                missed_tactic (an available tactic went unplayed),
                missed_fork / missed_pin_or_skewer / missed_mate (specific missed tactic),
                ignored_threat (didn't ask what the opponent's move threatened),
                failed_recapture (left a capture sequence unfinished),
                bad_trade (won material or traded at a strategic cost, e.g. gave up an active piece),
                king_safety (exposed own king / ignored an attack on it),
                premature_attack (attacked before development/safety was ready),
                development_neglect (left pieces undeveloped),
                pawn_structure (created a lasting structural weakness),
                endgame_technique (known endgame method misplayed),
                opening_theory (left known theory for a worse plan),
                time_management (the clock, not the board, caused the error),
                calculation_error (right idea, miscalculated sequence).
            error_explanation: One sentence explaining what went wrong in THIS position.
            preventative_habit: One actionable habit to avoid this mistake in the future.
            confidence: Your confidence in the diagnosis, 0.0-1.0.
            override_reason: ONLY when you choose a category DIFFERENT from the
                backend's recommended diagnosis: state the conversational evidence
                for the different root cause. OMIT this argument entirely when
                your category matches the recommendation.
        """
        category = (error_category or "").strip().lower()
        if category not in ERROR_TAXONOMY:
            return {"error": f"'{error_category}' is not in the taxonomy.",
                    "allowed_categories": ERROR_TAXONOMY}

        if any(d.get("fen") == fen for d in self.state.diagnostics):
            return {"success": True, "message": "Diagnosis already logged for this position. Do not call this tool again."}

        record = self._agenda_record_for_fen(fen)
        recommended = getattr(record, "recommended_diagnosis", None) if record else None
        mismatch_warning = None
        if recommended and category != recommended:
            mismatch_warning = (f"Backend recommended '{recommended}' for this moment; "
                                f"you logged '{category}'.")
            valid_override = _valid_diagnosis_override(override_reason, recommended, category)
            if not valid_override:
                return {"error": mismatch_warning + " Provide override_reason if the conversation revealed a different root cause.",
                        "recommended_diagnosis": recommended,
                        "chosen_category": category,
                        "lesson_objective": getattr(record, "lesson_objective", None)}

        diagnosis = {
            "fen": fen,
            "error_category": category,
            "error_explanation": (error_explanation or "")[:200],
            "preventative_habit": (preventative_habit or "")[:200],
            "confidence": float(confidence),
        }
        if recommended:
            diagnosis["recommended_diagnosis"] = recommended
        if override_reason:
            diagnosis["override_reason"] = override_reason[:200]
        self.state.diagnostics.append(diagnosis)
        self._persist_lesson_progress(fen, "diagnosis_logged")

        console.print()
        console.print(Panel(
            f"[dim white]Category:[/dim white] [dim red]{diagnosis['error_category']}[/dim red]\n"
            f"[dim white]What happened:[/dim white] [dim yellow]{diagnosis['error_explanation']}[/dim yellow]\n"
            f"[dim white]Habit:[/dim white] [dim green]{diagnosis['preventative_habit']}[/dim green]",
            title="[dim italic]Diagnosis logged[/dim italic]",
            border_style="dim blue", expand=False
        ))
        result = {"success": True}
        if mismatch_warning:
            result["warning"] = mismatch_warning
        return result

    def _persist_lesson_progress(self, fen: str = None, status: str = "awaiting_answer") -> None:
        if not getattr(self.config, "lesson_file", None):
            return
        # Only agenda moments drive lesson state: dossier drills, sparring, and
        # free-question grading must never overwrite a saved lesson position.
        if not (self.state.review and fen):
            return
        agenda = self.state.review.agenda
        idx = next((i for i, record in enumerate(agenda) if record.fen_before == fen), None)
        if idx is None:
            return
        meta = dict(self.state.lesson_metadata or {})
        if self.state.current_game_id:
            meta["last_game_id"] = self.state.current_game_id
        if status == "diagnosis_logged":
            if idx + 1 >= len(agenda):
                status = "lesson_complete"   # finished the final moment
                meta["last_agenda_index"] = idx
            else:
                meta["last_agenda_index"] = idx + 1
        else:
            meta["last_agenda_index"] = idx
        meta["lesson_status"] = status
        self.state.lesson_metadata = meta
        save_lesson_metadata(self.config.lesson_file, meta)

    def get_game_review_summary(self) -> dict:
        """
        Re-fetches the backend's review digest for the currently loaded game
        (move-quality annotations and the critical-moment agenda). Use this if
        you no longer have the digest in context.
        """
        if not self.state.review:
            return {"error": "No game review available. Ask the user to load a game first (/load_pgn or paste a PGN)."}
        return {"review_digest": self.state.review.digest()}

    def explain_pawn_structure_evolution(self, color: str = None) -> dict:
        """
        Returns a backend-verified pawn-structure story for the loaded game.
        Call this whenever the student asks how the pawn structure evolved.

        The result includes square-level events, file-status labels, king-shield
        facts, and language guardrails. Do not use structural terms like "open
        file", "collapsed", "isolated", "passed", or "pinned" unless this tool
        returned them.

        Args:
            color: Optional "white" or "black". Defaults to the student's side.
        """
        if not self.state.review:
            return {"error": "No game review available. Load a game first."}
        side = (color or self.state.review.student_color or "white").strip().lower()
        if side not in ("white", "black"):
            return {"error": "color must be 'white' or 'black'"}
        return analyze_pawn_structure_evolution(self.state.review, side)

    # ------------------------------------------------------------------
    # EXTERNAL DATA
    # ------------------------------------------------------------------

    def get_opening_repertoire(self, fen: str) -> dict:
        """
        Queries the Lichess Opening Databases for this position.

        Returns the opening name/ECO, what MASTERS play, and what amateurs at the
        student's Elo actually play. Use it during the opening phase to ground
        opening advice in real data instead of generic principles.

        Args:
            fen: The current FEN string.
        """
        try:
            chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        from app import external
        try:
            master_res = external.masters_moves(fen)
            if master_res is None:
                return {"error": "Opening explorer unavailable (network failure)."}

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
            valid_brackets = [str(b) for b in brackets if abs(b - self.config.user_elo) <= 250]
            ratings_param = ",".join(valid_brackets) if valid_brackets else "1000,1200"

            amateur_res = external.lichess_moves(fen, ratings_param)
            if amateur_res is None:
                return {"error": "Opening explorer (amateur DB) unavailable (network failure)."}

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

    def query_endgame_tablebase(self, fen: str) -> dict:
        """
        Queries the Syzygy 7-piece Endgame Tablebase for provably perfect play.
        Only valid with 7 or fewer total pieces on the board (the backend enforces this).
        Returns the proven result, the best move, AND the margin for error
        (how many of the legal moves preserve the result).

        Args:
            fen: The current FEN string.
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        piece_count = len(board.piece_map())
        if piece_count > 7:
            return {"error": f"Tablebase requires <= 7 pieces; this position has {piece_count}. "
                             "Use analyze_position instead."}
        from app import external
        res = external.tablebase_probe(fen)
        if res is None:
            return {"error": "Tablebase unavailable (network failure). Fall back to analyze_position."}
        moves = res.get("moves") or []
        pos_cat = external.normalize_category(res.get("category"))
        outcomes = {"winning": 0, "drawing": 0, "losing": 0}
        for m in moves:
            mc = external.invert_category(external.normalize_category(m.get("category")))
            if mc == "win":
                outcomes["winning"] += 1
            elif mc == "draw":
                outcomes["drawing"] += 1
            elif mc == "loss":
                outcomes["losing"] += 1
        preserving = outcomes.get({"win": "winning", "draw": "drawing", "loss": "losing"}.get(pos_cat, ""), 0)
        return {
            "status": res.get("category"),
            "dtz": res.get("dtz"),
            "dtm": res.get("dtm"),
            "best_move": moves[0].get("san", "None") if moves else "None",
            "move_outcomes_for_side_to_move": outcomes,
            "margin_for_error": (f"{preserving} of {len(moves)} legal moves preserve the proven "
                                 f"{pos_cat}" if pos_cat and moves else "unknown"),
            "note": "status is from the side to move's perspective ('win' = side to move wins with perfect play)."
        }

    def fetch_lichess_game_by_id(self, game_id: str, mode: str = "auto") -> dict:
        """
        Downloads a game from Lichess.org, runs the full backend review
        (move classification + critical-moment agenda), loads it into the
        timeline, and returns the review digest.

        If mode is "auto" and this game matches persisted lesson metadata,
        this tool does NOT load the game yet. It returns `resume_choice_required`
        so you can ask the student whether to continue from the saved moment or
        restart from move 1. Call again with mode="continue" or mode="restart".

        Args:
            game_id: The 8-character Lichess game ID. Do NOT pass the full URL.
            mode: "auto" (default), "continue", or "restart".
        """
        game_id = (game_id or "").strip()[:8]
        if not game_id.isalnum():
            return {"error": "Invalid Lichess game ID (must be 8 alphanumeric characters)."}
        mode = (mode or "auto").strip().lower()
        if mode not in ("auto", "continue", "restart"):
            return {"error": "mode must be one of: auto, continue, restart"}
        if mode == "auto" and self._has_resume_metadata(game_id):
            meta = self.state.lesson_metadata
            return {
                "resume_choice_required": True,
                "game_id": game_id,
                "saved_lesson": {
                    "last_agenda_index": meta.get("last_agenda_index", 0),
                    "lesson_status": meta.get("lesson_status", "unknown"),
                },
                "message": ("I recognize this game from a previous session. Ask the student: "
                            "continue from the saved lesson moment, or restart from move 1? "
                            "If they choose continue, call fetch_lichess_game_by_id with mode='continue'. "
                            "If they choose restart, call it with mode='restart'."),
            }
        try:
            url = f"https://lichess.org/game/export/{game_id}"
            params = {"evals": "false", "clocks": "true", "opening": "true", "tags": "true"}
            r = requests.get(url, params=params, headers={"Accept": "application/x-chess-pgn"}, timeout=15)
            if r.status_code == 404:
                return {"error": f"Game with ID '{game_id}' not found."}
            if r.status_code != 200:
                return {"error": f"Lichess API returned status {r.status_code}."}
            return self._review_and_load(r.text, source_game_id=game_id, resume=(mode == "continue"))
        except Exception as e:
            return {"error": str(e)}

    def fetch_chessgames_com_game(self, game_id: str) -> dict:
        """
        Downloads a Master-level game from chessgames.com, runs the full backend
        review, loads it into the timeline at move 0, and returns the review digest.

        Args:
            game_id: The numeric game ID from chessgames.com. Do NOT pass the full URL.
        """
        game_id = (game_id or "").strip()
        if not game_id.isdigit():
            return {"error": "Invalid chessgames.com game ID (must be numeric)."}
        try:
            url = f"https://www.chessgames.com/njs/api/game/viewPGN/{game_id}"
            headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/plain, application/x-chess-pgn, */*"}
            r = requests.get(url, params={"raw": "true"}, headers=headers, timeout=15)
            if r.status_code == 404:
                return {"error": f"Game with ID '{game_id}' not found."}
            if r.status_code != 200:
                return {"error": f"Chessgames API returned status {r.status_code}."}
            pgn_text = r.text.strip()
            if not pgn_text or not pgn_text.startswith("[Event"):
                return {"error": "Received malformed data."}
            return self._review_and_load(pgn_text)
        except Exception as e:
            return {"error": str(e)}

    def _has_resume_metadata(self, game_id: str) -> bool:
        meta = self.state.lesson_metadata or {}
        # A completed (or never-started) lesson has nothing to resume into.
        return (meta.get("last_game_id") == game_id
                and meta.get("lesson_status") not in (None, "", "unavailable",
                                                      "lesson_complete", "fresh_review"))

    def _review_and_load(self, pgn_text: str, source_game_id: str = None, resume: bool = False) -> dict:
        console.print("[dim yellow]↳ Running full engine review on the downloaded game (may take a minute)...[/dim yellow]")
        review = review_game(pgn_text, self.engines, self.config, on_progress=self.on_progress)
        if review is None:
            return {"error": "Could not parse a game with moves from the downloaded PGN."}
        self.timeline.load_game(review.game)
        self.state.review = review
        self.state.current_game_id = source_game_id
        self.state.socratic_attempts.clear()
        attach_board_tokens(review, self.state, invert=getattr(self.config, "dark_terminal", False))
        meta = dict(self.state.lesson_metadata or {})
        if source_game_id:
            meta["last_game_id"] = source_game_id
        if resume:
            agenda_index = int(meta.get("last_agenda_index") or 0)
            agenda_index = max(0, min(agenda_index, max(len(review.agenda_plies) - 1, 0)))
            if review.agenda_plies:
                self.timeline.go_to_ply(review.agenda_plies[agenda_index])
            meta["last_agenda_index"] = agenda_index
            meta["lesson_status"] = "awaiting_answer"
            self.state.lesson_metadata = meta
            if getattr(self.config, "lesson_file", None):
                save_lesson_metadata(self.config.lesson_file, meta)
            record = review.agenda[agenda_index] if review.agenda else None
            return {
                "review_digest": review.digest(),
                "resume": {
                    "agenda_index": agenda_index,
                    "moment_number": agenda_index + 1,
                    "move_label": record.move_label if record else None,
                    "fen_before": record.fen_before if record else None,
                    "board_token": record.board_token if record else None,
                    "lesson_status": meta.get("lesson_status"),
                },
                "board_state": "Loaded at the saved agenda position. Continue that moment unless the student redirects.",
            }
        meta.setdefault("last_agenda_index", 0)
        meta["lesson_status"] = "fresh_review"
        self.state.lesson_metadata = meta
        if source_game_id and getattr(self.config, "lesson_file", None):
            save_lesson_metadata(self.config.lesson_file, meta)
        return {"review_digest": review.digest(),
                "board_state": "Loaded at the STARTING position. The user steps through with /next."}

    def get_tool_list(self):
        """Returns the bound methods to be passed to the Gemini API."""
        return [
            self.analyze_position, self.evaluate_student_answer, self.evaluate_threat_answer,
            self.play_sparring_reply, self.execute_coach_move, self.submit_diagnosis,
            self.get_game_review_summary, self.explain_pawn_structure_evolution,
            self.get_opening_repertoire,
            self.query_endgame_tablebase, self.fetch_lichess_game_by_id,
            self.fetch_chessgames_com_game,
        ]
