import io
import random
import re

import chess
import chess.engine
import chess.pgn
import requests
from rich.panel import Panel

from app import external
from app.concepts import extract_board_features, see_exchange_gain
from app.drills import record_drill_outcome
from app.memory import load_lesson_metadata, save_lesson_metadata
from app.motifs import detect_motifs_in_line
from app.pawn_story import analyze_pawn_structure_evolution
from app.planner import _normalize_text
from app.rating import current_rating_enrichment, rating_offer_note
from app.render import render_board_token
from app.review import (review_game, attach_board_tokens, cp_to_win_pct, MATE_CP, _wdl_risk_profile,
                        _draw_rule, _record_board, GameReview)
from app.self_assessment import start_self_assessment, self_assessment_open_note
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
    "allows_forcing_sequence",
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

# Move notation inside an explanation: piece moves, pawn captures, castling (case-sensitive,
# so prose like "b4" or "bishop" is untouched; bare squares are not moves here).
_MOVE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+\.+\s*)?(?:O-O(?:-O)?|0-0(?:-0)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?)[+#]?[!?]*"
)

_EXPLANATION_GIVE_UP_RE = re.compile(
    r"\b(?:i\s*(?:don'?t|do\s*not)\s*know|no\s+idea|i\s*give\s*up|idk|"
    r"just\s+tell\s+me|tell\s+me|show\s+me|no\s+se|ni\s+idea|dimelo|"
    r"solo\s+dime|dime(?:lo)?|paso)\b",
    re.IGNORECASE,
)


def _score_to_str(score: chess.engine.PovScore) -> str:
    white = score.white()
    if white.is_mate():
        return f"#{white.mate()}"
    return f"{white.score(mate_score=MATE_CP) / 100.0:+.2f}"


from app.notation import legal_san_mentions as _legal_san_mentions, san_mentions as _san_mentions


def _valid_diagnosis_override(reason: str, recommended: str, chosen: str) -> bool:
    """An override reason must contain actual evidence, not just restate the
    category names or meta-talk about overriding. Arguing AGAINST the
    recommended category ("this was not a bad trade because...") is the
    natural — and valid — way to justify an override."""
    if not reason:
        return False
    text = reason.strip().casefold()
    if len(text) < 24:
        return False
    residue = text
    for phrase in (recommended.replace("_", " "), chosen.replace("_", " "),
                   recommended.casefold(), chosen.casefold(),
                   "matches backend recommendation", "backend recommendation",
                   "recommended category", "same as recommended",
                   "overriding", "override", "n/a"):
        residue = residue.replace(phrase, " ")
    return len(residue.split()) >= 5


def _file_pawn_named(text: str, square: str) -> bool:
    if not square or not re.match(r"^[a-h][1-8]$", square):
        return False
    return bool(re.search(rf"\b{square[0]}\s*[- ]?\s*pawn\b", text or ""))


_BARE_SQUARE_RE = re.compile(r"[a-h][1-8]")

_GENERIC_REASON_KEYWORDS = {
    "move", "moves", "piece", "pieces", "pawn", "pawns", "king", "queen",
    "rook", "rooks", "bishop", "bishops", "knight", "knights", "file",
    "rank", "square", "squares",
}


def _ungrounded_diagnosis_moves(record, explanation: str) -> list:
    """SAN tokens in a diagnosis explanation that match no verified truth:
    not legal in the position before or after the mistake (or as an opponent
    threat from the before-position), and absent from the verified
    best/punishment lines. Bare squares like 'b4' are skipped — they usually
    name a square, not a move."""
    tokens = [t for t in _san_mentions(explanation)
              if not _BARE_SQUARE_RE.fullmatch(t.rstrip("+#"))]
    if not tokens:
        return []
    verified = {san.rstrip("+#") for san in
                ([record.san, record.best_move_san]
                 + list(record.best_line or []) + list(record.punishment_line or []))
                if san}
    boards = []
    for fen in (record.fen_before, record.fen_after):
        try:
            boards.append(chess.Board(fen))
        except ValueError:
            pass
    if boards:
        # The opponent's standing threat: the before-position with the turn
        # flipped ("you needed to stop Qg4" must not be flagged).
        threat_board = boards[0].copy(stack=False)
        threat_board.turn = not threat_board.turn
        threat_board.ep_square = None
        boards.append(threat_board)

    def parses(board, san):
        try:
            board.parse_san(san.rstrip("+#"))
            return True
        except (ValueError, AssertionError):
            return False

    return [t for t in tokens
            if t.rstrip("+#") not in verified and not any(parses(b, t) for b in boards)]


def _weak_diagnosis_text(text: str) -> bool:
    text = (text or "").strip()
    if len(text.split()) < 5:
        return True
    lowered = text.rstrip(". ").casefold()
    dangling = (
        "with", "starting with", "because", "after", "by", "to", "from",
        "against", "allowing black to launch a devastating attack starting with",
    )
    return any(lowered.endswith(word) for word in dangling)


def _contradictory_diagnosis_text(record, explanation: str) -> bool:
    if not record or getattr(record, "quality", None) not in ("inaccuracy", "mistake", "blunder"):
        return False
    text = _normalize_text(explanation or "")
    played = _normalize_text(getattr(record, "san", None) or "")
    label = _normalize_text(getattr(record, "move_label", None) or "")
    if not played:
        return False
    move_patterns = [re.escape(played)]
    if label:
        move_patterns.append(re.escape(label))
    move = r"(?:`?\b(?:" + "|".join(move_patterns) + r")\b`?)"
    praise = r"(?:was|is)\s+(?:needed|required|best|correct|the\s+engine[- ]verified\s+improvement)"
    if re.search(move + r"[^.?!;\n]{0,80}\b" + praise, text):
        return True
    if re.search(r"\b" + praise + r"[^.?!;\n]{0,80}" + move, text):
        return True
    return False


GRADING_TOOLS = ("evaluate_student_answer", "evaluate_threat_answer", "evaluate_explanation")
_MOVE_VERDICTS = {
    "best": "{m} is the best move",
    "excellent_alternative": "{m} is an excellent alternative, nearly as good as the best move",
    "reasonable_but_imprecise": "{m} is reasonable but not the most precise move",
    "inaccurate": "{m} is inaccurate",
    "wrong": "{m} is a mistake here",
}
_THREAT_VERDICTS = {
    "correct_threat": "{m} is the real threat",
    "real_but_secondary_threat": "{m} is a real idea but not the main threat",
    "not_a_real_threat": "{m} is not a real threat here",
    "not_a_legal_opponent_move": "{m} is not a legal move for the opponent here",
}


_DRAW_AVOIDANCE_RE = re.compile(r"\b(avoid|escape|stop|prevent)\w*\b[^.]*\b(perpetual|repetition|draw)")


def _draw_avoidance_correction(record, normalized_text: str) -> str | None:
    """The student treats a perpetual check or repetition as a danger while their side is
    objectively worse, where it is the drawing chance (run sce7o0q6 S18: "Kh2, because it avoids
    the perpetual" stood uncorrected)."""
    if not _DRAW_AVOIDANCE_RE.search(normalized_text or ""):
        return None
    deep = record.deep_eval_before_cp if record.deep_eval_before_cp is not None else record.eval_before_cp
    mover_cp = deep if record.color == "white" else -deep
    if mover_cp > -200:
        return None
    side = record.color.capitalize()
    return (f"for {side}, who is worse here ({mover_cp / 100:+.1f} with best play), a perpetual check or a "
            "repetition is a drawing chance to welcome, not something to avoid.")


def feedback_sentence(tool: str, result: dict) -> str | None:
    """The backend's own student-facing wording of one grading result (run kjofd3ug: the coach
    graded, logged the diagnosis and moved on without saying the verdict)."""
    if not isinstance(result, dict) or result.get("error") \
            or result.get("verdict") == "stale_or_fabricated_student_input":
        return None
    if tool == "evaluate_explanation":
        if result.get("key_reason_named") is True:
            return "Your reason is right."
        if result.get("credited"):
            return "What you said is true: " + "; ".join(result["credited"]) + "."
        return None
    verdict = result.get("verdict")
    if tool == "evaluate_threat_answer":
        text, subject = _THREAT_VERDICTS.get(verdict, ""), result.get("student_threat")
    elif verdict == "illegal":
        text, subject = "{m} is not legal here", result.get("student_move")
    else:
        text, subject = _MOVE_VERDICTS.get(verdict, ""), result.get("student_move")
    if not text:
        return None
    sentence = text.format(m=subject or "that move")
    # A sentence opening with a move keeps it as written: "e4 is the best move", never "E4".
    return (sentence if subject else sentence[0].upper() + sentence[1:]) + "."


def grading_next_step(tool: str, result: dict, can_grade_explanation: bool = True) -> str | None:
    """The verdict to say first and the one thing to do next, for one grading result.

    Live runs: the coach often skipped telling the student the verdict (baseline cgxiksj3:
    S4, S11, S16, S18), and once read a correct THREAT answer as permission to reveal the
    withheld move (e054prsf S2). Used for the coach's own tool calls and for router pre-grading;
    the router grades a reason in the same message itself, so it passes can_grade_explanation=False."""
    if not isinstance(result, dict) or result.get("error") \
            or result.get("verdict") == "stale_or_fabricated_student_input":
        return None
    verdict = result.get("verdict")
    if tool == "evaluate_explanation":
        guidance = result.get("guidance") or "respond to the explanation grade."
        if result.get("key_reason_named") is True:
            return "Say first: their reason is right. Then: " + guidance
        if result.get("key_reason_named") is False and not result.get("student_passed"):
            return ("Say first what is right in their reason and that one key point is still missing. "
                    "Then: " + guidance)
        return guidance
    if tool == "evaluate_threat_answer":
        said = _THREAT_VERDICTS.get(verdict, "").format(m=result.get("student_threat") or "that")
        if verdict == "correct_threat":
            step = "ask which move the student would play; the better move stays withheld."
        elif result.get("reveal_allowed"):
            step = "reveal the real threat from the result; the better move stays withheld."
        else:
            step = "give the next hint toward the threat only; do not reveal it."
        return (f"Say first: {said}. Then: " if said else "Then: ") + step
    if verdict == "needs_commitment":
        return "Ask which ONE of the candidate moves the student chooses; do not grade either."
    if verdict == "illegal":
        said = "that move is not legal here"
    else:
        said = _MOVE_VERDICTS.get(verdict, "").format(m=result.get("student_move") or "that move")
    if verdict in ("best", "excellent_alternative"):
        # 19enjp8m S4: the student gave the reason with the move and was asked for it again.
        step = ("if their message already says why, grade those words with evaluate_explanation "
                "instead of asking again; otherwise ask WHY it works." if can_grade_explanation
                else "ask WHY it works.")
    elif result.get("reveal_allowed"):
        step = "reveal the better move and its verified line, and ask why it works."
    else:
        step = "give the next hint only; do not reveal the better move."
    return (f"Say first: {said}. Then: " if said else "Then: ") + step


def _human_moves_in_words(top_moves: list, elo: int) -> str | None:
    """A sentence the coach can quote. Router run t0zauzig: Qh6 31.6%, Qa6 12.6%, Qg6 11.6%
    became "most humans at this level try Qh6 or Qg6" (no majority; the second move dropped).
    The rating Maia was asked about is named (19enjp8m mixed it up with the game-time rating)."""
    moves = [m for m in top_moves or [] if m.get("move") and m.get("probability") is not None][:3]
    if not moves:
        return None
    cohort = f"players rated about {elo}"
    if moves[0]["probability"] >= 0.5:
        return f"Most {cohort} choose {moves[0]['move']} ({moves[0]['probability']:.0%})."
    listed = ", ".join(f"{m['move']} {m['probability']:.0%}" for m in moves)
    return f"No single move is chosen by most {cohort}: {listed}."


def _backend_diagnosis_sentence(record) -> str:
    primitives = getattr(record, "explanation_primitives", None) or {}
    sentence = primitives.get("diagnosis_sentence")
    if sentence:
        return sentence
    if getattr(record, "punishment_line", None):
        return (f"The student played {record.move_label}; the verified punishment is "
                f"{' '.join(record.punishment_line[:4])}.")
    if getattr(record, "best_move_san", None):
        return f"The student played {record.move_label} instead of {record.best_move_label or record.best_move_san}."
    return f"The student played {record.move_label} and lost ground in the position."


def _move_label(board: chess.Board, san: str) -> str:
    dots = "." if board.turn == chess.WHITE else "..."
    return f"{board.fullmove_number}{dots} {san}"


class CoachTools:
    def __init__(self, state, timeline, engines, config, on_progress=None):
        self.state = state
        self.timeline = timeline
        self.engines = engines
        self.config = config
        self.on_progress = on_progress  # optional callable(done, total) for review progress
        self.state.lesson_metadata = load_lesson_metadata(getattr(config, "lesson_file", None))
        self._diagnosis_rejections = {}  # fen -> count of ungrounded-explanation rejections

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
            board = self._game_board(fen)
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid FEN: {e}"}
        invented = self._invented_game_fen(board.fen())
        if invented:
            return invented

        invert = self.config.dark_terminal
        token_before = render_board_token(self.state, board, pov, invert)
        token_after = None

        game_history = bool(board.move_stack)  # known only for positions of the reviewed game
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
                              "expected_engine_line": " ".join(pv_san), "depth_reached": info.get("depth")})
        reached = min((info.get("depth") or 0) for info in infos) if infos else 0
        # Run e_adq2d7 S27: a 4-second search stopped at depth 14 gave 31. Kh2 -4.59 and the coach quoted
        # it; at depth 30 it is -8.76. A cut-short search's numbers are not quotable as exact.
        search_note = (f"The search stopped at depth {reached} of {safe_depth} (time limit): its evaluations are "
                       "rough. Say only which side is better and roughly how much (e.g. 'clearly lost'); do not "
                       "quote these numbers." if infos and reached < safe_depth else None)

        forced_check = self._forced_piece_check(board, infos, min(safe_depth, 18))
        draw_rule = _draw_rule(board)
        features = extract_board_features(board)
        threats = self.engines.threat_probe(board)
        motifs = detect_motifs_in_line(board, best_pv_san) if best_pv_san else []

        maia_summary = None
        hist = [m.uci() for m in board.move_stack] if game_history else None
        maia = self.engines.maia_predict(board.fen(), history_uci=hist, tier="deep")
        if maia:
            # Run sce7o0q6 S2: a share for the position after the applied moves was told as if it
            # were for the position before them; the sentence names its position.
            in_words = _human_moves_in_words(maia["top_moves"], self.config.user_elo)
            if in_words and moves_to_apply:
                in_words = f"After {' '.join(moves_to_apply[:10])}: {in_words}"
            maia_summary = {"top_human_moves": maia["top_moves"], "in_words": in_words,
                            "win_probability_for_side_to_move": maia["win_probability_for_side_to_move"]}
            w, d, l = maia["win_probability_for_side_to_move"], maia.get("draw_probability"), maia.get("loss_probability")
            source = f"Maia at about {self.config.user_elo}"
            record = None if moves_to_apply else self._agenda_record_for_fen(fen)
            if record is not None and record.human_draw_prob is not None and record.human_win_prob is not None:
                # Run sce7o0q6 S27: the digest called the move-31 position must-hold (10/53/37 at the
                # game-time rating) and this call, at the coaching rating, said drawish (17/63/19).
                # An agenda position keeps the digest's numbers so the coach is never told both.
                w, d, l = record.human_win_prob, record.human_draw_prob, record.human_loss_prob or 0.0
                maia_summary["win_probability_for_side_to_move"] = w
                source = (f"Maia at about {record.maia_elo or self.config.user_elo} (the review digest's "
                          "figure for this position)")
            if d is not None:
                w = w or 0.0
                profile, guidance = _wdl_risk_profile(w, d, l or 0.0)
                maia_summary["win_draw_loss_for_side_to_move"] = {
                    "win": round(w, 3), "draw": round(d, 3), "loss": round(l or 0.0, 3)}
                maia_summary["risk_profile"] = profile
                maia_summary["wdl_source"] = source
                if guidance:
                    maia_summary["risk_guidance"] = guidance
            if draw_rule:  # run h2z7j1nd S27: a drawn final position was called "balanced, sharp"
                maia_summary["risk_profile"] = "drawn by rule"
                maia_summary["risk_guidance"] = ("The game is drawn here by rule. Maia's figures describe play "
                                                 "going on, which the side that wants the draw does not have to allow.")

        return {
            "top_stockfish_lines": top_lines,
            **({"search_note": search_note} if search_note else {}),
            "tactical_motifs_in_best_line": motifs,
            "threats_if_side_to_move_passes": threats,
            "human_tendency_at_student_elo": maia_summary,
            "board_token_before_moves": token_before,
            "board_token_after_moves": token_after,
            "mathematically_legal_moves": ", ".join(board.san(m) for m in board.legal_moves),
            "positional_and_safety_features": features,
            # Run kjofd3ug S6: "moving the queen is absolutely forced" with nothing behind it. Use
            # this sentence whenever the student asks whether a move is forced.
            **({"forced_check": forced_check} if forced_check else {}),
            # Stockfish never scores a repetition at the root of its own search: 31. Kh1 in 2dtDqQgR
            # repeated the position a third time, so the game was drawn whatever the lines say.
            **({"draw_rule": f"{draw_rule.replace('_', ' ')}: the position reached is a draw by rule ("
                             + ("claimable" if draw_rule in ("threefold_repetition", "fifty_move_rule")
                                else "automatic") + "), whatever the engine lines say if play went on"}
               if draw_rule else {}),
        }

    def _forced_piece_check(self, board: chess.Board, infos: list, depth: int) -> str | None:
        """When every top engine line moves the same piece, say whether moving it is forced: the
        best reply that does NOT move it, against the best overall (side to move's view)."""
        firsts = [info["pv"][0] for info in infos if info.get("pv")]
        if len(firsts) < 2 or len({m.from_square for m in firsts}) != 1:
            return None
        square = firsts[0].from_square
        piece = board.piece_at(square)
        if piece is None:
            return None
        name = f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
        others = [m for m in board.legal_moves if m.from_square != square]
        if not others:
            return f"Only the {name} can move, so moving it is forced."
        try:
            restricted = self.engines.best_restricted(board, others, depth=depth)
        except Exception:
            return None
        if restricted is None:
            return None
        score, move = restricted
        best_cp = infos[0]["score"].pov(board.turn).score(mate_score=100000)
        other_cp = score.pov(board.turn).score(mate_score=100000)
        gap, other, best = (best_cp - other_cp) / 100, board.san(move), board.san(firsts[0])
        # Both evaluations, White's view: "3.6 pawns worse" alone was read as an absolute eval (s7writoz).
        white = 1 if board.turn == chess.WHITE else -1
        evals = (f" than {best} ({other} {white * other_cp / 100:+.1f}, {best} {white * best_cp / 100:+.1f}, "
                 "White's view)")
        if gap >= 3:
            return (f"Moving the {name} is practically forced: the best reply that does not move it, {other}, "
                    f"is {gap:.1f} pawns worse{evals}. Which square it goes to is a choice ({best} is the "
                    "engine's best).")
        if gap < 1:
            return (f"Moving the {name} is the engine's preference, not forced: {other}, which does not move "
                    f"it, is only {gap:.1f} pawns worse{evals}.")
        return (f"Moving the {name} is clearly best but not forced: the best reply that does not move it, "
                f"{other}, is {gap:.1f} pawns worse{evals}.")

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
            board = self._game_board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        cached = self._pregraded_result("evaluate_student_answer", fen)
        if cached is not None:
            return cached
        provenance_error = self._verify_latest_student_input(
            "evaluate_student_answer", "student_move_san", student_move_san, [board])
        if provenance_error:
            return provenance_error

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
            threat_san = self._opponent_only_threat_answer(board, student_move_san, fen)
            if threat_san:
                result = self.evaluate_threat_answer(fen, threat_san)
                result["routed_from"] = "evaluate_student_answer"
                result["interpreted_as"] = "opponent_threat"
                result["routing_note"] = (
                    "The student's answer was illegal for the side to move but legal for the "
                    "opponent after a null move in a threat-aware agenda prompt, so it was "
                    "graded as an opponent threat instead of an illegal candidate move."
                )
                return result
            attempts[fen] = attempts.get(fen, 0) + 1
            n = attempts[fen]
            result = {"verdict": "illegal",
                      "explanation": f"'{student_move_san}' is not a legal move here.",
                      "attempts_so_far": n, "reveal_allowed": n >= 2,
                      "legal_moves": ", ".join(board.san(m) for m in board.legal_moves)}
            self._record_active_drill_outcome(fen, result)
            return result

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
        # On an agenda moment the reference is the moment's verified (deep, digest-taught) best
        # move, not a fresh search: in lost positions several moves tie in win% and the search
        # named the played 31. Kh1 "best" while grading 31. Kh2 best (runs cgxiksj3, t0zauzig).
        record = self._agenda_record_for_fen(fen)
        agenda_move = played_move = None
        if record is not None and getattr(record, "best_move_san", None) and getattr(record, "best_line", None):
            try:
                agenda_move = board.parse_san(record.best_move_san)
                played_move = board.parse_san(record.san) if record.san else None
            except ValueError:
                agenda_move = played_move = None
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
        if agenda_move is not None:
            if move == agenda_move:
                verdict = "best"
            elif move == played_move and record.quality in ("blunder", "mistake", "inaccuracy"):
                verdict = "inaccurate" if record.quality == "inaccuracy" else "wrong"

        # In tablebase range, category truth beats centipawns: a "slow" move
        # that keeps the proven result is correct; one that downgrades it is wrong.
        tablebase_verdict = None
        if len(board.piece_map()) <= 7 and getattr(self.config, "use_online_databases", True):
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

        student_san = board.san(move)
        result = {
            "verdict": verdict,
            "student_move": student_san,
            "student_move_label": _move_label(board, student_san),
            "student_move_evaluation": f"{(student_cp if pov == chess.WHITE else -student_cp) / 100.0:+.2f}",
            "win_pct_lost_vs_best": round(drop, 1),
            "rank_among_engine_top5": student_rank,
            "attempts_so_far": n,
            "reveal_allowed": reveal,
        }
        if tablebase_verdict:
            result["tablebase_verdict"] = tablebase_verdict
        maia = self.engines.maia_predict(fen, history_uci=[m.uci() for m in board.move_stack] or None,
                                         tier="deep")
        if maia:
            prob = maia["raw_probs"].get(move.uci())
            if prob is not None and prob >= 0.15:
                result["human_popularity"] = (f"~{prob:.0%} of players rated about {self.config.user_elo} "
                                              "choose this move")

        if reveal:
            if agenda_move is not None:
                best_san, best_line_san = board.san(agenda_move), list(record.best_line[:8])
            else:
                best_san, best_line_san, temp = board.san(best_move), [], board.copy(stack=False)
                for m in best_pv[:8]:
                    best_line_san.append(temp.san(m))
                    temp.push(m)
            result["best_move"] = best_san
            result["best_move_label"] = _move_label(board, best_san)
            result["best_line"] = " ".join(best_line_san)
            if record and getattr(record, "explanation_primitives", None):
                result["moment_move_label"] = record.move_label
                if record.best_move_label:
                    result["best_move_label"] = record.best_move_label
                result["explanation_primitives"] = record.explanation_primitives
                result["recommended_diagnosis"] = getattr(record, "recommended_diagnosis", None)
                result["lesson_objective"] = getattr(record, "lesson_objective", None)
                result["next_step"] = ("Ask the student to explain WHY the move works before logging diagnosis; "
                                       "then teach from these concrete primitives. Vivid phrasing is fine, "
                                       "but immediately unpack it with the move, square, target, checked king, "
                                       "captured piece, or attacked piece. Prefer the backend "
                                       "diagnosis_sentence when logging to the dossier.")
        else:
            result["note"] = "Answer withheld by backend. Give ONE hint and ask the student to try once more."
            result["hint_facts"] = self._hint_facts(board, agenda_move or best_move)
        self._record_active_drill_outcome(fen, result)
        self._persist_lesson_progress(fen, "awaiting_diagnosis" if reveal else "awaiting_answer")
        return result

    def _opponent_only_threat_answer(self, board: chess.Board, text: str, fen: str) -> str | None:
        """If a wrong-tool call submits an opponent-only SAN in a threat prompt,
        return the normalized opponent SAN so evaluate_student_answer can route
        to threat grading instead of falsely marking it as an illegal student
        move."""
        record = self._agenda_record_for_fen(fen)
        if not record or board.is_check():
            return None
        question_text = " ".join(
            [getattr(record, "initial_question", None) or ""]
            + [str(v) for v in (getattr(record, "question_ladder", None) or {}).values()]
        ).lower()
        if "threat" not in question_text:
            return None
        if self._frames_as_own_candidate(text):
            return None
        if _legal_san_mentions(board, text):
            return None
        mentions = _san_mentions(text)
        if len(mentions) != 1:
            return None
        probe = board.copy(stack=False)
        probe.push(chess.Move.null())
        opponent_mentions = _legal_san_mentions(probe, text)
        if len(opponent_mentions) != 1:
            return None
        return opponent_mentions[0]

    _OWN_CANDIDATE_RE = re.compile(
        r"\b(?:i|we|white|black)\s*(?:would|will|want|can|could|should|try|play|move|choose|go)\b|"
        r"\b(?:my|our|white'?s|black'?s)\s+move\b|"
        r"\b(?:candidate|answer)\s+is\b",
        re.IGNORECASE)

    def _frames_as_own_candidate(self, text: str) -> bool:
        return bool(self._OWN_CANDIDATE_RE.search(text or ""))

    @staticmethod
    def _compact_student_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", _normalize_text(text or ""))

    @staticmethod
    def _student_words(text: str) -> list:
        return re.findall(r"[a-z0-9]+", _normalize_text(text or ""))

    @staticmethod
    def _typo_variant(a: str, b: str) -> bool:
        """One substitution, insertion, deletion, or adjacent swap apart ("teh"/"the")."""
        if a == b:
            return True
        if min(len(a), len(b)) < 3 or abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            diffs = [i for i in range(len(a)) if a[i] != b[i]]
            return len(diffs) == 1 or (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                                       and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]])
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))

    def _near_verbatim(self, submitted: str, latest: str) -> bool:
        """The submitted explanation is the latest message with at most typo-level edits:
        at least 85% of its (3+) words appear in the latest message, allowing one-letter
        slips. A replay of an older turn ("not sure") does not qualify."""
        submitted_words = self._student_words(submitted)
        latest_words = self._student_words(latest)
        if len(submitted_words) < 3:
            return False
        matched = sum(1 for w in submitted_words if any(self._typo_variant(w, l) for l in latest_words))
        return matched / len(submitted_words) >= 0.85

    def _pregraded_result(self, tool: str, fen: str) -> dict | None:
        """The backend router already graded the student's message with this tool on this
        position this turn: return that result instead of spending another attempt."""
        entry = (getattr(self.state, "pregraded", None) or {}).get(f"{tool}:{fen}")
        if entry is None:
            return None
        return {**entry["result"], "already_graded_this_turn": True}

    def _verify_latest_student_input(self, tool_name: str, field_name: str, submitted: str,
                                     boards=()) -> dict | None:
        latest = getattr(self.state, "latest_user_text", None)
        if not latest or not submitted:
            return None
        submitted_compact = self._compact_student_text(submitted)
        latest_compact = self._compact_student_text(latest)
        if submitted_compact and submitted_compact in latest_compact:
            return None
        # A move the student described in words ("knight to e4", "castle kingside").
        for board in boards:
            try:
                san = board.san(board.parse_san(submitted.strip()))
            except ValueError:
                continue
            if san in _legal_san_mentions(board, latest):
                return None
        if field_name == "student_explanation":
            # The student's own reason may come from an earlier message in this same moment
            # (given with the move, before a detour question); verbatim or typo-level only.
            for text in [latest] + list(getattr(self.state, "moment_student_texts", None) or []):
                if (submitted_compact and submitted_compact in self._compact_student_text(text)) \
                        or self._near_verbatim(submitted, text):
                    return None
        event = {
            "type": "stale_or_fabricated_student_input",
            "tool": tool_name,
            "field": field_name,
            "submitted": (submitted or "")[:160],
            "latest_user_text": (latest or "")[:240],
            "game_id": getattr(self.state, "current_game_id", None),
        }
        if hasattr(self.state, "grading_integrity_events"):
            self.state.grading_integrity_events.append(event)
            self.state.grading_integrity_events[:] = self.state.grading_integrity_events[-50:]
        return {
            "verdict": "stale_or_fabricated_student_input",
            "error": ("The submitted grading text does not match the student's latest message. "
                      "Do not grade or spend attempts using stale, summarized, or invented student input."),
            "required_action": "answer_or_clarify_without_grading",
        }

    def _record_active_drill_outcome(self, fen: str, result: dict) -> None:
        drill = getattr(self.state, "active_dossier_drill", None) or {}
        if not drill or drill.get("fen") != fen or not result.get("reveal_allowed"):
            return
        entry = record_drill_outcome(getattr(self.config, "dossier_file", None), drill, result)
        if entry:
            result["drill_outcome_recorded"] = {
                "error_category": entry.get("error_category"),
                "success": entry.get("success"),
                "verdict": entry.get("verdict"),
            }
        self.state.active_dossier_drill.clear()

    def _agenda_record_for_fen(self, fen: str):
        if not self.state.review:
            return None
        return next((r for r in self.state.review.agenda if r.fen_before == fen), None)

    def _game_board(self, fen: str) -> chess.Board:
        """The board for `fen` with the game's moves on its stack when `fen` is a position of the reviewed
        game or the current timeline, so engine searches see repetitions; a bare FEN otherwise. FENs carry
        move numbers, so a position that occurred several times still maps to one ply."""
        board = chess.Board(fen)
        review = self.state.review
        if review is not None:
            for ply, record in enumerate(review.records):
                if record.fen_before == fen:
                    return review.board_before(ply) if isinstance(review, GameReview) else _record_board(record)
                if record.fen_after == fen:  # run h2z7j1nd S27: the final position (after 31. Kh1) had no history
                    after = (review.board_before(ply) if isinstance(review, GameReview) else _record_board(record))
                    after.push_uci(record.uci)
                    return after
        if self.state.board.move_stack and self.state.board.fen() == fen:
            return self.state.board.copy()
        return board

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
        cached = self._pregraded_result("evaluate_threat_answer", fen)
        if cached is not None:
            return cached
        provenance_error = self._verify_latest_student_input(
            "evaluate_threat_answer", "threat_move_san", threat_move_san, [probe])
        if provenance_error:
            return provenance_error

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
            "engine_main_threat_line": threats[0].get("threat_line") if threats else None,
            "student_named_move_is_legal": True,
            "secondary_threat_guidance": "If the named move is legal and thematically relevant, say so, "
                                         "but state that it is not the verified main threat.",
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

    def evaluate_explanation(self, fen: str, student_explanation: str) -> dict:
        """
        Grade the student's stated WHY against the verified reason-set for this
        moment. Call this AFTER the student has found the right move and explained
        why it works, BEFORE submit_diagnosis.

        It tells you whether the student named the position's LOAD-BEARING reason
        (the actual point — e.g. that the move stops mate on h2), as opposed to a
        correct-but-partial reason (e.g. "it gains tempo"). When the key reason is
        missing, do NOT say "exactly" and move on, and do NOT reveal it outright:
        PROBE it — a focused question or hint that points toward the idea — so the
        student finds it, then re-check by calling this again.

        Args:
            fen: The FEN the moment is about (copy it exactly from the digest).
            student_explanation: The student's own words for why the move works.
        """
        record = self._agenda_record_for_fen(fen)
        if record is None:
            return {"error": "No agenda moment matches that FEN. Use the exact fen_before from the digest."}
        cached = self._pregraded_result("evaluate_explanation", fen)
        if cached is not None:
            return cached
        provenance_error = self._verify_latest_student_input(
            "evaluate_explanation", "student_explanation", student_explanation)
        if provenance_error:
            return provenance_error
        reasons = (getattr(record, "explanation_primitives", None) or {}).get("reasons") or []
        key_reasons = [r for r in reasons if r.get("key")]
        # Squares inside move notation ("Rd4") name the move, not the reason: mask piece
        # moves, pawn captures and castling before matching (bare squares like "g4" stay).
        explanation = _MOVE_TOKEN_RE.sub(" ", student_explanation or "")
        text = _normalize_text(explanation)
        if not re.search(r"[a-z]", text):
            return {"gradable": False,
                    "guidance": ("The student only named a move, not a reason. Ask why it works "
                                 "without revealing the reason; this does not count as an attempt.")}
        squares_in = set(re.findall(r"[a-h][1-8]", text))
        words_in = set(re.findall(r"[a-z]+", text))

        def named(reason):
            reason_text = str(reason.get("text") or "").lower()
            keywords = {str(k).lower() for k in reason.get("keywords", [])}
            meaningful_keywords = keywords - _GENERIC_REASON_KEYWORDS
            reason_words = set(re.findall(r"[a-z]+", reason_text)) | keywords
            defense_reason = bool(reason_words & {"defend", "defends", "defended", "guard", "guards",
                                                  "protect", "protects", "cover", "covers", "control",
                                                  "controls"})
            if defense_reason and any(_file_pawn_named(text, sq) for sq in reason.get("squares", [])):
                return True
            return (bool(set(reason.get("squares", [])) & squares_in)
                    or any(k in words_in for k in meaningful_keywords))

        if not key_reasons or key_reasons[0].get("generic"):
            # No concrete load-bearing reason to probe: none pinned, or only "the engine prefers
            # this move" plus its line (#31; a probe would invent a target). What the student said
            # that is a verified board fact is credited by name, and nothing else is confirmed: run
            # s2hy697w S11 answered "Yes, exactly" to true but secondary reasons, and router run
            # 5kf7rqwj S18 affirmed "avoids the perpetual" under the old "if sound, affirm it".
            if key_reasons:
                self.state.socratic_attempts[f"explain-complete:{fen}"] = True
                self._persist_lesson_progress(fen, "awaiting_diagnosis")
            credited = [r["text"] for r in reasons if not r.get("key") and named(r)]
            correction = _draw_avoidance_correction(record, text)
            result = {"gradable": False, "credited": credited,
                      "guidance": ((f"First correct the student's reasoning: {correction} " if correction else "")
                                   + ("No concrete load-bearing reason is verified for this move beyond the "
                                      "engine line. " if key_reasons else
                                      "No single load-bearing reason is pinned for this moment. ")
                                   + (f"True in the student's words (verified): {'; '.join(credited)}. Credit "
                                      "exactly these. " if credited else
                                      "Nothing in the student's words is verified; do not confirm it. ")
                                   + "Do not say 'exactly' or 'that's it', and do not confirm any other claim "
                                     "in their words. Teach the verified line in plain words, then proceed to "
                                     "submit_diagnosis. Do not probe.")}
            if correction:
                result["correction"] = correction
            return result

        key = key_reasons[0]
        key_named = named(key)
        # Backend-enforced probe budget (not prompt-trusted): count probes per FEN so
        # the coach can't loop on the same missing-reason verdict forever — after two
        # tries it is told to teach the reason and move on (mirrors the move-grading
        # two-attempt reveal). Reset once the student gets it.
        attempts = self.state.socratic_attempts
        akey = f"explain:{fen}"
        result = {
            "gradable": True,
            "key_reason_named": key_named,
            "also_named": [r["text"] for r in reasons if not r.get("key") and named(r)],
        }
        complete_key = f"explain-complete:{fen}"
        if _EXPLANATION_GIVE_UP_RE.search(_normalize_text(student_explanation or "")):
            attempts[akey] = 2
            attempts[complete_key] = True
            result["probes_so_far"] = 2
            result["reveal_allowed"] = True
            result["key_reason"] = key["text"]
            result["missing_key_reason"] = key["text"]
            result["student_passed"] = True
            result["guidance"] = (
                "The student explicitly passed on explaining the reason. Stop probing: teach the "
                "load-bearing reason directly and kindly, then submit_diagnosis."
            )
            self._persist_lesson_progress(fen, "awaiting_diagnosis")
            return result

        if key_named:
            attempts.pop(akey, None)
            attempts[complete_key] = True
            result["probes_so_far"] = 0
            result["reveal_allowed"] = True
            result["key_reason"] = key["text"]
            result["guidance"] = ("The student named the load-bearing reason. Affirm briefly, then "
                                  "submit_diagnosis.")
            self._persist_lesson_progress(fen, "awaiting_diagnosis")
            return result

        attempts[akey] = attempts.get(akey, 0) + 1
        n = attempts[akey]
        anchor = ", ".join(key.get("squares", []))
        result["probes_so_far"] = n
        if n >= 2:
            attempts[complete_key] = True
            result["reveal_allowed"] = True
            result["key_reason"] = key["text"]
            result["missing_key_reason"] = key["text"]
            result["guidance"] = (
                "The student has now missed the load-bearing reason twice. Stop probing: teach it "
                "directly and kindly (the lesson matters more than the interrogation), then "
                "submit_diagnosis.")
            self._persist_lesson_progress(fen, "awaiting_diagnosis")
        else:
            result["reveal_allowed"] = False
            result.update(self._explanation_probe_packet(record, key))
            result["guidance"] = (
                "The student's explanation did NOT address the load-bearing reason. Do not "
                "rubber-stamp it and do not reveal the reason outright — PROBE it: ask a focused "
                "question or give a hint pointing toward "
                + (f"the {anchor} square / " if anchor else "")
                + "the key idea, so the student finds it. Re-check with evaluate_explanation after "
                "their reply.")
            result["guidance"] = (
                "The student's explanation did NOT address the load-bearing reason. Do not "
                "rubber-stamp it and do not reveal the reason outright. Ask probe_question "
                "verbatim or near-verbatim, using only probe_facts. Re-check with "
                "evaluate_explanation after their reply.")
            self._persist_lesson_progress(fen, "awaiting_explanation")
        return result

    def _explanation_probe_packet(self, record, key_reason: dict) -> dict:
        squares = list(key_reason.get("squares") or [])
        keywords = list(key_reason.get("keywords") or [])
        best = getattr(record, "best_move_label", None) or getattr(record, "best_move_san", None) or "the best move"
        square_text = squares[0] if squares else "the critical square"
        reason_text = " ".join([str(key_reason.get("text") or "").lower()] + [k.lower() for k in keywords])
        moved_piece = self._best_move_piece_on(record, square_text)
        if any(word in reason_text for word in ("defend", "guard", "protect", "cover", "control")):
            question = (f"After {best}, look at {square_text}. What piece covers it, "
                        "and along which line?")
            focus = "defense_relation"
        elif moved_piece:
            # The square is where the better move's own piece lands: "around e1" asks
            # about nothing in particular; ask what that piece does from there.
            question = f"After {best}, what does your {moved_piece} do from {square_text}?"
            focus = "moved_piece_role"
        elif any(word in reason_text for word in ("mate", "check", "forcing")):
            question = f"After {best}, what forcing threat on {square_text} is being stopped?"
            focus = "forcing_threat"
        elif any(word in reason_text for word in ("fork", "pin", "skewer", "tactic")):
            question = f"After {best}, what tactical idea around {square_text} changes?"
            focus = "tactical_relation"
        else:
            question = f"After {best}, what changes around {square_text}?"
            focus = "key_square"
        facts = {
            "best_move": best,
            "target_squares": squares,
            "keywords": keywords,
            "focus": focus,
            "usage": ("Use these facts only to ask the probe_question. Do not invent a "
                      "new missing detail or conceptual distinction."),
        }
        line_motifs = [m.get("detail") for m in getattr(record, "best_line_motifs", None) or []
                       if m.get("detail") and any(k in str(m.get("motif")) for k in keywords)]
        if line_motifs:
            facts["line_motifs"] = line_motifs
        return {"probe_question": question, "probe_facts": facts}

    @staticmethod
    def _best_move_piece_on(record, square: str) -> str | None:
        """Name of the better move's piece if it lands on `square`, else None."""
        try:
            board = chess.Board(record.fen_before)
            move = board.parse_san(record.best_move_san)
        except (ValueError, TypeError, AttributeError):
            return None
        if chess.square_name(move.to_square) != square:
            return None
        piece = board.piece_at(move.from_square)
        return chess.piece_name(piece.piece_type) if piece else None

    def explain_move(self, fen: str, move_san: str) -> dict:
        """
        Verified facts about a SPECIFIC move, so you explain WHY it is good or bad
        from backend truth ONLY — never invent geometry (a pin, a block, "the pawn
        is pinned", an attack) the data does not state. Call this before narrating a
        move's merits or flaws, and phrase your explanation using only what it returns.

        Args:
            fen: The position the move is made in (copy it exactly from backend data).
            move_san: The move in SAN (e.g. "Rfe1").
        """
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"error": f"Invalid FEN: {e}"}
        try:
            move = board.parse_san(move_san)
        except ValueError:
            return {"legal": False,
                    "error": f"'{move_san}' is not legal in this position — do not explain it as if it were."}

        mover = board.turn
        san = board.san(move)
        after = board.copy(stack=False)
        after.push(move)

        attacked = []
        for sq in after.attacks(move.to_square):
            target = after.piece_at(sq)
            if target and target.color != mover and target.piece_type != chess.PAWN:
                attacked.append(f"{chess.piece_name(target.piece_type)} on {chess.square_name(sq)}")

        moving = board.piece_at(move.from_square)
        facts = {
            "legal": True,
            "move": san,
            "moving_piece": chess.piece_name(moving.piece_type) if moving else None,
            "is_capture": board.is_capture(move),
            "gives_check": board.gives_check(move),
            "enemy_pieces_attacked_after": attacked[:4],
            # Does the move simply hang the piece it moved? (opponent profits capturing it)
            "moved_piece_hangs": see_exchange_gain(after, move.to_square) > 0,
        }
        if board.is_capture(move):
            facts["capture_see_centipawns"] = see_exchange_gain(board, move.to_square)
        motifs = detect_motifs_in_line(board, [san])
        if motifs:
            facts["motifs"] = [{"motif": m["motif"], "detail": m["detail"]} for m in motifs]
        try:
            before_score, _ = self.engines.quick_eval(fen)
            after_score, _ = self.engines.quick_eval(after.fen())
            facts["win_pct_before_for_mover"] = round(
                cp_to_win_pct(before_score.pov(mover).score(mate_score=MATE_CP)), 1)
            facts["win_pct_after_for_mover"] = round(
                cp_to_win_pct(after_score.pov(mover).score(mate_score=MATE_CP)), 1)
        except Exception:
            pass
        facts["usage"] = ("VERIFIED facts only. Explain the move's merit or flaw from THESE — never "
                          "assert a pin, block, fork, or attack that is not listed here. Call "
                          "analyze_position if you need more geometry.")
        return facts

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
        # Feed the live game's history so the sparring opponent plays in character
        # across the game rather than position-by-position.
        spar_hist = [m.uci() for m in board.move_stack] if board.move_stack else None
        maia = self.engines.maia_predict(board.fen(), history_uci=spar_hist, tier="deep")
        if maia and maia["top_moves"]:
            moves = maia["top_moves"]
            # spar_temperature shapes the sampling sharpness over Maia's top moves:
            # 1.0 = sample in proportion to human likelihood (most human-realistic);
            # <1 sharper (→0 = always the most likely move); >1 flatter / more varied.
            temp = float(getattr(self.config, "spar_temperature", 1.0))
            if temp <= 0:
                san = max(moves, key=lambda m: m["probability"])["move"]
            else:
                weights = [m["probability"] ** (1.0 / temp) for m in moves]
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
            result["post_mortem"] = "Offer /review_spar to run the standard review pipeline on this sparring game."
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
                allows_forcing_sequence (your move created a forcing reply; different from ignoring a threat that already existed),
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
        played = next((r for r in (self.state.review.records if self.state.review else [])
                       if r.fen_before == fen), None)
        if played is not None and played.draw_rule:
            # 2dtDqQgR: both decisive runs logged 31. Kh1 (a threefold repetition) as ignored_threat.
            return {"error": (f"{played.move_label} reached a draw by rule "
                              f"({played.draw_rule.replace('_', ' ')}); it is not a mistake and must not be "
                              "logged as one. Tell the student the game was drawn there."),
                    "draw_rule": played.draw_rule}

        if any(d.get("fen") == fen for d in self.state.diagnostics):
            return {"success": True, "message": "Diagnosis already logged for this position. Do not call this tool again."}

        record = self._agenda_record_for_fen(fen)
        recommended = getattr(record, "recommended_diagnosis", None) if record else None
        reasons = ((getattr(record, "explanation_primitives", None) or {}).get("reasons")
                   if record else [])
        if any(reason.get("key") for reason in (reasons or [])):
            complete_key = f"explain-complete:{fen}"
            if not self.state.socratic_attempts.get(complete_key):
                attempts_used = int(self.state.socratic_attempts.get(f"explain:{fen}", 0) or 0)
                remaining = max(1, 2 - attempts_used)
                plural = "" if remaining == 1 else "s"
                return {
                    "error": (f"Explanation grading is not complete for this position. "
                              f"{remaining} more explanation attempt{plural} required, or the "
                              "student may pass. Ask why the move works, then call "
                              "evaluate_explanation before submitting the diagnosis."),
                    "required_action": "evaluate_explanation",
                    "remaining_explanation_attempts": remaining,
                    "student_may_pass": True,
                }
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

        if record:
            if _contradictory_diagnosis_text(record, error_explanation):
                return {
                    "error": ("error_explanation contradicts the verified agenda: it describes the "
                              "played mistake as the needed/best move. Rebuild the explanation using "
                              "the backend diagnosis sentence below."),
                    "played_move": record.san,
                    "played_move_label": record.move_label,
                    "verified_quality": record.quality,
                    "best_move": record.best_move_san,
                    "backend_diagnosis_sentence": _backend_diagnosis_sentence(record),
                }
            ungrounded = _ungrounded_diagnosis_moves(record, error_explanation)
            if ungrounded:
                self._diagnosis_rejections[fen] = self._diagnosis_rejections.get(fen, 0) + 1
                if self._diagnosis_rejections[fen] >= 2:
                    # Don't let a stubborn fabrication stall the lesson: log a
                    # backend-built explanation instead of the invented one.
                    error_explanation = _backend_diagnosis_sentence(record)
                else:
                    return {"error": f"error_explanation names {', '.join(ungrounded)} — not a legal move "
                                     "in this position and not part of any verified line. NEVER invent "
                                     "tactical sequences: rebuild the explanation using ONLY the verified "
                                     "data below, then resubmit.",
                            "played_move": record.san,
                            "best_move": record.best_move_san,
                            "verified_punishment_line": record.punishment_line or None,
                            "verified_best_line": record.best_line or None}
            elif _weak_diagnosis_text(error_explanation):
                error_explanation = _backend_diagnosis_sentence(record)

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
        self._record_teaching_ledger_event(record, diagnosis)
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
        if not (self.state.review_context or {}).get("dossier_consent", True):
            result["dossier_logging"] = "disabled_for_borrowed_game_without_opt_in"
            result["message"] = (
                "Diagnosis logged for this session only. Do not claim it was saved to the "
                "student's personal dossier unless they opted into borrowed-game logging."
            )
        return result

    def _record_teaching_ledger_event(self, record, diagnosis: dict) -> None:
        if not record:
            return
        ledger = self.state.teaching_ledger
        moments = list(ledger.get("explained_moments") or [])
        if not any(m.get("fen") == record.fen_before for m in moments):
            theme_ids = []
            review = getattr(self.state, "review", None)
            for theme in getattr(review, "themes", []) or []:
                if record.ply in (getattr(theme, "plies", []) or []):
                    theme_ids.append(getattr(theme, "id", None))
            moments.append({
                "fen": record.fen_before,
                "ply": record.ply,
                "move_label": record.move_label,
                "diagnosis": diagnosis.get("error_category"),
                "recommended_diagnosis": getattr(record, "recommended_diagnosis", None),
                "lesson_objective": getattr(record, "lesson_objective", None),
                "quality": getattr(record, "quality", None),
                "theme_ids": [t for t in theme_ids if t],
            })
        ledger["explained_moments"] = moments[-12:]

        concepts = list(ledger.get("explained_concepts") or [])
        for concept in (
            diagnosis.get("error_category"),
            getattr(record, "recommended_diagnosis", None),
            getattr(record, "lesson_objective", None),
        ):
            if not concept:
                continue
            if not any(c.get("concept") == concept and c.get("fen") == record.fen_before for c in concepts):
                concepts.append({
                    "concept": concept,
                    "fen": record.fen_before,
                    "move_label": record.move_label,
                    "ply": record.ply,
                })
        ledger["explained_concepts"] = concepts[-24:]

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
        if status == "diagnosis_logged":
            if idx + 1 >= len(agenda):
                status = "lesson_complete"   # finished the final moment
                next_idx = idx
            else:
                next_idx = idx + 1
        else:
            next_idx = idx
        # Batch lessons use their own slot: running /batch must never clobber
        # an unfinished single-game lesson, and vice versa.
        batch_id = getattr(self.state.review, "batch_id", None)
        if batch_id:
            meta["batch_id"] = batch_id
            meta["batch_agenda_index"] = next_idx
            meta["batch_lesson_status"] = status
        else:
            if self.state.current_game_id:
                meta["last_game_id"] = self.state.current_game_id
            meta["last_agenda_index"] = next_idx
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

    def get_game_position(self, move_number: int, side: str = "white") -> dict:
        """
        The loaded game's real position before a given move, the move played there, and the position
        after it. Use this for any game position that is not an agenda moment (for example the final
        move) instead of constructing a FEN yourself; pass the returned FEN to analyze_position.

        Args:
            move_number: The game's move number (e.g. 31).
            side: "white" or "black": whose move it was.
        """
        review = self.state.review
        if not review or not review.records:
            return {"error": "No game is loaded."}
        side = (side or "white").strip().lower()
        record = next((r for r in review.records if r.number == int(move_number) and r.color == side), None)
        if record is None:
            return {"error": f"The game has no {side} move {move_number}.",
                    "game_moves": f"{review.records[0].move_label} to {review.records[-1].move_label}"}
        result = {"move_played": record.move_label, "fen_before": record.fen_before,
                  "fen_after": record.fen_after, "verified_quality": record.quality}
        if record.draw_rule:
            result["draw_rule"] = (f"{record.move_label} reached a draw by rule "
                                   f"({record.draw_rule.replace('_', ' ')}): it is not a mistake.")
            played_on = record.eval_without_draw_rule_cp
            mover_cp = (played_on if record.color == "white" else -played_on) if played_on is not None else None
            if mover_cp is not None and mover_cp <= -200:
                # Run 61tm0k1p S18: "to avoid the perpetual" went uncorrected (the explanation grader that
                # corrects it only runs on agenda moments).
                result["correction"] = (f"For {record.color.capitalize()}, who would have been worse if play had "
                                        "gone on, a perpetual check or repetition is the draw to welcome, not "
                                        "something to avoid. Say so if the student wanted to avoid it.")
        if record is review.records[-1]:
            result["game_ending"] = review.ending()
        return result

    def _invented_game_fen(self, fen: str) -> dict | None:
        """An error when `fen` claims a game move's number and side to move but is not that game position
        (run pr40jw2f analysed a made-up 'move 31' with two black queens and reported its loss odds)."""
        review = self.state.review
        if not review or not review.records:
            return None
        fields = fen.split()
        known = {self.state.board.fen(), *getattr(self.state, "claim_gate_extra_fens", [])}
        if len(fields) < 6 or fen in known:
            return None
        for game_fen in [r.fen_before for r in review.records] + [review.records[-1].fen_after]:
            game = game_fen.split()
            if (game[1], game[5]) == (fields[1], fields[5]) and game[:4] != fields[:4]:
                return {"error": (f"This FEN is not the game's position with {'White' if game[1] == 'w' else 'Black'} "
                                  f"to move at move {game[5]}. Never construct a FEN: use get_game_position for a "
                                  "game position, or moves_to_apply from a real FEN for a hypothetical line."),
                        "game_fen": game_fen}
        return None

    def explain_pawn_structure_evolution(self, color: str = None) -> dict:
        """
        Returns a backend-verified pawn-structure story for the loaded game.
        Call this whenever the student asks how the pawn structure evolved.

        The result includes square-level events, file-status labels, king-shield
        facts, and language guardrails. Tell the story from `chronology`, in order,
        without skipping a step that undoes an earlier one. Do not use structural
        terms like "open file", "collapsed", "isolated", "passed", or "pinned" unless
        this tool returned them.

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
        # Lichess redirects anonymous GET /game/export/{id} to /signup; the bulk
        # export endpoint still serves single games with server evals and clocks.
        pgn_text = external.lichess_game_pgn(game_id)
        if pgn_text is None:
            return {"error": ("Lichess did not return the game (network failure or a non-PGN response). "
                              "Tell the student the download failed; do not suggest the game is private.")}
        if not pgn_text.strip():
            return {"error": f"Game with ID '{game_id}' not found."}
        try:
            return self._review_and_load(pgn_text, source_game_id=game_id, resume=(mode == "continue"))
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
        if review.student_color is None:
            self.timeline.clear()
            self.state.pending_unmatched_review = {
                "review": review,
                "source": f"downloaded game {source_game_id}" if source_game_id else "downloaded game",
                "source_game_id": source_game_id,
            }
            self.state.review_context = {
                "ownership": "unmatched_pending",
                "dossier_consent": False,
            }
            return {
                "pov_choice_required": True,
                "white": review.headers.get("White", "?"),
                "black": review.headers.get("Black", "?"),
                "default_dossier_consent": False,
                "message": (
                    "No configured username matched this game. Ask the student which POV to review "
                    "(white or black). Tell them dossier saving is off by default for borrowed games; "
                    "they can add 'save' if they explicitly want this game logged."
                ),
            }
        self.state.pending_unmatched_review.clear()
        self.timeline.load_game(review.game)
        self.state.review = review
        self.state.current_game_id = source_game_id
        self.state.review_context = {
            "ownership": "own",
            "pov": review.student_color,
            "dossier_consent": True,
        }
        rating_enrichment = current_rating_enrichment(review, self.config)
        if rating_enrichment:
            self.state.review_context["rating_enrichment"] = rating_enrichment
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
                "review_digest": review.digest() + rating_offer_note(rating_enrichment),
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
        start_self_assessment(self.state, self.config, review)
        self.state.lesson_metadata = meta
        if source_game_id and getattr(self.config, "lesson_file", None):
            save_lesson_metadata(self.config.lesson_file, meta)
        opener = self_assessment_open_note(self.state, review)
        return {"review_digest": review.digest() + rating_offer_note(rating_enrichment) + opener,
                "self_assessment_opener": opener,
                "board_state": "Loaded at the STARTING position. The user steps through with /next."}

    def get_tool_list(self):
        """Returns the bound methods to be passed to the Gemini API."""
        return [
            self.analyze_position, self.evaluate_student_answer, self.evaluate_threat_answer,
            self.evaluate_explanation, self.explain_move,
            self.play_sparring_reply, self.execute_coach_move, self.submit_diagnosis,
            self.get_game_review_summary, self.get_game_position, self.explain_pawn_structure_evolution,
            self.get_opening_repertoire,
            self.query_endgame_tablebase, self.fetch_lichess_game_by_id,
            self.fetch_chessgames_com_game,
        ]
