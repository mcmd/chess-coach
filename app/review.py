"""Game review pipeline.

Classifies every move of a game by evaluation swing (in win-percentage space,
the standard approach, so a -2 -> -5 slide in a lost position is not called a
'blunder'), selects the most instructive moments as a lesson agenda, and
produces a compact digest the LLM coach can teach from.

This module is UI-free: pass `on_progress(done, total)` for progress display.
"""
import io
import math
from dataclasses import dataclass, field

import chess
import chess.pgn

from app.concepts import get_game_phase
from app.motifs import detect_motifs_in_line
from app import trace as tracing

QUALITY_GLYPHS = {
    "blunder": "??", "mistake": "?", "inaccuracy": "?!",
    "only_move": "!", "best": "", "good": "", "ok": "", "forced": "",
}

MATE_CP = 100000


def cp_to_win_pct(cp: float) -> float:
    """Centipawns (mover POV) -> expected win percentage (Lichess model)."""
    cp = max(-1500, min(1500, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def classify_move(drop_pct: float, is_best: bool, is_only: bool, is_forced: bool) -> str:
    if is_forced:
        return "forced"
    if is_only:
        return "only_move"
    if is_best:
        return "best"
    if drop_pct >= 30:
        return "blunder"
    if drop_pct >= 20:
        return "mistake"
    if drop_pct >= 10:
        return "inaccuracy"
    if drop_pct <= 2:
        return "good"
    return "ok"


def time_flag_for(quality: str, time_spent, clock_remaining, is_forced: bool):
    if time_spent is None:
        return None
    if quality in ("blunder", "mistake") and time_spent < 10:
        if clock_remaining is not None and clock_remaining < 30:
            return "time_scramble"
        if clock_remaining is None or clock_remaining > 60:
            return "hope_chess"
    if is_forced and time_spent > 60:
        return "freezing"
    return None


@dataclass
class MoveRecord:
    ply: int                 # 0-based index into the mainline
    number: int              # fullmove number
    color: str               # "white" / "black"
    san: str
    uci: str
    fen_before: str
    fen_after: str
    eval_before_cp: int      # white POV
    eval_after_cp: int       # white POV
    drop_pct: float          # win% the mover lost with this move (>= 0)
    quality: str
    best_move_san: str = None
    is_capture: bool = False
    time_spent: float = None
    clock_remaining: float = None
    time_flag: str = None
    # Agenda enrichment (populated only for critical moments):
    is_agenda: bool = False
    best_line: list = field(default_factory=list)
    punishment_line: list = field(default_factory=list)
    motifs: list = field(default_factory=list)
    maia_top: list = field(default_factory=list)
    maia_played_prob: float = None
    was_maia_top: bool = False              # student played Maia's most likely human move
    avoided_natural_mistake: str = None     # praise: sidestepped the statistically natural error
    brilliancy_note: str = None             # praise: only good move AND peers rarely find it
    error_typicality: str = None            # "typical_for_level" | "atypical_for_level" | None
    best_move_human_prob: float = None      # findability: % of peers who play the engine-best move
    phase: str = None                       # opening / middlegame / endgame at the time of the move
    human_win_prob: float = None            # Maia: how often peers actually score from here (mover POV)
    likely_wrong_answer: str = None         # the guess the student will most likely offer, pre-refuted
    aspirational: dict = None               # what stronger players choose here ({elo, move, probability})
    board_token: str = None  # pre-minted diagram token for the position before the move
    best_line_motifs: list = field(default_factory=list)  # tactics the student MISSED
    suggested_categories: list = field(default_factory=list)
    position_facts: str = None  # backend-verified positional facts for the position before the move

    @property
    def move_label(self) -> str:
        dots = "." if self.color == "white" else "..."
        return f"{self.number}{dots} {self.san}"


@dataclass
class GameReview:
    headers: dict
    records: list
    student_color: str          # "white" / "black" / None
    agenda_plies: list          # plies of agenda records, in game order
    game: chess.pgn.Game = field(repr=False, default=None)
    extra_games_ignored: bool = False
    white_elo: int = None
    black_elo: int = None

    def record_at(self, ply: int):
        return self.records[ply] if 0 <= ply < len(self.records) else None

    @property
    def agenda(self):
        return [self.records[p] for p in self.agenda_plies]

    def annotated_movetext(self) -> str:
        parts = []
        for r in self.records:
            glyph = "!!" if r.brilliancy_note else QUALITY_GLYPHS.get(r.quality, "")
            if r.color == "white":
                parts.append(f"{r.number}. {r.san}{glyph}")
            else:
                parts.append(f"{r.san}{glyph}")
        return " ".join(parts)

    def digest(self) -> str:
        h = self.headers
        lines = ["=== GAME REVIEW (backend-verified by Stockfish) ==="]
        lines.append(
            f"White: {h.get('White', '?')} ({h.get('WhiteElo', '?')}) | "
            f"Black: {h.get('Black', '?')} ({h.get('BlackElo', '?')}) | "
            f"Result: {h.get('Result', '?')}"
        )
        if h.get("Opening"):
            lines.append(f"Opening: {h.get('Opening')} ({h.get('ECO', '')})")
        if self.student_color:
            lines.append(f"The STUDENT played {self.student_color.upper()}. Coach that side.")
        else:
            lines.append("Student color unknown — ask the student which side they played before coaching.")
        if self.extra_games_ignored:
            lines.append("(Note: input contained multiple games; only the first was reviewed.)")

        counts = {}
        for r in self.records:
            if self.student_color and r.color != self.student_color:
                continue
            counts[r.quality] = counts.get(r.quality, 0) + 1
        summary = ", ".join(f"{counts[q]} {q}(s)" for q in
                            ("blunder", "mistake", "inaccuracy", "only_move") if counts.get(q))
        if summary:
            lines.append(f"Student move quality: {summary}.")

        student_records = [r for r in self.records
                           if self.student_color is None or r.color == self.student_color]
        with_maia = [r for r in student_records if r.maia_top]
        if with_maia:
            matched = sum(1 for r in with_maia if r.was_maia_top)
            by_phase = {}
            for r in with_maia:
                hit, total = by_phase.get(r.phase or "?", (0, 0))
                by_phase[r.phase or "?"] = (hit + (1 if r.was_maia_top else 0), total + 1)
            breakdown = ", ".join(f"{ph} {hit}/{total}" for ph, (hit, total) in by_phase.items())
            lines.append(f"Human-alignment: the student played Maia's most-likely human move in "
                         f"{matched}/{len(with_maia)} moves ({breakdown}). Big divergence in one "
                         "phase shows WHERE the student differs from peers.")
        praised = [(r, r.brilliancy_note or r.avoided_natural_mistake)
                   for r in student_records
                   if r.brilliancy_note or r.avoided_natural_mistake][:3]
        if praised:
            lines.append("PRAISE-WORTHY moments (brilliancies and sidestepped traps — "
                         "recognize these explicitly during the review):")
            for r, text in praised:
                lines.append(f"  {r.move_label}: {text}")

        lines.append("")
        lines.append("Annotated moves (?? blunder, ? mistake, ?! inaccuracy, ! only-move):")
        lines.append(self.annotated_movetext())
        lines.append("")

        if not self.agenda_plies:
            lines.append("No critical moments detected (no move lost 10+ win%). "
                         "Praise the student's accuracy and offer to discuss plans or openings.")
            return "\n".join(lines)

        lines.append("CRITICAL MOMENTS — the lesson agenda. Work through them ONE AT A TIME, in order.")
        for i, r in enumerate(self.agenda, 1):
            wb = cp_to_win_pct(r.eval_before_cp if r.color == "white" else -r.eval_before_cp)
            wa = cp_to_win_pct(r.eval_after_cp if r.color == "white" else -r.eval_after_cp)
            lines.append(f"[Moment {i}] {r.move_label} by {r.color} — {r.quality} "
                         f"(win% {wb:.0f} -> {wa:.0f}; eval {r.eval_before_cp / 100.0:+.2f} -> {r.eval_after_cp / 100.0:+.2f}).")
            lines.append(f"  FEN before the move: {r.fen_before}")
            if r.board_token:
                lines.append(f"  Board diagram token for this position: {r.board_token} "
                             "(paste this EXACT token to show the diagram)")
            if r.position_facts:
                lines.append(f"  Verified position facts (cite ONLY these for positional claims): {r.position_facts}")
            if r.maia_played_prob is not None:
                lines.append(f"  Maia: ~{r.maia_played_prob:.0%} of similar-rated players choose {r.san}. "
                             f"Top human moves: " + ", ".join(f"{m['move']} ({m['probability']:.0%})" for m in r.maia_top))
            if r.error_typicality == "typical_for_level":
                lines.append("  Typicality: this mistake is COMMON at this level — normalize it first, "
                             "then teach the underlying pattern thoroughly (it is a real knowledge gap).")
            elif r.error_typicality == "atypical_for_level":
                lines.append("  Typicality: players at this level RARELY play this — treat it as a "
                             "process slip (time, attention, tilt), NOT a knowledge gap. Check the clock "
                             "data and coach the habit, not the pattern.")
            if r.best_move_human_prob is not None:
                lines.append(f"  Findability: ~{r.best_move_human_prob:.0%} of similar-rated players find "
                             "the right move here — calibrate your hints and your praise accordingly.")
            if r.human_win_prob is not None:
                engine_pct = cp_to_win_pct(r.eval_before_cp if r.color == "white" else -r.eval_before_cp)
                human_pct = r.human_win_prob * 100
                if abs(engine_pct - human_pct) >= 20:
                    lines.append(f"  Practical difficulty: the engine win% here is {engine_pct:.0f}, but players "
                                 f"at this level actually score ~{human_pct:.0f}% — the position is "
                                 + ("HARDER to play than the eval suggests; teach technique, not the number."
                                    if human_pct < engine_pct else
                                    "easier in practice than the eval suggests."))
            if r.time_spent is not None:
                flag = f" (flag: {r.time_flag})" if r.time_flag else ""
                lines.append(f"  Time: spent {int(r.time_spent)}s, {int(r.clock_remaining)}s remaining{flag}.")
            secret = []
            if r.best_move_san:
                secret.append(f"better was {r.best_move_san}" +
                              (f" (line: {' '.join(r.best_line)})" if r.best_line else ""))
            if r.best_line_motifs:
                secret.append("what the student missed: " + "; ".join(f"{m['motif']}: {m['detail']}" for m in r.best_line_motifs))
            if r.punishment_line:
                secret.append(f"the punishment of {r.san} is: {' '.join(r.punishment_line)}")
            if r.motifs:
                secret.append("motifs in the punishment: " + "; ".join(f"{m['motif']}: {m['detail']}" for m in r.motifs))
            if r.likely_wrong_answer:
                secret.append(f"most likely student guess: {r.likely_wrong_answer}")
            if r.aspirational:
                secret.append(f"aspirational instinct: at ~{r.aspirational['elo']} Elo most players choose "
                              f"{r.aspirational['move']} ({r.aspirational['probability']:.0%}) — the instinct "
                              "the student is growing toward; share it AFTER the moment is resolved")
            if secret:
                lines.append("  WITHHELD ANSWER (do NOT reveal until the student attempts; "
                             "verify attempts with evaluate_student_answer): " + " | ".join(secret))
            if r.suggested_categories:
                lines.append(f"  Suggested diagnosis category: {' or '.join(r.suggested_categories)} "
                             "(use it unless the conversation reveals a different root cause).")
        lines.append("")
        lines.append("Coaching protocol: for each moment, set the scene at the FEN BEFORE the move, "
                     "ask ONE focused question (e.g. 'What was the threat?' or 'Find the better move'), "
                     "grade answers with evaluate_student_answer, then teach the named motif/concept. "
                     "Log each diagnosed mistake once with submit_diagnosis.")
        return "\n".join(lines)


def _terminal_eval_cp(board: chess.Board):
    if board.is_checkmate():
        return -MATE_CP if board.turn == chess.WHITE else MATE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return None


def _header_elo(headers, key, fallback):
    try:
        return int(headers.get(key, ""))
    except (TypeError, ValueError):
        return fallback


def detect_student_color(headers, config):
    name = (getattr(config, "player_name", None) or "").strip().casefold()
    if not name:
        return None
    if name in headers.get("White", "").casefold():
        return "white"
    if name in headers.get("Black", "").casefold():
        return "black"
    return None


def review_game(pgn_text: str, engines, config, on_progress=None, agenda_size: int = 4):
    """Full review of the first game in `pgn_text`. Returns a GameReview, or
    None if no game with moves could be parsed."""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return None
    nodes = list(game.mainline())
    if not nodes:
        return None
    extra = chess.pgn.read_game(pgn_io) is not None

    headers = dict(game.headers)
    student_color = detect_student_color(headers, config)

    board = game.board()
    records = []
    total = len(nodes)
    prev_eval_cp = None
    prev_best_uci = None
    white_elo = _header_elo(headers, "WhiteElo", getattr(config, "user_elo", 1200))
    black_elo = _header_elo(headers, "BlackElo", getattr(config, "user_elo", 1200))

    tracing.trace(f"review: scanning {total} plies (Stockfish quick eval + Maia prediction "
                  "per move; engine-level traces suppressed during the scan)")
    for ply, node in enumerate(nodes):
        move = node.move
        fen_before = board.fen()

        eval_time = float(getattr(config, "review_eval_time", 0.5))
        if prev_eval_cp is None:
            eval_before_cp, best_uci = _eval_with_best(engines, board, eval_time)
        else:
            eval_before_cp, best_uci = prev_eval_cp, prev_best_uci

        mover_white = board.turn == chess.WHITE
        san = board.san(move)
        uci = move.uci()
        is_capture = board.is_capture(move)
        is_forced = board.legal_moves.count() == 1
        is_best = best_uci is not None and uci == best_uci
        phase = get_game_phase(board)

        # Clock data: a player's previous clock reading is two plies back.
        time_spent = clock_remaining = None
        clk = node.clock()
        if clk is not None:
            clock_remaining = clk
            if ply >= 2 and nodes[ply - 2].clock() is not None:
                time_spent = max(0.0, nodes[ply - 2].clock() - clk)

        board.push(move)
        fen_after = board.fen()
        eval_after_cp, next_best = _eval_with_best(engines, board, eval_time)

        before_m = eval_before_cp if mover_white else -eval_before_cp
        after_m = eval_after_cp if mover_white else -eval_after_cp
        drop = max(0.0, cp_to_win_pct(before_m) - cp_to_win_pct(after_m))

        # Maia on EVERY move: the human-vs-engine comparison is high-quality
        # signal both for normalizing errors and for spotting praise moments.
        maia_top, maia_played_prob, was_maia_top, avoided = [], None, False, None
        mover_elo = white_elo if mover_white else black_elo
        oppo_elo = black_elo if mover_white else white_elo
        with tracing.suspended():
            maia = engines.maia_predict(fen_before, elo_self=mover_elo, elo_oppo=oppo_elo)
        best_move_human_prob = None
        human_win_prob = None
        if maia:
            human_win_prob = maia.get("win_probability_for_side_to_move")
            maia_top = maia["top_moves"]
            prob = maia["raw_probs"].get(uci)
            maia_played_prob = round(float(prob), 3) if prob is not None else None
            if best_uci:
                best_prob = maia["raw_probs"].get(best_uci)
                best_move_human_prob = round(float(best_prob), 3) if best_prob is not None else 0.0
            natural_uci = max(maia["raw_probs"], key=maia["raw_probs"].get)
            natural_prob = maia["raw_probs"][natural_uci]
            was_maia_top = natural_uci == uci
            if not was_maia_top and drop < 5 and natural_prob >= 0.35:
                avoided = _check_avoided_natural_mistake(
                    engines, fen_before, mover_white, before_m, natural_uci, natural_prob, san, eval_time)

        is_only = False
        if is_best and not is_forced and abs(before_m) < 400:
            is_only = _check_only_move(engines, fen_before)

        quality = classify_move(drop, is_best, is_only, is_forced)

        # Typicality lens: brilliancies peers miss, errors peers share, errors
        # peers DON'T make (process slips, not knowledge gaps).
        brilliancy = None
        typicality = None
        if maia:
            played_prob = maia_played_prob if maia_played_prob is not None else 0.0
            if quality == "only_move" and played_prob <= 0.15:
                brilliancy = (f"{san} was the ONLY good move, and only ~{played_prob:.0%} of "
                              "similar-rated players find it")
            if quality in ("blunder", "mistake"):
                if played_prob >= 0.30:
                    typicality = "typical_for_level"
                elif played_prob <= 0.08:
                    typicality = "atypical_for_level"
        records.append(MoveRecord(
            ply=ply, number=(ply // 2) + 1, color="white" if mover_white else "black",
            san=san, uci=uci, fen_before=fen_before, fen_after=fen_after,
            eval_before_cp=eval_before_cp, eval_after_cp=eval_after_cp,
            drop_pct=round(drop, 1), quality=quality,
            best_move_san=_uci_to_san(fen_before, best_uci),
            is_capture=is_capture, time_spent=time_spent, clock_remaining=clock_remaining,
            time_flag=time_flag_for(quality, time_spent, clock_remaining, is_forced),
            maia_top=maia_top, maia_played_prob=maia_played_prob,
            was_maia_top=was_maia_top, avoided_natural_mistake=avoided,
            brilliancy_note=brilliancy, error_typicality=typicality,
            best_move_human_prob=best_move_human_prob,
            phase=phase, human_win_prob=human_win_prob,
        ))
        prev_eval_cp, prev_best_uci = eval_after_cp, next_best
        if on_progress:
            on_progress(ply + 1, total)

    review = GameReview(headers=headers, records=records, student_color=student_color,
                        agenda_plies=[], game=game, extra_games_ignored=extra,
                        white_elo=white_elo, black_elo=black_elo)
    review.agenda_plies = _select_agenda(records, student_color, agenda_size)
    _enrich_agenda(review, engines)
    return review


def _eval_with_best(engines, board: chess.Board, eval_time: float = 0.5):
    terminal = _terminal_eval_cp(board)
    if terminal is not None:
        return terminal, None
    try:
        with tracing.suspended():  # bulk scan: one trace line would become hundreds
            score, best_uci = engines.quick_eval(board.fen(), eval_time)
        return score.white().score(mate_score=MATE_CP), best_uci
    except Exception:
        return 0, None


def _check_avoided_natural_mistake(engines, fen_before, mover_white, before_m,
                                   natural_uci, natural_prob, played_san, eval_time=0.5):
    """The student declined the move most humans play here. If that natural
    move loses significant win%, this is an earned praise moment."""
    try:
        nb = chess.Board(fen_before)
        natural_move = chess.Move.from_uci(natural_uci)
        if natural_move not in nb.legal_moves:
            return None
        natural_san = nb.san(natural_move)
        nb.push(natural_move)
        terminal = _terminal_eval_cp(nb)
        if terminal is not None:
            natural_cp = terminal
        else:
            with tracing.suspended():
                score, _ = engines.quick_eval(nb.fen(), eval_time)
            natural_cp = score.white().score(mate_score=MATE_CP)
        natural_m = natural_cp if mover_white else -natural_cp
        natural_drop = cp_to_win_pct(before_m) - cp_to_win_pct(natural_m)
        if natural_drop >= 15:
            return (f"{natural_prob:.0%} of similar-rated players choose {natural_san} here, "
                    f"which loses {natural_drop:.0f} win% — the student played {played_san} instead")
    except Exception:
        pass
    return None


def _check_only_move(engines, fen: str) -> bool:
    """Was the best move dramatically better than the second best?"""
    try:
        board = chess.Board(fen)
        with tracing.suspended():
            infos = engines.top_lines(board, multipv=2, depth=10, time_cap=0.4)
        if len(infos) < 2:
            return False
        pov = board.turn
        best = infos[0]["score"].pov(pov).score(mate_score=MATE_CP)
        second = infos[1]["score"].pov(pov).score(mate_score=MATE_CP)
        return (best - second) >= 200
    except Exception:
        return False


def _uci_to_san(fen: str, uci: str):
    if not uci:
        return None
    try:
        board = chess.Board(fen)
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, AssertionError):
        return None


def _select_agenda(records, student_color, agenda_size):
    def candidates(color_filter):
        return [r for r in records
                if r.drop_pct >= 10 and (color_filter is None or r.color == color_filter)]

    pool = candidates(student_color) or candidates(None)
    pool.sort(key=lambda r: r.drop_pct, reverse=True)
    chosen = sorted(r.ply for r in pool[:agenda_size])
    return chosen


def _enrich_agenda(review: GameReview, engines):
    for ply in review.agenda_plies:
        r = review.records[ply]
        r.is_agenda = True
        tracing.trace(f"review: deep-analyzing agenda moment {r.move_label} "
                      f"({r.quality}, -{r.drop_pct} win%)")
        board_before = chess.Board(r.fen_before)
        board_after = chess.Board(r.fen_after)

        try:
            infos = engines.top_lines(board_before, multipv=3, depth=18, time_cap=3.0)
            pv = infos[0].get("pv") or []
            r.best_line = _pv_to_san(board_before, pv[:8])
            r.best_line_motifs = detect_motifs_in_line(board_before, r.best_line)
        except Exception:
            pass

        try:
            infos = engines.top_lines(board_after, multipv=1, depth=16, time_cap=2.0)
            pv = infos[0].get("pv") or []
            r.punishment_line = _pv_to_san(board_after, pv[:8])
            r.motifs = detect_motifs_in_line(board_after, r.punishment_line)
        except Exception:
            pass
        r.suggested_categories = suggest_error_categories(r)
        r.position_facts = position_facts(board_before)
        # Per-move Maia data was already collected during the main scan.
        _anticipate_wrong_answer(r, engines, board_before)
        _aspirational_instinct(r, engines, review)


def _anticipate_wrong_answer(r: MoveRecord, engines, board_before: chess.Board) -> None:
    """The student's most likely guess to 'find the better move' is Maia's top
    pick (excluding what they played and the actual answer). Pre-grade it so
    the coach can respond to the real attempt instantly and precisely."""
    guess = next((m for m in r.maia_top
                  if m["move"] != r.san and m["move"] != r.best_move_san), None)
    if not guess:
        return
    try:
        gb = chess.Board(r.fen_before)
        gb.push(gb.parse_san(guess["move"]))
        terminal = _terminal_eval_cp(gb)
        if terminal is not None:
            guess_cp = terminal
        else:
            with tracing.suspended():
                score, _ = engines.quick_eval(gb.fen())
            guess_cp = score.white().score(mate_score=MATE_CP)
        mover_white = r.color == "white"
        before_m = r.eval_before_cp if mover_white else -r.eval_before_cp
        guess_m = guess_cp if mover_white else -guess_cp
        guess_drop = cp_to_win_pct(before_m) - cp_to_win_pct(guess_m)
        if guess_drop >= 10:
            infos = engines.top_lines(gb, multipv=1, depth=12, time_cap=1.0)
            pv = infos[0].get("pv") or []
            refutation = " ".join(_pv_to_san(gb, pv[:6]))
            r.likely_wrong_answer = (f"{guess['move']} (~{guess['probability']:.0%} of peers try it) — "
                                     f"loses {guess_drop:.0f} win%; refute with: {refutation}")
        elif guess_drop <= 5:
            r.likely_wrong_answer = (f"{guess['move']} (~{guess['probability']:.0%} of peers try it) — "
                                     "actually a decent alternative; give credit if the student offers it")
    except Exception:
        pass


def _aspirational_instinct(r: MoveRecord, engines, review: GameReview) -> None:
    """What do players a few hundred points stronger choose here? The next
    level's instinct, as a concrete target."""
    mover_elo = (review.white_elo if r.color == "white" else review.black_elo) or 1200
    target_elo = min(mover_elo + 350, 1900)
    if target_elo <= mover_elo + 50:
        return
    asp = engines.maia_predict(r.fen_before, elo_self=target_elo, elo_oppo=target_elo)
    if asp and asp["top_moves"]:
        r.aspirational = {"elo": target_elo, **asp["top_moves"][0]}


def position_facts(board: chess.Board) -> str:
    """Compact, backend-verified positional facts for one position, so the
    coach's prose color (open files, hanging pieces, king exposure) is grounded
    instead of imagined. Empty categories are stated explicitly where silence
    would invite invention (open files), otherwise omitted."""
    from app.concepts import (_analyze_king_perimeter, _get_loose_pieces,
                              _get_overloaded_defenders, _get_passed_pawns,
                              _get_pawn_data, _get_pawn_tension, _get_piece_safety,
                              _get_trapped_pieces, _has_opposite_colored_bishops,
                              _is_back_rank_vulnerable)
    files = "abcdefgh"
    w = _get_pawn_data(board, chess.WHITE)
    b = _get_pawn_data(board, chess.BLACK)
    open_files = [f for f in files if w["file_counts"][f] == 0 and b["file_counts"][f] == 0]
    half_w = [f for f in files if w["file_counts"][f] == 0 and b["file_counts"][f] > 0]
    half_b = [f for f in files if b["file_counts"][f] == 0 and w["file_counts"][f] > 0]

    parts = ["open files: " + (", ".join(open_files) if open_files else "NONE")]
    if half_w:
        parts.append("half-open for White: " + ", ".join(half_w))
    if half_b:
        parts.append("half-open for Black: " + ", ".join(half_b))

    tension = _get_pawn_tension(board)
    if tension:
        parts.append("pawn tension: " + ", ".join(tension))
    if _has_opposite_colored_bishops(board):
        parts.append("opposite-colored bishops (attacker-friendly middlegame, drawish endgame)")

    for color, label in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
        safety = _get_piece_safety(board, color)
        if safety["hanging"]:
            parts.append(f"{label} hanging: " + ", ".join(safety["hanging"]))
        if safety["weakly_defended"]:
            parts.append(f"{label} weakly defended: " + ", ".join(safety["weakly_defended"]))
        loose = _get_loose_pieces(board, color)
        if loose:
            parts.append(f"{label} loose (zero defenders): " + ", ".join(loose))
        overloaded = _get_overloaded_defenders(board, color)
        if overloaded:
            parts.append(f"{label} overloaded: " + "; ".join(overloaded))
        trapped = _get_trapped_pieces(board, color)
        if trapped:
            parts.append(f"{label} trapped: " + ", ".join(trapped))
        if _is_back_rank_vulnerable(board, color):
            parts.append(f"{label} king has NO luft (back-rank vulnerable)")
        perimeter = _analyze_king_perimeter(board, color)
        if perimeter["enemy_pieces_attacking_perimeter"]:
            parts.append(f"enemy pieces attacking {label} king perimeter ({perimeter['king_square']}): "
                         + ", ".join(perimeter["enemy_pieces_attacking_perimeter"]))
        passed = _get_passed_pawns(board, color)
        if passed:
            parts.append(f"{label} passed pawns: " + ", ".join(passed))
    return " | ".join(parts)


def suggest_error_categories(record: MoveRecord) -> list:
    """Evidence-based diagnosis suggestions: tactics in the line the student
    MISSED map to missed_*; tactics in the opponent's punishment map to what
    the student failed to see coming."""
    suggestions = []
    missed_kinds = {m["motif"] for m in record.best_line_motifs}
    punish_kinds = {m["motif"] for m in record.motifs}

    if any(k.endswith("fork") for k in missed_kinds):
        suggestions.append("missed_fork")
    if missed_kinds & {"pin", "skewer"}:
        suggestions.append("missed_pin_or_skewer")
    if missed_kinds & {"checkmate", "back_rank_mate", "smothered_mate"}:
        suggestions.append("missed_mate")
    if missed_kinds & {"discovered_attack", "discovered_check", "double_check", "zwischenzug"}:
        suggestions.append("missed_tactic")

    if "zwischenzug" in punish_kinds:
        # The student likely calculated an exchange assuming recaptures were forced.
        suggestions.append("calculation_error")
    if punish_kinds & {"checkmate", "back_rank_mate", "smothered_mate"}:
        suggestions.append("king_safety")
    elif (any(k.endswith("fork") for k in punish_kinds)
          or punish_kinds & {"pin", "skewer", "discovered_check", "discovered_attack",
                             "double_check", "removal_of_defender"}):
        suggestions.append("ignored_threat")

    if record.time_flag in ("hope_chess", "time_scramble"):
        suggestions.append("time_management")
    return suggestions[:2]


def attach_board_tokens(review: GameReview, state, invert: bool = False) -> None:
    """Pre-mint a real diagram token for each agenda moment so the model never
    has a reason to invent one. Oriented to the student's side."""
    from app.render import render_board_token
    pov = review.student_color or "white"
    for record in review.agenda:
        board = chess.Board(record.fen_before)
        record.board_token = render_board_token(state, board, pov, invert)


def _pv_to_san(board: chess.Board, pv) -> list:
    sans, b = [], board.copy(stack=False)
    for m in pv:
        try:
            sans.append(b.san(m))
            b.push(m)
        except (ValueError, AssertionError):
            break
    return sans
