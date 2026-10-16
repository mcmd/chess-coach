"""Game review pipeline.

Classifies every move of a game by evaluation swing (in win-percentage space,
the standard approach, so a -2 -> -5 slide in a lost position is not called a
'blunder'), selects the most instructive moments as a lesson agenda, and
produces a compact digest the LLM coach can teach from.

This module is UI-free: pass `on_progress(done, total)` for progress display.
"""
import io
import math
import re
from dataclasses import dataclass, field
from typing import Literal

import chess
import chess.pgn

from app.concepts import get_game_phase
from app.config import DEFAULT_USER_ELO
from app.engines import history_key
from app.human_danger import empty_human_danger, human_danger_mass_for_record
from app.motifs import detect_motifs_in_line
from app import trace as tracing

QUALITY_GLYPHS = {
    "blunder": "??", "mistake": "?", "inaccuracy": "?!",
    "only_move": "!", "best": "", "good": "", "ok": "", "forced": "",
}

MATE_CP = 100000
MATING_LINE_MAX_PLIES = 19   # a whole mate in 10; longer mating lines are cut here


def cp_to_win_pct(cp: float) -> float:
    """Centipawns (mover POV) -> expected win percentage (Lichess model)."""
    cp = max(-1500, min(1500, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def classify_move(drop_pct: float, is_best: bool, is_only: bool, is_forced: bool) -> str:
    if is_forced:
        return "forced"
    # The calibrated win%-loss stream owns severity. Local best-move flags can
    # disagree with PGN evals or shallow searches; they must not relabel a large
    # measured loss as praise.
    if drop_pct >= 30:
        return "blunder"
    if drop_pct >= 20:
        return "mistake"
    if drop_pct >= 10:
        return "inaccuracy"
    if is_only:
        return "only_move"
    if is_best:
        return "best"
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
    punisher_uci: str = None     # engine's best reply to the played move (free from the scan)
    history_uci: list = field(default_factory=list)  # UCI moves reaching fen_before (history-aware maia3)
    prev_move_san: str = None    # the opponent's move that led into fen_before (game orientation)
    in_check: bool = False       # side to move is already in check at fen_before
    # Agenda enrichment (populated only for critical moments):
    is_agenda: bool = False
    best_line: list = field(default_factory=list)
    punishment_line: list = field(default_factory=list)
    motifs: list = field(default_factory=list)
    maia_top: list = field(default_factory=list)
    maia_played_prob: float = None
    maia_elo: int = None                    # rating Maia was asked about (the mover's game-time Elo)
    was_maia_top: bool = False              # student played Maia's most likely human move
    avoided_natural_mistake: str = None     # praise: sidestepped the statistically natural error
    brilliancy_note: str = None             # praise: only good move AND peers rarely find it
    error_typicality: str = None            # "typical_for_level" | "atypical_for_level" | None
    best_move_human_prob: float = None      # findability: % of peers who play the engine-best move
    phase: str = None                       # opening / middlegame / endgame at the time of the move
    human_win_prob: float = None            # Maia: how often peers actually score from here (mover POV)
    human_draw_prob: float = None           # Maia WDL draw share, mover POV (maia3 only)
    human_loss_prob: float = None           # Maia WDL loss share, mover POV (maia3 only)
    likely_wrong_answer: str = None         # the guess the student will most likely offer, pre-refuted
    aspirational: dict = None               # what stronger players choose here ({elo, move, probability})
    book_share: float = None                # share of master games playing this move (it IS theory)
    tablebase_note: str = None              # provable endgame truth about this move
    is_opening_deviation: bool = False      # the move where the student left established theory
    opening_theory_brief: str = None        # withheld theory answer (master + peer moves with scores)
    practical_note: str = None              # PRACTICAL (sharp but holdable) vs CONCRETE (refuted by force)
    saving_resource: str = None             # withheld: the narrow defense after the mistake, with findability
    board_token: str = None  # pre-minted diagram token for the position before the move
    best_line_motifs: list = field(default_factory=list)  # tactics the student MISSED
    deep_eval_before_cp: int = None  # white POV: best play from fen_before at agenda depth (vs the scan eval)
    engine_only_refutation: str = None  # the better move loses only to a deep engine line (practical framing)
    suggested_categories: list = field(default_factory=list)
    position_facts: str = None  # backend-verified positional facts for the position before the move
    lesson_objective: str = None
    recommended_diagnosis: str = None
    initial_question: str = None
    question_ladder: dict = field(default_factory=dict)
    explanation_primitives: dict = field(default_factory=dict)
    human_danger: dict = field(default_factory=dict)  # provisional internal P0-1b ranking signal
    agenda_score: dict = field(default_factory=dict)  # non-narrated criticality components
    # Exact draw rule reached by this move (threefold/fivefold repetition, fifty/seventy-five-move rule).
    # Rules beat engine and imported evals: 2dtDqQgR's 31. Kh1 repeated the position a third time, yet
    # Lichess's [%eval] of the final position (-4.45) made it a "blunder" taught as a mistake.
    draw_rule: str = None
    eval_without_draw_rule_cp: int = None  # white POV: the eval the draw rule replaced (play on, no claim)

    @property
    def move_label(self) -> str:
        dots = "." if self.color == "white" else "..."
        return f"{self.number}{dots} {self.san}"

    @property
    def best_move_label(self) -> str:
        if not self.best_move_san:
            return None
        dots = "." if self.color == "white" else "..."
        return f"{self.number}{dots} {self.best_move_san}"


@dataclass
class ReviewTheme:
    """A verified cross-moment cluster of the single-game agenda (P0-1a). The
    synthesis reads these instead of inventing a theme, and must NOT assert a theme
    across moments the backend placed in different clusters. Seeded minimal now so
    the P1-18 theme-lifecycle can extend this shape rather than rewrite it."""
    id: str
    label: str
    plies: list                 # agenda plies in this cluster, in game order
    confidence: Literal["high", "medium", "low"]
    safe_summary: str
    avoid_language: list = field(default_factory=list)


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
    opening_name: str = None
    opening_eco: str = None
    opening_deviation: dict = None   # where the game left established theory
    tablebase_entry: dict = None     # where the game entered provable endgame territory
    slides: list = None              # within-game eval-only one-sided slide segments
    eval_source: str = "engine"      # "pgn" (used the PGN's [%eval]) or "engine" (local search)
    themes: list = field(default_factory=list)  # ReviewTheme clusters (verified single-game theming)
    conversation_elo: int = None     # the STUDENT'S level (config.user_elo); anchors coaching/aspiration.
                                     # Distinct from white_elo/black_elo, which are the PGN game-time
                                     # (move-typicality) ratings. Plan P0-4a "two purposes, one anchor".

    def record_at(self, ply: int):
        return self.records[ply] if 0 <= ply < len(self.records) else None

    def board_before(self, ply: int) -> chess.Board:
        """The position before `ply` with the game's moves on its stack, so engine searches see
        repetitions (a bare `chess.Board(record.fen_before)` cannot)."""
        board = chess.Board(self.records[0].fen_before)
        for record in self.records[:ply]:
            board.push_uci(record.uci)
        return board

    @property
    def agenda(self):
        return [self.records[p] for p in self.agenda_plies]

    def ending(self) -> str | None:
        """How the game ended, from the final position and the PGN headers. Live eval 5kf7rqwj:
        2dtDqQgR was drawn by repetition right after 31. Kh1, and the coach, not told so, credited
        the draw to the recommended 31. Kh2."""
        if not self.records:
            return None
        last = self.records[-1]
        board = chess.Board(last.fen_after)
        result = self.headers.get("Result", "*")
        label = last.move_label
        key = lambda fen: " ".join(fen.split()[:4])  # noqa: E731  position identity for repetition
        repeats = ([key(self.records[0].fen_before)] + [key(r.fen_after) for r in self.records]).count(
            key(last.fen_after))
        if board.is_checkmate():
            text = f"{last.color.capitalize()} delivered checkmate with {label}"
        elif board.is_stalemate():
            text = f"a draw by stalemate after {label}"
        elif board.is_insufficient_material():
            text = f"a draw by insufficient material after {label}"
        elif result == "1/2-1/2" and repeats >= 3:
            ordinal = {3: "third", 4: "fourth", 5: "fifth"}.get(repeats, f"{repeats}th")
            text = (f"a draw by repetition right after {label}: the position after it occurred for the "
                    f"{ordinal} time")
        elif result == "1/2-1/2" and board.halfmove_clock >= 100:
            text = f"a draw by the fifty-move rule after {label}"
        elif "time" in self.headers.get("Termination", "").lower():
            text = f"on time after {label} (result {result})"
        elif result in ("1-0", "0-1"):
            text = f"{'Black' if result == '1-0' else 'White'} resigned after {label}"
        elif result == "1/2-1/2":
            text = f"a draw after {label}, by agreement or claim (no board rule applies)"
        else:
            text = f"no result yet after {label} (result {result})"
        if result == "1/2-1/2" and self.student_color:
            final_cp = last.eval_without_draw_rule_cp if last.draw_rule else last.eval_after_cp
            student_cp = final_cp if self.student_color == "white" else -final_cp
            if student_cp <= -200:
                text += (f" — a good result for the student, who was objectively worse "
                         f"({student_cp / 100:+.2f} at the end)")
        return "The game ended in " + text if not text[0].isupper() else "The game ended: " + text

    def continuation_after(self, r) -> str:
        """What actually happened after `r` in the game, so the coach never credits later events
        or the result to a move that was not played."""
        if r.ply >= len(self.records) - 1:
            return f"This was the final move of the game. {self.ending()}."
        nxt = self.records[r.ply + 1]
        who = "the opponent" if self.student_color and nxt.color != self.student_color else nxt.color.capitalize()
        return f"In the game, {who} replied {nxt.move_label}."

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
        if self.ending():
            lines.append(f"{self.ending()}. Never attribute the result to any other cause, or to a move "
                         "that was not played.")
        if self.opening_deviation:
            d = self.opening_deviation
            line = (f"Opening report: {self.opening_name or 'known theory'}"
                    + (f" ({self.opening_eco})" if self.opening_eco else "")
                    + f". The game left established theory at {d['label']}"
                    + (f" (a move seen in only {d['played_share']:.1%} of master games)"
                       if d['played_games'] else " (a move essentially absent from master games)")
                    + f". Master moves there: {d['theory']}.")
            if d.get("amateur"):
                line += f" Peers at the student's rating play: {d['amateur']}."
            if self.student_color and d["color"] == self.student_color:
                line += " The deviation was the STUDENT'S — if it caused problems, recommend this line as a study item."
            elif self.student_color:
                line += " The deviation was the OPPONENT'S — the student was out of book through no fault of their own."
            lines.append(line)
        elif self.opening_name:
            lines.append(f"Opening report: {self.opening_name} ({self.opening_eco or ''}) — "
                         "the scanned opening moves stayed within established theory.")
        elif h.get("Opening"):
            lines.append(f"Opening: {h.get('Opening')} ({h.get('ECO', '')})")
        if self.tablebase_entry:
            e = self.tablebase_entry
            lines.append(f"Tablebase entry: from {e['move_label']} the position was a PROVEN "
                         f"{e['category'].upper()} for {e['side_to_move']} with perfect play — "
                         "from that point the game is pure technique, and mistakes after it are provable.")
        if self.student_color:
            lines.append(f"The STUDENT played {self.student_color.upper()}. Coach that side.")
            game_elo = self.white_elo if self.student_color == "white" else self.black_elo
            if self.conversation_elo and game_elo and abs(game_elo - self.conversation_elo) > 50:
                lines.append(
                    f"Rating note: move-commonness here is judged at the ~{game_elo} the student was rated "
                    f"in THIS game, but coaching is pitched at their current level (~{self.conversation_elo}). "
                    f"When you say a move is 'common at your level', label which — e.g. 'at the ~{game_elo} "
                    "level you were playing then'.")
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
            opening_alignment = _opening_alignment_facts(student_records)
            if opening_alignment:
                lines.append(opening_alignment)
        praised = [(r, r.brilliancy_note or r.avoided_natural_mistake)
                   for r in student_records
                   if r.brilliancy_note or r.avoided_natural_mistake][:3]
        if praised:
            lines.append("PRAISE-WORTHY moments (brilliancies and sidestepped traps — "
                         "recognize these explicitly during the review):")
            for r, text in praised:
                lines.append(f"  {r.move_label}: {text}")

        if self.slides:
            lines.append("")
            lines.append("WITHIN-GAME SLIDE (cumulative loss, no single blunder):")
            for s in self.slides:
                lines.append(
                    f"  Moves {s['start_move']}-{s['end_move']} ({s['plies']} of the student's moves): "
                    f"winning chances fell from {s['start_winpct']:.0f}% to {s['end_winpct']:.0f}% "
                    f"(net {s['net_loss']:.0f}% lost; no single move lost more than {s['max_single']:.0f}%). "
                    "The damage was CUMULATIVE drift, not one mistake — coach the META-PATTERN "
                    "(consolidating an advantage, staying alert after the critical moment, handling "
                    "rising complexity) using the eval/time facts in the moments. State only what the "
                    "evals show; never mind-read.")

        lines.append("")
        lines.append("Annotated moves (?? blunder, ? mistake, ?! inaccuracy, ! only-move):")
        lines.append(self.annotated_movetext())
        lines.append("")

        if not self.agenda_plies:
            lines.append("No critical moments detected (no move lost 10+ win%). "
                         "Praise the student's accuracy and offer to discuss plans or openings.")
            return "\n".join(lines)

        ply_to_moment = {p: i for i, p in enumerate(self.agenda_plies, 1)}
        if self.themes:
            lines.append("VERIFIED THEMES (backend-clustered by diagnosis category — the synthesis MUST "
                         "use these and MUST NOT assert a single theme across moments in different "
                         "clusters; that over-fit is the exact defect this block fixes):")
            for t in self.themes:
                moments = ", ".join(f"moment {ply_to_moment[p]}" for p in t.plies if p in ply_to_moment)
                lines.append(f"  [{t.label}] ({t.confidence} confidence) — {t.safe_summary}"
                             + (f"  ({moments})" if moments else ""))
            if any(t.confidence == "high" for t in self.themes):
                lines.append("  A high-confidence theme repeats within the game (a real in-game pattern); "
                             "a low-confidence one is a single moment — teach it, but do not generalise it "
                             "into a 'habit' on one instance.")
            lines.append("")

        lines.append("CRITICAL MOMENTS — the lesson agenda. Work through them ONE AT A TIME, in order.")
        for i, r in enumerate(self.agenda, 1):
            lines.extend(moment_digest_lines(i, r, continuation=self.continuation_after(r)))
        lines.append("")
        lines.append("Coaching protocol: for each moment, set the scene at the FEN BEFORE the move, "
                     "ask the Initial question EXACTLY as the first prompt, "
                     "use the hint ladder only after the student has made an attempt or asked for help, "
                     "grade answers with evaluate_student_answer, ask the student WHY the move works, "
                     "then GRADE that 'why' with evaluate_explanation: if it reports a missing "
                     "load-bearing reason, probe the gap (a question or hint, not the answer) before "
                     "moving on; only then teach using the reveal-phase explanation primitives. "
                     "Log each diagnosed mistake once with submit_diagnosis. "
                     "For the closing synthesis, group moments ONLY by the VERIFIED THEMES above — "
                     "never invent a theme that spans moments the backend placed in different clusters.")
        return "\n".join(lines)


def _opening_alignment_facts(student_records: list) -> str:
    """Separate master-book facts from Maia human-commonness facts.
    The coach was observed merging these into "most common human moves for the
    first 7 moves" without the exact backend fact being present."""
    opening = [r for r in student_records if r.phase == "opening"]
    if not opening:
        return ""
    maia_seen = [r for r in opening if r.maia_top]
    book_seen = [r for r in opening if r.book_share]
    parts = []
    if book_seen:
        last = book_seen[-1].move_label
        parts.append(f"book/theory: {len(book_seen)}/{len(opening)} student opening moves were in the masters database through {last}")
    if maia_seen:
        matches = sum(1 for r in maia_seen if r.was_maia_top)
        streak = 0
        for r in maia_seen:
            if r.was_maia_top:
                streak += 1
            else:
                break
        parts.append(f"Maia human-commonness: matched Maia's top move on {matches}/{len(maia_seen)} student opening moves"
                     + (f"; opening streak from move 1 was {streak}" if streak else ""))
    if not parts:
        return ""
    return ("Opening alignment facts (book theory and human-commonness are DIFFERENT; "
            "do not merge them): " + "; ".join(parts) + ".")


def _wdl_risk_profile(win: float, draw: float, loss: float) -> tuple:
    """Translate a Maia WDL spread (mover POV, fractions) into a coaching risk
    verdict. The backend decides the verdict; the coach narrates it. Returns
    (label, guidance); 'balanced' means no notable skew worth a digest line.

    Calibrated to real maia3 output, which (trained on decisive online games) has
    structurally LOW draw in live positions and high draw only in genuinely held
    ones. So we flag only clear skews and deliberately do NOT try to detect
    'sharpness' — a tactical 50/3/47 and a calm 50/3/47 are indistinguishable in
    the value head, so a balanced spread stays 'balanced'."""
    if loss - win >= 0.25:
        return ("must-hold", "you are under real pressure (loss chances clearly exceed winning ones) "
                             "— prioritise solid defence over ambition")
    if win - loss >= 0.25 and win >= 0.55:
        return ("winning", "you hold the better practical chances — convert with low-risk technique "
                           "rather than sharp complications")
    if draw >= 0.50:
        return ("drawish", "drawish at this level (a draw is the most likely result) — don't force "
                           "it; press only with safe, incremental improvements")
    return ("balanced", "")


# A scan/deep disagreement this large changes how the position should be described.
DEEP_EVAL_DISAGREEMENT_CP = 150
# "Practical depth": roughly what a strong club player calculates; refutations that appear only
# well beyond it are engine-only for coaching purposes.
PRACTICAL_DEPTH = 10


def _deep_eval_note(r: MoveRecord) -> str:
    """What the moment header says beyond the scan eval. An engine-only refutation of the better
    move gets the practical framing; otherwise a large scan/deep disagreement is stated as a fact."""
    if r.engine_only_refutation:
        return " " + r.engine_only_refutation
    deep = r.deep_eval_before_cp
    if deep is None or abs(deep - r.eval_before_cp) < DEEP_EVAL_DISAGREEMENT_CP:
        return ""
    return (f" Deep check: with best play the eval before the move is {deep / 100.0:+.2f} "
            f"(the scan said {r.eval_before_cp / 100.0:+.2f}).")


def _practical_refutation(r: MoveRecord, engines, board_before: chess.Board, board_after: chess.Board,
                          deep_best_white_cp) -> str | None:
    """The better move loses only to a long engine line while the played move loses at once.

    Move 31 of 2dtDqQgR (measured 2026-10-13): after 31. Kh2 Stockfish gives -1.95 at depth 8 but
    -7.16 at depth 20 (...Qh5+, ...Qg6+, ...Rb6!); after 31. Kh1 it is -7.63 at every depth. Told
    only the deep eval, the coach called the position lost; told only the scan, balanced. Both were
    wrong for a student: the lesson is the practical difference."""
    if deep_best_white_cp is None or not r.best_move_san:
        return None
    sign = 1 if r.color == "white" else -1
    try:
        after_best = board_before.copy(stack=False)
        after_best.push_san(r.best_move_san)
        best_practical = engines.practical_eval(after_best, depth=PRACTICAL_DEPTH).white().score(mate_score=MATE_CP)
        played_practical = engines.practical_eval(board_after, depth=PRACTICAL_DEPTH).white().score(mate_score=MATE_CP)
    except Exception:
        return None
    deep_m, best_m, played_m = sign * deep_best_white_cp, sign * best_practical, sign * played_practical
    if deep_m > -300 or best_m - deep_m < 300 or best_m - played_m < 200:
        return None
    line = _numbered_line(r.fen_before, r.best_line[:8]) if r.best_line else r.best_move_san
    return (f"Practical framing: even after {r.best_move_san} the engine finds a long win for the opponent "
            f"({line}; {deep_best_white_cp / 100.0:+.2f}), but only well beyond practical depth: at depth "
            f"{PRACTICAL_DEPTH} (about what a human calculates) {r.best_move_san} looks defensible "
            f"({best_practical / 100.0:+.2f}) while {r.san} loses at once ({played_practical / 100.0:+.2f}). "
            f"{r.best_move_san} also loses with best play: it does not hold, save or draw the game. Teach the "
            f"practical difference: {r.san} lost to a short forcing line, {r.best_move_san} would have kept the "
            f"opponent's task hard. For the worse side a perpetual check or repetition is a drawing chance to "
            f"welcome, not something to avoid. Do not call the position balanced, and do not call "
            f"{r.best_move_san} simply losing.")




def moment_digest_lines(i: int, r: MoveRecord, continuation: str = None) -> list:
    """The digest block for one agenda moment — shared by single-game reviews
    and batch training reports so the coaching protocol sees identical data."""
    lines = []
    wb = cp_to_win_pct(r.eval_before_cp if r.color == "white" else -r.eval_before_cp)
    wa = cp_to_win_pct(r.eval_after_cp if r.color == "white" else -r.eval_after_cp)
    lines.append(f"[Moment {i}] {r.move_label} by {r.color} — {r.quality} "
                 f"(win% {wb:.0f} -> {wa:.0f}; eval {r.eval_before_cp / 100.0:+.2f} -> {r.eval_after_cp / 100.0:+.2f})."
                 + _deep_eval_note(r))
    orient = f"  Orientation: this is move {r.number} of the game"
    if r.prev_move_san:
        # The opponent's move is one ply earlier: a Black reply precedes a White
        # move of the same number; a White move precedes a Black move +1.
        if r.color == "white":
            orient += f"; the opponent has just played {r.number - 1}... {r.prev_move_san}"
        else:
            orient += f"; the opponent has just played {r.number}. {r.prev_move_san}"
    lines.append(orient + " (state this when setting the scene).")
    if continuation:
        lines.append(f"  What happened next in the actual game (the student played {r.move_label}): "
                     f"{continuation}")
    lines.append(f"  FEN before the move: {r.fen_before}")
    if r.board_token:
        lines.append(f"  Board diagram token for this position: {r.board_token} "
                     "(paste this EXACT token to show the diagram)")
    if r.position_facts:
        lines.append(f"  Verified position facts (cite ONLY these for positional claims): {r.position_facts}")
    # The rating is in the sentence itself: "at your level" was said of game-time numbers (19enjp8m).
    cohort = f"players rated ~{r.maia_elo}" if r.maia_elo else "similar-rated players"
    if r.maia_played_prob is not None:
        lines.append(f"  Maia: ~{r.maia_played_prob:.0%} of {cohort} choose {r.san}. "
                     f"Top human moves: " + ", ".join(f"{m['move']} ({m['probability']:.0%})" for m in r.maia_top))
    if r.book_share:
        lines.append(f"  Theory note: {r.san} is established master theory (~{r.book_share:.0%} of "
                     "master games) — do NOT scold it; if the engine dislikes it slightly, explain "
                     "the idea behind the theory instead.")
    if r.is_opening_deviation:
        lines.append("  OPENING DEVIATION moment: this is where the student left established "
                     "theory. Teach the theory move and its idea, and recommend this line as a "
                     "study item — the student will reach this position again.")
    if r.tablebase_note:
        lines.append(f"  {r.tablebase_note}")
    if r.practical_note:
        lines.append(f"  Nature of the mistake: {r.practical_note}")
    if r.error_typicality == "typical_for_level":
        lines.append("  Typicality: this mistake is COMMON at this level — normalize it first, "
                     "then teach the underlying pattern thoroughly (it is a real knowledge gap).")
    elif r.error_typicality == "atypical_for_level":
        lines.append("  Typicality: players at this level RARELY play this — treat it as a "
                     "process slip (time, attention, tilt), NOT a knowledge gap. Check the clock "
                     "data and coach the habit, not the pattern.")
    if r.best_move_human_prob is not None:
        lines.append(f"  Findability: ~{r.best_move_human_prob:.0%} of {cohort} find "
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
    if r.human_draw_prob is not None and r.human_win_prob is not None:
        w, d, l = r.human_win_prob, r.human_draw_prob, (r.human_loss_prob or 0.0)
        profile, guidance = _wdl_risk_profile(w, d, l)
        if profile != "balanced":
            lines.append(f"  Risk profile (Maia WDL, mover POV {w * 100:.0f}/{d * 100:.0f}/{l * 100:.0f}): "
                         f"{profile} — {guidance}.")
    if r.time_spent is not None:
        flag = f" (flag: {r.time_flag})" if r.time_flag else ""
        lines.append(f"  Time: spent {int(r.time_spent)}s, {int(r.clock_remaining)}s remaining{flag}.")
    if r.lesson_objective:
        lines.append(f"  Lesson objective: {r.lesson_objective}.")
    if r.recommended_diagnosis:
        lines.append(f"  Recommended diagnosis: {r.recommended_diagnosis} "
                     "(use this unless the student's explanation proves a different root cause).")
    if r.initial_question:
        lines.append(f"  Initial question (ask this EXACTLY first; do not start from hints): {r.initial_question}")
    if r.question_ladder:
        ladder = " | ".join(f"L{level}: {text}" for level, text in r.question_ladder.items())
        lines.append("  Hint ladder (use only after the student attempts or asks for help): "
                     + ladder)
    secret = []
    if r.best_move_san:
        secret.append(f"better was {_numbered_line(r.fen_before, [r.best_move_san])}" +
                      (f" (line: {_numbered_line(r.fen_before, r.best_line)})" if r.best_line else ""))
    if r.best_line_motifs:
        secret.append("what the student missed: " + "; ".join(f"{m['motif']}: {m['detail']}" for m in r.best_line_motifs))
    if r.punishment_line:
        secret.append(f"the punishment of {r.move_label} is: {_numbered_line(r.fen_after, r.punishment_line)}")
    if r.motifs:
        secret.append("motifs in the punishment: " + "; ".join(f"{m['motif']}: {m['detail']}" for m in r.motifs))
    if r.opening_theory_brief:
        secret.append(r.opening_theory_brief)
    if r.saving_resource:
        secret.append(r.saving_resource)
    if r.likely_wrong_answer:
        secret.append(f"most likely student guess: {r.likely_wrong_answer}")
    if r.aspirational:
        secret.append(f"aspirational instinct: at ~{r.aspirational['elo']} Elo most players choose "
                      f"{r.aspirational['move']} ({r.aspirational['probability']:.0%}) — the instinct "
                      "the student is growing toward; share it AFTER the moment is resolved")
    if r.explanation_primitives:
        secret.append("reveal-phase explanation primitives: " + "; ".join(
            f"{k}={v}" for k, v in r.explanation_primitives.items() if v and k != "reasons"))
    if secret:
        lines.append("  WITHHELD ANSWER (do NOT reveal until the student attempts; "
                     "verify attempts with evaluate_student_answer): " + " | ".join(secret))
    if r.suggested_categories:
        lines.append(f"  Suggested diagnosis category: {' or '.join(r.suggested_categories)} "
                     "(use it unless the conversation reveals a different root cause).")
    return lines


def _terminal_eval_cp(board: chess.Board):
    if board.is_checkmate():
        return -MATE_CP if board.turn == chess.WHITE else MATE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    return None


def _draw_rule(board: chess.Board) -> str | None:
    """The draw rule the position on `board` satisfies, from its move stack. A claimable draw
    (threefold, fifty moves) is worth 0 to both sides: whichever side prefers the draw takes it."""
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    if board.is_seventyfive_moves():
        return "seventy_five_move_rule"
    if board.is_repetition(3):
        return "threefold_repetition"
    if board.is_fifty_moves():
        return "fifty_move_rule"
    return None


def _header_elo(headers, key, fallback):
    try:
        return int(headers.get(key, ""))
    except (TypeError, ValueError):
        return fallback


def _parse_increment(headers) -> int:
    """Increment seconds from the PGN TimeControl ("base+inc", e.g. "300+3"), else 0."""
    tc = (headers.get("TimeControl") or "").strip()
    if "+" in tc:
        try:
            return int(tc.split("+", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0


def resolve_student_elo(side: str, headers: dict, config, purpose: str):
    """The Elo for one purpose + its source (mirrors the eval_source honesty
    pattern, GT-12 / plan P0-4a). Returns (elo, source) with source ∈ {config, pgn}.

    - purpose="conversation": the student's anchor level — ALWAYS config.user_elo
      (their declared identity; never prompted, never overridden by a PGN). Grounds
      "your level", coaching, drills, hints, the aspirational base.
    - purpose="typicality": how common a *played* move was — the PGN game-time Elo
      of `side` ("white"/"black"), because "common at your level" means the level of
      whoever made the move when they made it. Falls back to the conversation anchor
      when the PGN carries no rating. Label the prose when source == "pgn" and it
      diverges from the anchor ("at the ~1140 level you were playing then…").
    """
    anchor = int(getattr(config, "user_elo", DEFAULT_USER_ELO) or DEFAULT_USER_ELO)
    if purpose == "conversation":
        return anchor, "config"
    if purpose == "typicality":
        pgn_elo = _header_elo(headers or {}, "WhiteElo" if side == "white" else "BlackElo", None)
        return (pgn_elo, "pgn") if pgn_elo is not None else (anchor, "config")
    raise ValueError(f"unknown elo purpose: {purpose!r}")


def detect_student_color(headers, config):
    names = configured_player_identities(config)
    if not names:
        return None
    # Exact (normalized) equality, not substring: substring made "ann" match
    # "JoanneSmith". Lichess/chess.com headers are bare usernames, so case-folded
    # exact match is correct; ties broken White-first by check order.
    if _normalize_player_identity(headers.get("White")) in names:
        return "white"
    if _normalize_player_identity(headers.get("Black")) in names:
        return "black"
    return None


def _normalize_player_identity(name: str) -> str:
    return (name or "").strip().casefold()


def configured_player_identities(config) -> set[str]:
    """All configured handles that may identify the student in PGN headers.

    P0-4b generalizes the legacy single `player_name` to:
    {"usernames": {"lichess": ["mcmd"], "chesscom": ["mcmd_cc"]}}.
    Matching remains exact and case-insensitive; `player_name` is retained as a
    back-compat alias and joins the same set.
    """
    identities = set()
    legacy = _normalize_player_identity(getattr(config, "player_name", None))
    if legacy:
        identities.add(legacy)

    usernames = getattr(config, "usernames", None) or {}
    if isinstance(usernames, dict):
        iterable = usernames.values()
    else:
        iterable = [usernames]
    for handles in iterable:
        if isinstance(handles, str):
            handles = [handles]
        try:
            for handle in handles or []:
                normalized = _normalize_player_identity(handle)
                if normalized:
                    identities.add(normalized)
        except TypeError:
            continue
    return identities


def has_configured_player_identity(config) -> bool:
    return bool(configured_player_identities(config))


def adopt_review_pov(review: GameReview, student_color: str, engines, config,
                     agenda_size: int = 4) -> GameReview:
    """Rebuild the agenda for an explicitly chosen POV on an unmatched game.

    The scan records are side-agnostic; the agenda and themes are not. When a
    borrowed/friend/master game does not match a configured username, P0-4c asks
    the student which side to study and then reruns the lightweight selection and
    agenda enrichment for that side instead of coaching both players.
    """
    color = (student_color or "").strip().lower()
    if color not in ("white", "black"):
        raise ValueError("student_color must be 'white' or 'black'")

    review.student_color = color
    review.tablebase_entry = None
    review.themes = []
    for record in review.records:
        record.is_agenda = False
        record.board_token = None

    online = bool(getattr(config, "use_online_databases", True))
    forced_plies = []
    if online:
        try:
            forced_plies += _tablebase_category_scan(review)
        except Exception:
            pass
    forced_plies += _opening_deviation_ply(review)
    review.agenda_plies = _select_agenda(review.records, color, agenda_size, forced_plies)
    _enrich_agenda(review, engines, online)
    _apply_opening_moment(review)
    review.themes = _derive_themes(review)
    return review


def review_game(pgn_text: str, engines, config, on_progress=None, agenda_size: int = 4,
                eval_time: float = None, with_maia: bool = True, enrich: bool = True):
    """Full review of the first game in `pgn_text`. Returns a GameReview, or
    None if no game with moves could be parsed.

    Batch fast-scan knobs (defaults reproduce single-game behavior exactly):
    `eval_time` overrides config.review_eval_time; `with_maia=False` skips Maia
    entirely (the CPU-bound part); `enrich=False` skips agenda selection, deep
    enrichment, and online lookups, returning scan-level records only."""
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
    prev_move_san = None  # SAN of the move leading into the current position (orientation)
    pgn_eval_moves = 0    # moves whose eval came from the PGN's [%eval] (not local engine)
    white_elo = _header_elo(headers, "WhiteElo", getattr(config, "user_elo", DEFAULT_USER_ELO))
    black_elo = _header_elo(headers, "BlackElo", getattr(config, "user_elo", DEFAULT_USER_ELO))
    increment = _parse_increment(headers)  # clock added back per move (for think-time)

    scan_eval_time = float(eval_time) if eval_time is not None else float(
        getattr(config, "review_eval_time", 0.5))
    tracing.trace(f"review: scanning {total} plies (Stockfish quick eval"
                  + (" + Maia prediction" if with_maia else "")
                  + " per move; engine-level traces suppressed during the scan)")
    for ply, node in enumerate(nodes):
        move = node.move
        fen_before = board.fen()
        # Game line reaching this position — fed to maia3 for history-aware
        # prediction. Only built when we actually run Maia (the batch scan skips
        # it to keep its JSON cache lean).
        history_before = [m.uci() for m in board.move_stack] if with_maia else None

        eval_time_ply = scan_eval_time
        if prev_eval_cp is None:
            eval_before_cp, best_uci = _eval_with_best(engines, board, eval_time_ply)
            parent_eval = node.parent.eval() if node.parent else None
            if parent_eval is not None:   # prefer the PGN's own eval here too
                eval_before_cp = parent_eval.white().score(mate_score=MATE_CP)
        else:
            eval_before_cp, best_uci = prev_eval_cp, prev_best_uci

        mover_white = board.turn == chess.WHITE
        san = board.san(move)
        uci = move.uci()
        is_capture = board.is_capture(move)
        is_forced = board.legal_moves.count() == 1
        is_best = best_uci is not None and uci == best_uci
        in_check = board.is_check()
        phase = get_game_phase(board)

        # Clock data: a player's previous clock reading is two plies back.
        time_spent = clock_remaining = None
        clk = node.clock()
        if clk is not None:
            clock_remaining = clk
            if ply >= 2 and nodes[ply - 2].clock() is not None:
                # think time = clock two plies ago + increment gained this move - now.
                # Omitting the increment underestimates (often to 0) in increment games,
                # producing false hope_chess / time_scramble flags.
                time_spent = max(0.0, nodes[ply - 2].clock() + increment - clk)

        board.push(move)
        fen_after = board.fen()
        eval_after_cp, next_best = _eval_with_best(engines, board, eval_time_ply)
        # The win% stream that drives drop/quality and slide detection is far more
        # reliable from the PGN's own deep [%eval] annotations than from a sub-second
        # local search — and the slide baseline (~17%) was calibrated on exactly those
        # Lichess evals (scripts/scan_eval_slides.py). Prefer them when present; the
        # engine eval is the fallback for un-annotated PGNs (flagged via eval_source).
        # Stockfish is still consulted for the BEST move / punisher (evals lack those).
        node_eval = node.eval()
        if node_eval is not None:
            eval_after_cp = node_eval.white().score(mate_score=MATE_CP)
            pgn_eval_moves += 1
        draw_rule = _draw_rule(board)
        eval_without_draw_rule_cp = None
        if draw_rule:  # an exact rule fact outranks any evaluation, imported or local
            eval_without_draw_rule_cp, eval_after_cp = eval_after_cp, 0

        before_m = eval_before_cp if mover_white else -eval_before_cp
        after_m = eval_after_cp if mover_white else -eval_after_cp
        # Round ONCE, here: classification, agenda selection, and display must
        # all see the same number (9.96 classified as "ok" but displayed and
        # agenda-selected as 10.0 produced "critical moment (ok)" absurdities).
        drop = round(max(0.0, cp_to_win_pct(before_m) - cp_to_win_pct(after_m)), 1)

        # Maia on EVERY move: the human-vs-engine comparison is high-quality
        # signal both for normalizing errors and for spotting praise moments.
        maia_top, maia_played_prob, was_maia_top, avoided = [], None, False, None
        mover_elo = white_elo if mover_white else black_elo
        oppo_elo = black_elo if mover_white else white_elo
        maia = None
        if with_maia:
            with tracing.suspended():
                maia = engines.maia_predict(fen_before, elo_self=mover_elo, elo_oppo=oppo_elo,
                                            history_uci=history_before, tier="scan")
        best_move_human_prob = None
        human_win_prob = None
        human_draw_prob = human_loss_prob = None
        if maia:
            human_win_prob = maia.get("win_probability_for_side_to_move")
            human_draw_prob = maia.get("draw_probability")
            human_loss_prob = maia.get("loss_probability")
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
                    engines, fen_before, mover_white, before_m, natural_uci, natural_prob, san, eval_time_ply,
                    elo=mover_elo)

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
                              f"players rated ~{mover_elo} find it")
            if quality in ("blunder", "mistake"):
                if played_prob >= 0.30:
                    typicality = "typical_for_level"
                elif played_prob <= 0.08:
                    typicality = "atypical_for_level"
        records.append(MoveRecord(
            ply=ply, number=(ply // 2) + 1, color="white" if mover_white else "black",
            san=san, uci=uci, fen_before=fen_before, fen_after=fen_after,
            eval_before_cp=eval_before_cp, eval_after_cp=eval_after_cp,
            drop_pct=drop, quality=quality,
            best_move_san=_uci_to_san(fen_before, best_uci),
            is_capture=is_capture, time_spent=time_spent, clock_remaining=clock_remaining,
            time_flag=time_flag_for(quality, time_spent, clock_remaining, is_forced),
            punisher_uci=next_best, history_uci=history_before or [],
            prev_move_san=prev_move_san, in_check=in_check,
            maia_top=maia_top, maia_played_prob=maia_played_prob, maia_elo=mover_elo if maia else None,
            was_maia_top=was_maia_top, avoided_natural_mistake=avoided,
            brilliancy_note=brilliancy, error_typicality=typicality,
            best_move_human_prob=best_move_human_prob,
            phase=phase, human_win_prob=human_win_prob,
            human_draw_prob=human_draw_prob, human_loss_prob=human_loss_prob, draw_rule=draw_rule,
            eval_without_draw_rule_cp=eval_without_draw_rule_cp,
        ))
        prev_eval_cp, prev_best_uci = eval_after_cp, next_best
        prev_move_san = san  # next iteration's "opponent's last move"
        if on_progress:
            on_progress(ply + 1, total)

    # "pgn" when the game shipped its own [%eval] (Lichess analysed) on essentially
    # every move — the eval source the slide detector was calibrated on; "engine"
    # when we fell back to the local sub-second search (less reliable; flagged in
    # the slide output so the coach never overstates a fallback read).
    eval_source = "pgn" if total and pgn_eval_moves >= 0.8 * total else "engine"
    # conversation_elo = the student's CURRENT level (config), the anchor for
    # coaching/aspiration; distinct from the PGN game-time ratings above (P0-4a).
    conversation_elo = resolve_student_elo(student_color or "white", headers, config, "conversation")[0]
    review = GameReview(headers=headers, records=records, student_color=student_color,
                        agenda_plies=[], game=game, extra_games_ignored=extra,
                        white_elo=white_elo, black_elo=black_elo, eval_source=eval_source,
                        conversation_elo=conversation_elo)
    _detect_slides(review)   # eval-only; cheap and safe in scan and full review
    if not enrich:
        return review
    online = bool(getattr(config, "use_online_databases", True))
    forced_plies = []
    if online:
        try:
            _analyze_opening(review, config)
        except Exception:
            pass  # network enrichment must never break a review
        try:
            forced_plies += _tablebase_category_scan(review)
        except Exception:
            pass
    forced_plies += _opening_deviation_ply(review)
    _attach_agenda_human_danger(
        records, student_color, agenda_size, engines,
        white_elo=white_elo, black_elo=black_elo,
        eval_time=min(float(eval_time_ply), 0.3) if eval_time_ply else 0.2,
        enabled=with_maia,
    )
    review.agenda_plies = _select_agenda(records, student_color, agenda_size, forced_plies)
    _enrich_agenda(review, engines, online)
    _apply_opening_moment(review)
    review.themes = _derive_themes(review)
    return review


class EngineUnavailable(RuntimeError):
    """Stockfish evaluation failed and could not be recovered. Raised instead of
    fabricating a 0.00 ('equal') eval — a silent zero invents chess truth, creating
    phantom win%-swings, false blunders/slides, and bad agenda selection."""


def _eval_with_best(engines, board: chess.Board, eval_time: float = 0.5):
    terminal = _terminal_eval_cp(board)
    if terminal is not None:
        return terminal, None
    last_err = None
    for _ in range(2):  # one retry — a transient hiccup recovers; a dead engine won't
        try:
            with tracing.suspended():  # bulk scan: one trace line would become hundreds
                score, best_uci = engines.quick_eval(board.fen(), eval_time, history_key(board))
            return score.white().score(mate_score=MATE_CP), best_uci
        except Exception as e:
            last_err = e
    raise EngineUnavailable(f"Stockfish evaluation failed for {board.fen()}: {last_err}") from last_err


def _check_avoided_natural_mistake(engines, fen_before, mover_white, before_m,
                                   natural_uci, natural_prob, played_san, eval_time=0.5, elo=None):
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
            cohort = f"players rated ~{elo}" if elo else "similar-rated players"
            return (f"{natural_prob:.0%} of {cohort} choose {natural_san} here, "
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


def _agenda_candidate_records(records, student_color):
    return [r for r in records
            if r.drop_pct >= 10 and (student_color is None or r.color == student_color)]


def _typicality_elo_for_record(record: MoveRecord, white_elo: int, black_elo: int):
    return white_elo if record.color == "white" else black_elo


def _attach_agenda_human_danger(records, student_color, agenda_size: int, engines, *,
                                white_elo: int, black_elo: int, eval_time: float,
                                enabled: bool = True) -> None:
    """Compute the cheap P0-1b danger signal only where it can affect selection.

    Candidate eligibility remains the calibrated eval-drop rule. This pass only
    fills a bounded, explicitly provisional tie/priority signal for the strongest
    candidate mistakes, not a whole-game root-move map.
    """
    for record in records:
        record.human_danger = empty_human_danger(record, "not_computed")

    if not enabled or not getattr(engines, "stockfish", None):
        reason = "maia_disabled" if not enabled else "engine_unavailable"
        for record in _agenda_candidate_records(records, student_color):
            record.human_danger = empty_human_danger(record, reason)
        return

    pool = sorted(_agenda_candidate_records(records, student_color),
                  key=lambda r: (r.drop_pct, -r.ply), reverse=True)
    limit = max(agenda_size * 3, agenda_size + 4)
    for record in pool[:limit]:
        if not record.maia_top:
            record.human_danger = empty_human_danger(record, "no_maia_top_moves")
            continue
        try:
            record.human_danger = human_danger_mass_for_record(
                record, engines,
                elo=_typicality_elo_for_record(record, white_elo, black_elo),
                eval_time=eval_time,
            )
        except Exception as exc:
            record.human_danger = empty_human_danger(record, f"probe_failed:{type(exc).__name__}")


def _agenda_score_components(record: MoveRecord) -> dict:
    """Internal criticality score for P0-1b agenda ranking.

    The base remains the calibrated win%-drop. Maia/local-eval factors are capped
    boosts for teachability and human-likelihood only; they never rewrite
    `quality`, `drop_pct`, slide detection, or human corpus labels.
    """
    danger = record.human_danger or empty_human_danger(record, "not_computed")
    danger_boost = 0.0
    if danger.get("available"):
        bad_mass = float(danger.get("bad_or_worse_mass") or 0.0)
        mb_mass = float(danger.get("mistake_or_blunder_mass") or 0.0)
        danger_boost = min(8.0, (bad_mass * 5.0) + (mb_mass * 7.0))

    played_prob = record.maia_played_prob if record.maia_played_prob is not None else 0.0
    if played_prob >= 0.30:
        typicality_boost = 4.0
    elif played_prob >= 0.20:
        typicality_boost = 2.0
    else:
        typicality_boost = 0.0

    best_prob = record.best_move_human_prob
    if best_prob is None:
        findability_boost = 0.0
    elif best_prob <= 0.10:
        findability_boost = 3.0
    elif best_prob <= 0.20:
        findability_boost = 1.5
    else:
        findability_boost = 0.0

    human_boost = min(10.0, danger_boost + typicality_boost + findability_boost)
    total = round(record.drop_pct + human_boost, 2)
    return {
        "schema_version": 1,
        "base_drop_pct": record.drop_pct,
        "human_boost_cap": 10.0,
        "human_danger_boost": round(danger_boost, 2),
        "maia_typicality_boost": round(typicality_boost, 2),
        "findability_boost": round(findability_boost, 2),
        "human_boost": round(human_boost, 2),
        "total": total,
        "human_danger_available": bool(danger.get("available")),
        "human_danger_provisional": bool(danger.get("provisional", True)),
        "human_danger_source": danger.get("source"),
        "note": "Internal ranking only; severity remains drop_pct/quality from the calibrated eval stream.",
    }


def _select_agenda(records, student_color, agenda_size, forced_plies=()):
    """Top critical moments plus moments forced by evidence the eval swing can miss.

    Candidate eligibility is still eval-drop based. P0-1b adds a capped
    human-factor priority boost, so a slightly smaller but highly teachable/common
    mistake can outrank a nearby raw-drop mistake without changing severity.
    """

    # Only widen to BOTH colours when we don't know the student's side. With a
    # known side and no student mistakes, the agenda must stay empty (→ "praise the
    # accuracy") — never fall back to coaching the OPPONENT's blunders as if they
    # were the student's. (Audit: clean student game taught a Black blunder.)
    pool = _agenda_candidate_records(records, student_color) if student_color else _agenda_candidate_records(records, None)
    for record in pool:
        record.agenda_score = _agenda_score_components(record)
    pool.sort(key=lambda r: (r.agenda_score["total"], r.drop_pct, -r.ply), reverse=True)
    chosen = {r.ply for r in pool[:agenda_size]}
    chosen |= {p for p in forced_plies if 0 <= p < len(records)}
    return sorted(chosen)


def _theme_key(r: MoveRecord) -> str:
    """The verified category a single-game agenda moment belongs to. Prefers the
    deep, taxonomy-grounded `recommended_diagnosis` (what actually 'churned' in the
    session: premature_attack / bad_trade / ignored_threat …), so distinct error
    classes don't get smoothed into one narrative."""
    return r.recommended_diagnosis or r.lesson_objective or "uncategorized"


def _derive_themes(review: GameReview) -> list:
    """Cluster the enriched agenda by verified category + phase (P0-1a), mirroring
    batch.aggregate_patterns down onto a single game. The synthesis reads these and
    must not assert a theme across moments the backend placed in different clusters
    — that is exactly the over-fit that put a forced recapture in 'don't grab
    material' in game 2dtDqQgRy2OD."""
    groups: dict = {}
    for ply in review.agenda_plies:
        if not (0 <= ply < len(review.records)):
            continue
        r = review.records[ply]
        if review.student_color and r.color != review.student_color:
            continue
        key = (_theme_key(r), r.phase or "?")
        groups.setdefault(key, []).append(ply)

    themes = []
    for (cat, phase), plies in groups.items():
        plies = sorted(plies)
        moves = ", ".join(review.records[p].move_label for p in plies)
        label = cat.replace("_", " ")
        count = len(plies)
        themes.append(ReviewTheme(
            id=f"{cat}:{phase}",
            label=f"{label} ({phase})" if phase != "?" else label,
            plies=plies,
            # A repeated category within one game is a real in-game pattern; a
            # singleton is a one-off (teach it, but don't generalise to a "habit").
            confidence="high" if count >= 2 else "low",
            safe_summary=(f"{count}x {label} in the {phase}: {moves}." if count >= 2
                          else f"a single {label} moment in the {phase}: {moves}."),
            avoid_language=[f"do not group these plies with moments outside cluster '{cat}:{phase}'"],
        ))
    # Largest (most repeated) clusters first — the most teachable patterns lead.
    themes.sort(key=lambda t: (-len(t.plies), t.id))
    return themes


def _tablebase_category_scan(review: GameReview) -> list:
    """In tablebase range, game-theoretic truth can flip (WIN -> DRAW) while the
    engine's eval swing stays small and never reaches the agenda. Probe every
    <=7-piece ply (cached): force the student's category-change moments onto the
    agenda and mark where the game entered tablebase territory."""
    from app import external

    forced = []
    for r in review.records:
        board_before = chess.Board(r.fen_before)
        if len(board_before.piece_map()) > 7:
            continue
        tb_before = external.tablebase_probe(r.fen_before)
        if not tb_before:
            continue
        before_cat = external.normalize_category(tb_before.get("category"))  # mover POV
        if review.tablebase_entry is None and before_cat:
            review.tablebase_entry = {"move_label": r.move_label,
                                      "side_to_move": r.color, "category": before_cat}
        after_cat = None
        board_after = chess.Board(r.fen_after)
        if len(board_after.piece_map()) <= 7:
            tb_after = external.tablebase_probe(r.fen_after)
            if tb_after:
                after_cat = external.invert_category(
                    external.normalize_category(tb_after.get("category")))  # back to mover POV
        if (before_cat and after_cat and before_cat != after_cat
                and (review.student_color is None or r.color == review.student_color)):
            tracing.trace(f"review: tablebase category flip at {r.move_label} "
                          f"({before_cat} -> {after_cat}) — forced onto agenda")
            forced.append(r.ply)
    return forced[:3]


def _opening_deviation_ply(review: GameReview) -> list:
    """A costly opening deviation by the student becomes a structured agenda
    moment (it is the most repeatable lesson in the game)."""
    d = review.opening_deviation
    if not d:
        return []
    if review.student_color and d["color"] != review.student_color:
        return []
    record = review.records[d["ply"]]
    if record.drop_pct < 5:
        return []  # harmless novelty: report it in the header, don't teach it
    return [d["ply"]]


def _apply_opening_moment(review: GameReview) -> None:
    """Re-frame the deviation record as an opening-theory lesson (runs after
    enrichment so it can override the generic primitives)."""
    d = review.opening_deviation
    if not d or d["ply"] not in review.agenda_plies:
        return
    r = review.records[d["ply"]]
    r.is_opening_deviation = True
    r.lesson_objective = "opening_theory_knowledge"
    r.recommended_diagnosis = "opening_theory"
    if "opening_theory" not in r.suggested_categories:
        r.suggested_categories = ["opening_theory"] + r.suggested_categories
    r.question_ladder[1] = ("This position is still established theory. What do strong players "
                            "play here, and what does the game move give up?")
    brief = f"theory here: {d['theory']}"
    if d.get("amateur"):
        brief += f"; at the student's rating: {d['amateur']}"
    r.opening_theory_brief = brief


OPENING_SCAN_PLIES = 24
OPENING_BOOK_MIN_GAMES = 50


def _analyze_opening(review: GameReview, config) -> None:
    """Walk the opening through the Lichess masters explorer: name the opening,
    tag book moves (so theory is never scolded), and find the exact move where
    the game left established theory."""
    from app import external

    name = eco = None
    deviation = None
    for r in review.records[:OPENING_SCAN_PLIES]:
        data = external.masters_moves(r.fen_before)
        if data is None:
            return  # offline / rate-limited: skip the opening report entirely
        opening = data.get("opening")
        if opening:
            name, eco = opening.get("name"), opening.get("eco")
        moves = data.get("moves") or []
        totals = {m["san"]: m["white"] + m["draws"] + m["black"] for m in moves}
        position_total = sum(totals.values())
        if position_total < OPENING_BOOK_MIN_GAMES:
            break  # book ran out without a clear deviation

        played_games = totals.get(r.san, 0)
        share = (played_games / position_total) if position_total else 0.0
        if played_games >= 5 and share >= 0.005:
            r.book_share = round(share, 3)
            continue

        mover_white = r.color == "white"

        def scored(m, total):
            games = m["white"] + m["draws"] + m["black"]
            wins = m["white"] if mover_white else m["black"]
            score = (wins + m["draws"] / 2) / games if games else 0
            return f"{m['san']} ({games / total:.0%} of games, scores {score:.0%})"

        theory = ", ".join(scored(m, position_total) for m in moves[:3])
        deviation = {"ply": r.ply, "label": r.move_label, "color": r.color,
                     "played_share": share, "played_games": played_games, "theory": theory}
        brackets = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500]
        user_elo = getattr(config, "user_elo", DEFAULT_USER_ELO)
        valid = [str(b) for b in brackets if abs(b - user_elo) <= 250]
        amateur = external.lichess_moves(r.fen_before, ",".join(valid) if valid else "1000,1200")
        if amateur and amateur.get("moves"):
            am_total = sum(m["white"] + m["draws"] + m["black"] for m in amateur["moves"]) or 1
            deviation["amateur"] = ", ".join(scored(m, am_total) for m in amateur["moves"][:3])
        tracing.trace(f"review: opening deviation found at {r.move_label}")
        break

    review.opening_name, review.opening_eco, review.opening_deviation = name, eco, deviation


def _tablebase_verify(r: MoveRecord) -> None:
    """In <=7-piece positions, replace 'the engine thinks' with provable truth:
    did this move change the game-theoretic result, and how many moves didn't?"""
    from app import external

    board_before = chess.Board(r.fen_before)
    if len(board_before.piece_map()) > 7:
        return
    tb_before = external.tablebase_probe(r.fen_before)
    if not tb_before:
        return
    before_cat = external.normalize_category(tb_before.get("category"))  # mover POV
    moves = tb_before.get("moves") or []
    preserving = sum(1 for m in moves
                     if external.invert_category(external.normalize_category(m.get("category"))) == before_cat)

    after_cat = None
    board_after = chess.Board(r.fen_after)
    if len(board_after.piece_map()) <= 7:
        tb_after = external.tablebase_probe(r.fen_after)
        if tb_after:
            after_cat = external.invert_category(
                external.normalize_category(tb_after.get("category")))  # back to mover POV

    if before_cat and after_cat and before_cat != after_cat:
        r.tablebase_note = (f"TABLEBASE TRUTH (provable, not engine opinion): before {r.san} the position "
                            f"was a proven {before_cat.upper()} for the player; after {r.san} it is a proven "
                            f"{after_cat.upper()}. {preserving} of {len(moves)} legal moves preserved the {before_cat}.")
    elif before_cat:
        r.tablebase_note = (f"Tablebase: the position before {r.san} is a proven {before_cat}; "
                            f"{preserving} of {len(moves)} legal moves preserve that result.")


def _engine_setting(engines, name: str, default):
    return getattr(getattr(engines, "config", None), name, default)


def _record_board(r: MoveRecord) -> chess.Board:
    """The position before `r` with its game history when the record carries it (batch reviews mix
    games, so their records are replayed one by one); a bare FEN when it does not."""
    board = chess.Board()
    try:
        for uci in r.history_uci or []:
            board.push_uci(uci)
    except ValueError:
        return chess.Board(r.fen_before)
    return board if r.history_uci and board.fen() == r.fen_before else chess.Board(r.fen_before)


def _enrich_agenda(review: GameReview, engines, online: bool = True):
    for ply in review.agenda_plies:
        r = review.records[ply]
        r.is_agenda = True
        tracing.trace(f"review: deep-analyzing agenda moment {r.move_label} "
                      f"({r.quality}, -{r.drop_pct} win%)")
        # With the game's moves on the stack, so the searches see repetitions.
        board_before = review.board_before(ply) if isinstance(review, GameReview) else _record_board(r)
        board_after = board_before.copy()
        board_after.push_uci(r.uci)
        if online:
            try:
                _tablebase_verify(r)
            except Exception:
                pass

        deep_best_white_cp = None
        try:
            infos = engines.top_lines(
                board_before, multipv=3,
                depth=int(_engine_setting(engines, "agenda_best_depth", 20)),
                time_cap=float(_engine_setting(engines, "agenda_best_time", 6.0)))
            r.best_line = _select_best_line_for_record(r, board_before, infos)
            if infos and infos[0].get("score") is not None:
                r.deep_eval_before_cp = infos[0]["score"].white().score(mate_score=MATE_CP)
            for info in infos or []:
                pv = info.get("pv") or []
                if pv and r.best_line and info.get("score") is not None \
                        and _same_san(board_before.san(pv[0]), r.best_line[0]):
                    deep_best_white_cp = info["score"].white().score(mate_score=MATE_CP)
                    break
            r.best_line_motifs = detect_motifs_in_line(board_before, r.best_line)
            # The deep multipv PV is more authoritative than the shallow scan eval
            # that first set best_move_san. When they disagree on the FIRST move,
            # the explanation (best_line + its motifs) describes the deep line while
            # the "answer" still names the shallow move — which produced tactically
            # FALSE coaching (e.g. teaching 22.Nb3 as a 'discovered check' when the
            # check only appears after the line's real first move 22.Rd1). Reconcile
            # the answer to the deep PV so the move taught and the move explained are
            # the same. Done before the lesson primitives / diagnosis read best_move.
            if r.best_line:
                r.best_move_san = r.best_line[0]
            elif r.quality in ("inaccuracy", "mistake", "blunder"):
                r.best_move_san = None
        except Exception:
            pass

        deep_after_white_cp = None
        try:
            infos = engines.top_lines(
                board_after, multipv=1,
                depth=int(_engine_setting(engines, "agenda_punishment_depth", 18)),
                time_cap=float(_engine_setting(engines, "agenda_punishment_time", 4.0)))
            pv = infos[0].get("pv") or []
            # A mating line is kept whole (mate in N is 2N-1 plies) so the mate itself is seen:
            # YbjNb8wD 20. Rxe8 allows mate in 5, whose ...Qxf1# is ply 9.
            r.punishment_line = _pv_to_san(board_after, pv[:MATING_LINE_MAX_PLIES]
                                           if infos[0]["score"].is_mate() else pv[:8])
            r.motifs = detect_motifs_in_line(board_after, r.punishment_line)
            deep_after_white_cp = infos[0]["score"].white().score(mate_score=MATE_CP)
        except Exception:
            pass
        r.suggested_categories = suggest_error_categories(r)
        r.position_facts = position_facts(board_before)
        _attach_lesson_primitives(r, board_before, board_after)
        _assess_practicality(r, engines, review, deep_after_white_cp)
        r.engine_only_refutation = _practical_refutation(r, engines, board_before, board_after, deep_best_white_cp)
        # Per-move Maia data was already collected during the main scan.
        _anticipate_wrong_answer(r, engines, board_before)
        _aspirational_instinct(r, engines, review)


def _same_san(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[+#?!]+$", "", str(s or "").strip())
    return bool(a and b and norm(a) == norm(b))


def _select_best_line_for_record(r: MoveRecord, board_before: chess.Board, infos: list) -> list:
    """Return the PV line used as the teachable improvement.

    Calibrated PGN evals can mark the played move as a blunder while a shallow or
    locally unstable Stockfish PV still starts with that played move. For agenda
    records, never teach a bad played move as its own improvement; use the best
    non-played MultiPV line when available, otherwise leave the answer unset.
    """
    candidate_lines = []
    for info in infos or []:
        line = _pv_to_san(board_before, (info.get("pv") or [])[:8])
        if line:
            candidate_lines.append(line)
    if not candidate_lines:
        return []
    if r.quality not in ("inaccuracy", "mistake", "blunder"):
        return candidate_lines[0]
    for line in candidate_lines:
        if not _same_san(line[0], r.san):
            return line
    return []


def _assess_practicality(r: MoveRecord, engines, review: GameReview, deep_after_white_cp) -> None:
    """Was the mistake REFUTED BY FORCE, or did it 'merely' enter sharpness
    where the saving moves are precise and human-unlikely? The coach must lead
    with this honestly and proactively — '17. d5 does not lose by force, but
    only one narrow path holds and few humans find it' — instead of implying a
    forced loss and walking it back when the student pushes.

    Observed live: the student challenged 'why is ...c5 dangerous?' and only
    then learned the played move was actually the engine's top choice to hold.
    """
    if deep_after_white_cp is None:
        return
    mover_white = r.color == "white"
    before_m = r.eval_before_cp if mover_white else -r.eval_before_cp
    deep_after_m = deep_after_white_cp if mover_white else -deep_after_white_cp
    deep_drop = cp_to_win_pct(before_m) - cp_to_win_pct(deep_after_m)

    if deep_drop >= 15:
        r.practical_note = (f"CONCRETE — even with best play after {r.san}, the damage stands "
                            "(verified at depth). The punishment line is real; teach it as such.")
        return
    if deep_drop > 5:
        return  # partly recoverable: no strong claim either way

    # Deep analysis says the move did NOT lose by force. How narrow is the path?
    saving = None
    try:
        if r.punishment_line:
            reply_board = chess.Board(r.fen_after)
            reply_board.push_san(r.punishment_line[0])
            if not reply_board.is_game_over():
                infos = engines.top_lines(
                    reply_board, multipv=2,
                    depth=int(_engine_setting(engines, "practicality_depth", 16)),
                    time_cap=float(_engine_setting(engines, "practicality_time", 3.0)))
                pov = reply_board.turn
                pv = infos[0].get("pv") or []
                if pv:
                    gap = None
                    if len(infos) > 1:
                        gap = (infos[0]["score"].pov(pov).score(mate_score=MATE_CP)
                               - infos[1]["score"].pov(pov).score(mate_score=MATE_CP))
                    prob = None
                    mover_elo = (review.white_elo if mover_white else review.black_elo)
                    reply_hist = (list(r.history_uci) + [r.uci, reply_board.move_stack[-1].uci()]
                                  if r.history_uci else None)
                    maia = engines.maia_predict(reply_board.fen(), elo_self=mover_elo,
                                                elo_oppo=mover_elo, history_uci=reply_hist, tier="deep")
                    if maia:
                        prob = maia["raw_probs"].get(pv[0].uci())
                    saving = {"san": reply_board.san(pv[0]), "gap": gap, "prob": prob}
    except Exception:
        pass

    narrow = bool(saving and (((saving["gap"] or 0) >= 150)
                              or (saving["prob"] is not None and saving["prob"] <= 0.15)))
    r.practical_note = (
        f"PRACTICAL, not forced — deep analysis shows {r.san} does NOT lose by force; the position "
        "stays defensible with precise play. "
        + ("The defense is NARROW: essentially one good path, which players at this level rarely find. "
           if narrow else "The defense is demanding. ")
        + "Say this up front, BEFORE questioning: the real value of the better move is avoiding the "
          "need for that precision — never imply the game move was refuted outright.")
    if saving:
        detail = f"the saving resource after {r.san} {r.punishment_line[0]} is {saving['san']}"
        if saving.get("gap") is not None and saving["gap"] >= 150:
            detail += " (essentially the only good move)"
        if saving.get("prob") is not None:
            detail += f"; only ~{saving['prob']:.0%} of players at this level find it"
        r.saving_resource = detail


def _attach_lesson_primitives(r: MoveRecord, board_before: chess.Board, board_after: chess.Board) -> None:
    r.lesson_objective = _infer_lesson_objective(r)
    r.recommended_diagnosis = _recommended_diagnosis(r)
    risk = (_wdl_risk_profile(r.human_win_prob, r.human_draw_prob, r.human_loss_prob or 0.0)[0]
            if r.human_win_prob is not None and r.human_draw_prob is not None else None)
    r.initial_question = _initial_question(board_before, risk)
    r.question_ladder = _question_ladder(r, board_before)
    r.explanation_primitives = _explanation_primitives(r, board_before, board_after)


def _infer_lesson_objective(r: MoveRecord) -> str:
    cats = set(r.suggested_categories)
    if "king_safety" in cats:           # an allowed mate outranks every material lesson (#9)
        return "king_safety_before_material"
    if _is_simple_capture_punishment(r):
        return "material_vs_activity"
    if "allows_forcing_sequence" in cats:
        return "check_opponent_forcing_replies"
    if _punishment_starts_with_forcing_check(r):
        return "defend_against_direct_threat"
    if "ignored_threat" in cats:
        return "defend_against_direct_threat"
    if "bad_trade" in cats:
        return "material_vs_activity"
    if "premature_attack" in cats:
        return "improve_pieces_before_committing_pawns"
    if r.time_flag in ("hope_chess", "time_scramble"):
        return "threat_scan_under_time_pressure"
    if r.is_capture:
        return "material_temptation_vs_position_priority"
    return "priority_selection"


def _recommended_diagnosis(r: MoveRecord) -> str:
    # #9 precedence: an allowed mate, a simple recapture, a threat the move created, then one
    # that already existed.
    if "king_safety" in r.suggested_categories:
        return "king_safety"
    if _is_simple_capture_punishment(r):
        return "bad_trade"
    if r.lesson_objective == "check_opponent_forcing_replies":
        return "allows_forcing_sequence"
    if _punishment_starts_with_forcing_check(r):
        return "ignored_threat"
    objective_map = {
        "defend_against_direct_threat": "ignored_threat",
        "material_vs_activity": "bad_trade",
        "king_safety_before_material": "king_safety",
        "improve_pieces_before_committing_pawns": "premature_attack",
        "threat_scan_under_time_pressure": "time_management",
        "material_temptation_vs_position_priority": "bad_trade",
    }
    mapped = objective_map.get(r.lesson_objective)
    if mapped:
        return mapped
    if r.suggested_categories:
        return r.suggested_categories[0]
    if r.is_capture:
        return "bad_trade"
    return "calculation_error"


def _punishment_starts_with_forcing_check(r: MoveRecord) -> bool:
    if not r.punishment_line:
        return False
    first = r.punishment_line[0]
    if first.endswith(("+", "#")):
        return True
    try:
        board = chess.Board(r.fen_after)
        move = board.parse_san(first)
        return board.gives_check(move)
    except (ValueError, AssertionError):
        return False


def _is_simple_capture_punishment(r: MoveRecord) -> bool:
    """A capture punished by an immediate recapture is usually not a standing
    "threat" lesson. The teachable point is the student's material temptation
    or trade choice, not that the opponent was threatening the occupied square.
    """
    if not (r.is_capture and r.punishment_line and r.best_move_san):
        return False
    try:
        board_after = chess.Board(r.fen_after)
        try:
            reply = board_after.parse_san(r.punishment_line[0])
        except ValueError:
            reply = next(
                (move for move in board_after.legal_moves
                 if board_after.is_capture(move)
                 and board_after.san(move).rstrip("+#?!") == str(r.punishment_line[0]).rstrip("+#?!")),
                None,
            )
            if reply is None:
                return False
        if board_after.gives_check(reply) or not board_after.is_capture(reply):
            return False
        played = chess.Move.from_uci(r.uci)
        return reply.to_square == played.to_square
    except (ValueError, AssertionError):
        return False


def _initial_question(board_before: chess.Board, risk: str = None) -> str:
    """`risk` is the moment's Maia risk verdict (_wdl_risk_profile); the in-check question names
    the goal only when that verdict supports it."""
    side = "White" if board_before.turn == chess.WHITE else "Black"
    if board_before.is_check():
        goal = {"must-hold": "which gives you the best chance to hold the game",
                "winning": "which keeps your winning chances"}.get(risk, "which is your best option")
        return f"{side} to move. You're in check - what are your options, and {goal}?"
    opponent = "Black" if side == "White" else "White"
    return f"{side} to move. What candidate moves would you consider, and what is {opponent} threatening?"


def _question_ladder(r: MoveRecord, board_before: chess.Board) -> dict:
    ladder = {}
    objective = r.lesson_objective or ""
    if objective == "defend_against_direct_threat":
        ladder[1] = "First identify the forcing threat against your king or material, then choose a move that answers it."
    elif objective == "check_opponent_forcing_replies":
        ladder[1] = "Before choosing, ask: what checks, captures and threats would your move give the opponent?"
    elif objective == "material_vs_activity":
        ladder[1] = "This is not primarily about grabbing material. What move improves your active piece or supports your main asset?"
    elif objective == "king_safety_before_material":
        ladder[1] = "Before recapturing or taking material near your king, what defensive move reduces the opponent's forcing play?"
    elif objective == "improve_pieces_before_committing_pawns":
        ladder[1] = "Before changing the pawn structure, which piece still needs to join the position?"
    else:
        ladder[1] = "What is the position's most urgent priority before considering natural material moves?"

    if r.best_move_san:
        try:
            move = board_before.parse_san(r.best_move_san)
            piece = board_before.piece_at(move.from_square)
            piece_name = chess.piece_name(piece.piece_type) if piece else "piece"
            home_rank = 0 if board_before.turn == chess.WHITE else 7
            if chess.square_rank(move.from_square) == home_rank:
                ladder[2] = "Look at your back-rank pieces."
            else:
                ladder[2] = f"Look at your {piece_name}s."
            article = "An" if piece_name[0].lower() in "aeiou" else "A"
            ladder[3] = f"{article} {piece_name} move solves the problem."
        except (ValueError, AttributeError):
            pass
    return ladder


def _explanation_primitives(r: MoveRecord, board_before: chess.Board, board_after: chess.Board) -> dict:
    data = {
        "lesson_objective": r.lesson_objective,
        "played_move": r.san,
        "played_move_label": r.move_label,
        "best_move": r.best_move_san,
        "best_move_label": r.best_move_label,
        "punishment_after_played_move": (_numbered_line(r.fen_after, r.punishment_line[:6])
                                         if r.punishment_line else None),
        "diagnosis_sentence": _diagnosis_sentence(r),
    }
    if r.is_capture:
        try:
            played = chess.Move.from_uci(r.uci)
            captured = board_before.piece_at(played.to_square)
            if captured:
                data["human_temptation"] = f"won a {chess.piece_name(captured.piece_type)} on {chess.square_name(played.to_square)}"
        except ValueError:
            pass

    simple_capture_trade = _is_simple_capture_punishment(r)
    if simple_capture_trade:
        first = _numbered_line(r.fen_after, r.punishment_line[:1]) if r.punishment_line else "a recapture"
        data["why_wrong_move_fails"] = (
            f"{r.san} wins material but allows {first}, trading away the active piece "
            "instead of improving the position.")
        if r.best_move_san:
            activity = _best_move_activity_data(board_before, r.best_move_san)
            if activity:
                data["why_best_move_works"] = activity["text"]
                data["best_move_reason_squares"] = activity["squares"]
                data["best_move_keywords"] = activity["keywords"]
            else:
                data["why_best_move_works"] = (
                    f"{r.best_move_san} was the engine-verified improvement because it keeps the "
                    "piece active instead of trading it for material.")
                data["best_move_keywords"] = ["active", "trade", "material"]
    forcing = None if simple_capture_trade else _first_forcing_punishment(board_after, r.punishment_line)
    if forcing:
        san, move = forcing
        consequences = _forcing_move_consequences(board_after, san, move)
        data["forcing_move"] = san
        data["target_square"] = chess.square_name(move.to_square)
        data["forcing_move_consequences"] = consequences
        data["why_wrong_move_fails"] = (
            f"{r.san} does not address {san}; the opponent gets a forcing move on "
            f"{chess.square_name(move.to_square)}.")
        if r.best_move_san:
            works = _best_move_answer_to_target(board_before, r.best_move_san, move.to_square)
            if works:
                data["why_best_move_works"] = works
            if _best_move_controls(board_before, r.best_move_san, move.to_square):
                data["best_move_controls_target"] = True
            elif _best_line_allows(r.best_line, san):
                # The verified best line itself answers with the forcing move (e.g.
                # 31. Kh2 Qh5+): never claim the better move stops it; contrast the lines.
                data["forcing_move_not_prevented"] = True
                data["why_wrong_move_fails"] = (
                    f"After {r.move_label}, the opponent plays {_numbered_line(r.fen_after, [san])}. "
                    f"{r.best_move_san} also allows {san}; the difference is in the verified lines: after "
                    f"{r.best_move_san}: {_numbered_line(r.fen_before, r.best_line[:6]) or r.best_move_san}; "
                    f"after {r.san}: {_numbered_line(r.fen_after, r.punishment_line[:6])}.")
            else:
                data["forcing_move_prevented"] = True
    elif not simple_capture_trade and r.best_move_san:
        data["why_best_move_works"] = (
            f"{r.best_move_san} was the engine-verified improvement; use the line "
            f"{_numbered_line(r.fen_before, r.best_line)} to explain the concrete consequence." if r.best_line
            else f"{r.best_move_san} was the engine-verified improvement.")

    if r.lesson_objective == "check_opponent_forcing_replies":
        data["process_habit"] = ("After choosing a candidate move, ask: what forcing checks, captures, and "
                                 "threats does this move give my opponent?")
    elif r.time_flag:
        data["process_habit"] = "Before moving quickly, ask what forcing checks, captures, or threats the opponent has."
    elif r.is_capture:
        data["process_habit"] = "Before taking material, ask what the position's urgent priority is."
    else:
        data["process_habit"] = "Name the position's urgent priority before choosing a natural move."
    data["reasons"] = _reason_set(r, board_before, data)
    return data


# Motif-name fragments the student can plausibly *say* (so they double as
# explanation signal tokens). Verified motifs are detected in code; here we only
# map them to a word a human might use when explaining the point.
_MOTIF_KEYWORDS = {
    "fork", "pin", "skewer", "mate", "discovered", "double", "smothered",
    "deflection", "decoy", "interference", "clearance", "zwischenzug", "promotion",
    "back", "rank", "removal", "trap", "overload",
}


def _san_dest_square(board: chess.Board, san: str):
    """The destination square name of a SAN move on `board`, or None."""
    try:
        return chess.square_name(board.parse_san(san).to_square)
    except (ValueError, AttributeError):
        return None


def _motif_keywords(motif_dicts) -> list:
    """The human-sayable motif words present in a list of detected-motif dicts."""
    out = []
    for m in motif_dicts or []:
        name = (m.get("motif") or "").lower().replace("-", "_")
        out += [tok for tok in name.split("_") if tok in _MOTIF_KEYWORDS]
    return out


def _reason_set(r: MoveRecord, board_before: chess.Board, data: dict) -> list:
    """Structured, gradable reasons for *why* the moment matters (P0-2). Each is
    {text, key, squares, keywords}; exactly one `key` (load-bearing) reason is
    flagged when a concrete signal exists, so evaluate_explanation can tell a
    partial 'why' (e.g. 'gains tempo') from one that names the actual point
    (e.g. 'stops mate on h2') and probe the gap instead of rubber-stamping it."""
    reasons = []
    target = data.get("target_square")
    if target:
        # A forcing tactic is the point: the student must grasp the square it lands
        # on and (when present) the motif — not just that they "lost material".
        keywords = _motif_keywords(r.motifs) + _motif_keywords(r.best_line_motifs)
        if (data.get("forcing_move") or "").endswith("#"):
            keywords.append("mate")
        wrong_move = {
            "text": (data.get("why_wrong_move_fails") or data.get("forcing_move_consequences")
                     or f"the opponent's forcing move on {target}"),
            "squares": [target],
            "keywords": sorted(set(keywords)),
        }
        if data.get("best_move_controls_target") and data.get("why_best_move_works"):
            # The load-bearing reason is what the better move does to the threatened
            # square (e.g. 26. Rd4 controls g4), so "it defends my g pawn" can pass and
            # the probe asks what covers the square instead of inventing a target.
            dest = _san_dest_square(board_before, r.best_move_san)
            reasons.append({
                "text": data["why_best_move_works"],
                "key": True,
                "squares": [target] + ([dest] if dest and dest != target else []),
                "keywords": wrong_move["keywords"],
            })
            reasons.append({**wrong_move, "key": False})
        else:
            # Gradable unless the verified best line itself answers with the forcing move
            # (then a "what is being stopped?" probe would be false).
            reasons.append({**wrong_move, "key": not data.get("forcing_move_not_prevented")})
    elif data.get("why_best_move_works") and r.best_move_san:
        dest = _san_dest_square(board_before, r.best_move_san)
        squares = list(data.get("best_move_reason_squares") or ([dest] if dest else []))
        keywords = sorted(set(_motif_keywords(r.best_line_motifs)
                              + list(data.get("best_move_keywords") or [])))
        # Positional point: key when there is a concrete square or motif to anchor on.
        # Without verified best-move facts it is only the generic "engine-verified
        # improvement" sentence: keep it as the key reason (the human-labelled corpus
        # endorses its tokens) but mark it generic so the grader teaches instead of probing.
        reasons.append({
            "text": data["why_best_move_works"],
            "key": bool(squares or keywords),
            "generic": not data.get("best_move_reason_squares"),
            "squares": squares,
            "keywords": keywords,
        })
    if data.get("human_temptation"):
        m = re.search(r" on ([a-h][1-8])", data["human_temptation"])
        reasons.append({
            "text": data["human_temptation"], "key": False,
            "squares": [m.group(1)] if m else [], "keywords": ["material", "won"],
        })
    if r.best_move_san:
        reasons.extend(_basic_move_reasons(board_before, r.best_move_san))
    return reasons


def _basic_move_reasons(board: chess.Board, best_san: str) -> list:
    """Plain board facts about the better move that a student may name as reasons (never key).
    Run s2hy697w S11: "it moves my rook to safety, and offers a rook trade" was true (the g2 pawn
    attacked the f1 rook; Rfe1 faces the e8 rook) but nothing let the grader credit it."""
    try:
        move = board.parse_san(best_san)
    except ValueError:
        return []
    piece = board.piece_at(move.from_square)
    if piece is None:
        return []
    name, origin = chess.piece_name(piece.piece_type), chess.square_name(move.from_square)
    out = []
    attackers = sorted(board.attackers(not board.turn, move.from_square))
    if attackers:
        attacker = attackers[0]
        out.append({
            "text": (f"{best_san} moves the {name} on {origin} out of the attack by the "
                     f"{chess.piece_name(board.piece_at(attacker).piece_type)} on {chess.square_name(attacker)}"),
            "key": False, "squares": [origin, chess.square_name(attacker)],
            "keywords": ["safety", "safe", "escape", "away", "save", "saves", "threatened", "danger", "attacked"],
        })
    after = board.copy(stack=False)
    after.push(move)
    opposed = [sq for sq in after.attacks(move.to_square)
               if after.piece_at(sq) and after.piece_at(sq).color != piece.color
               and after.piece_at(sq).piece_type == piece.piece_type]
    if opposed and piece.piece_type != chess.KING:
        out.append({
            "text": f"{best_san} offers a trade of {name}s with the {name} on {chess.square_name(opposed[0])}",
            "key": False, "squares": [chess.square_name(opposed[0])],
            "keywords": ["trade", "trades", "exchange", "exchanges", "swap", "challenge", "challenges",
                         "oppose", "opposes", "contest"],
        })
    return out


def _diagnosis_sentence(r: MoveRecord) -> str:
    if _is_simple_capture_punishment(r):
        first = _numbered_line(r.fen_after, r.punishment_line[:1]) if r.punishment_line else "a recapture"
        return (f"Played {r.move_label} to win material, but after {first} the exchange did not solve "
                f"the position's main priority; {r.best_move_label or r.best_move_san} kept the stronger plan.")
    if _punishment_starts_with_forcing_check(r):
        first = _numbered_line(r.fen_after, r.punishment_line[:1]) if r.punishment_line else "a forcing check"
        return (f"Played {r.move_label}, which allowed the forcing {first}; "
                f"{r.best_move_label or r.best_move_san} was the engine-verified defence.")
    if r.lesson_objective == "improve_pieces_before_committing_pawns":
        return (f"Played {r.move_label} before improving the pieces; "
                f"{r.best_move_label or r.best_move_san} supported the position before changing the pawn structure.")
    if r.best_move_san:
        return f"Played {r.move_label}; {r.best_move_label or r.best_move_san} was the engine-verified improvement."
    return f"Played {r.move_label} and lost ground in the position."


def _first_forcing_punishment(board_after: chess.Board, punishment_line: list):
    if not punishment_line:
        return None
    try:
        move = board_after.parse_san(punishment_line[0])
    except ValueError:
        return None
    if board_after.gives_check(move) or board_after.is_capture(move):
        return punishment_line[0], move
    return None


def _forcing_move_consequences(board: chess.Board, san: str, move: chess.Move) -> dict:
    mover = board.turn
    moving_piece = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    after = board.copy(stack=False)
    after.push(move)
    consequences = {
        "move": san,
        "moving_piece": (chess.piece_name(moving_piece.piece_type) if moving_piece else None),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "captured_piece": (f"{chess.piece_name(captured.piece_type)} on {chess.square_name(move.to_square)}"
                           if captured else None),
        "gives_check": board.gives_check(move),
        "checks": None,
        "also_attacks": [],
    }
    if board.gives_check(move):
        king = after.king(not mover)
        consequences["checks"] = f"king on {chess.square_name(king)}" if king is not None else "king"
    attacker_sq = move.to_square
    attacked = []
    for sq in after.attacks(attacker_sq):
        piece = after.piece_at(sq)
        if piece and piece.color != mover and piece.piece_type != chess.KING:
            attacked.append(f"{chess.piece_name(piece.piece_type)} on {chess.square_name(sq)}")
    consequences["also_attacks"] = sorted(attacked)
    return consequences


def _best_move_activity_data(board_before: chess.Board, best_san: str) -> dict | None:
    try:
        best = board_before.parse_san(best_san)
    except ValueError:
        return None
    piece = board_before.piece_at(best.from_square)
    if not piece:
        return None
    after = board_before.copy(stack=False)
    after.push(best)
    moved_piece = after.piece_at(best.to_square)
    if not moved_piece:
        return None
    attacked = []
    keywords = ["active"]
    for sq in after.attacks(best.to_square):
        target = after.piece_at(sq)
        if not target or target.color == moved_piece.color or target.piece_type == chess.PAWN:
            continue
        attacked.append((sq, target))
        keywords.append(chess.piece_name(target.piece_type))
    if attacked:
        square, target = attacked[0]
        target_name = chess.piece_name(target.piece_type)
        if target.piece_type in (chess.QUEEN, chess.ROOK, chess.KING):
            keywords.append("tempo")
        return {
            "text": (f"After {best_san}, the {chess.piece_name(piece.piece_type)} moves to "
                     f"{chess.square_name(best.to_square)} and attacks the {target_name} on "
                     f"{chess.square_name(square)} with tempo, keeping the piece active instead "
                     "of trading it for material."),
            "squares": [chess.square_name(best.to_square), chess.square_name(square)],
            "keywords": sorted(set(keywords)),
        }
    return {
        "text": (f"After {best_san}, the {chess.piece_name(piece.piece_type)} improves to "
                 f"{chess.square_name(best.to_square)}, keeping the piece active instead of "
                 "trading it for material."),
        "squares": [chess.square_name(best.to_square)],
        "keywords": ["active", "trade", "material"],
    }


def _best_move_answer_to_target(board_before: chess.Board, best_san: str, target_sq: chess.Square) -> str:
    try:
        best = board_before.parse_san(best_san)
    except ValueError:
        return None
    piece = board_before.piece_at(best.from_square)
    after_best = board_before.copy(stack=False)
    after_best.push(best)
    if target_sq in after_best.attacks(best.to_square):
        return (f"After {best_san}, the {chess.piece_name(piece.piece_type) if piece else 'piece'} "
                f"on {chess.square_name(best.to_square)} controls {chess.square_name(target_sq)}, "
                "so the opponent's forcing capture can be met tactically.")
    defenders = [sq for sq in after_best.attackers(not after_best.turn, target_sq)
                 if after_best.piece_at(sq) and after_best.piece_at(sq).color == board_before.turn]
    if defenders:
        names = ", ".join(chess.square_name(sq) for sq in defenders[:3])
        return f"After {best_san}, {chess.square_name(target_sq)} is defended by {names}."
    return None


def _best_move_controls(board_before: chess.Board, best_san: str, target_sq: chess.Square) -> bool:
    """True when the better move's own piece lands where it controls the target square."""
    try:
        best = board_before.parse_san(best_san)
    except ValueError:
        return False
    after_best = board_before.copy(stack=False)
    after_best.push(best)
    return target_sq in after_best.attacks(best.to_square)


def _best_line_allows(best_line: list, forcing_san: str) -> bool:
    """True when the opponent's reply in the verified best line is the forcing move itself."""
    def bare(san):
        return str(san or "").rstrip("+#!?")
    return len(best_line or []) > 1 and bare(best_line[1]) == bare(forcing_san)


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
    """What do players a few hundred points above the STUDENT'S level choose here?
    The next level's instinct, as a concrete target. Anchored on conversation_elo
    (who the student is now), NOT the game-time rating — aspiration is for the
    student's growth, so an old 1140-rated game still aspires from today's level
    (plan P0-4a). Falls back to the mover's rating for duck-typed reviews (batch)."""
    base = (getattr(review, "conversation_elo", None)
            or (review.white_elo if r.color == "white" else review.black_elo)
            or DEFAULT_USER_ELO)
    target_elo = min(base + 350, 1900)
    if target_elo <= base + 50:
        return
    asp = engines.maia_predict(r.fen_before, elo_self=target_elo, elo_oppo=target_elo,
                               history_uci=(r.history_uci or None), tier="deep")
    if asp and asp["top_moves"]:
        # Divergence guard (P0-6): only surface an aspirational instinct when the
        # stronger band would play something OTHER than the move the student played.
        # The digest frames this as "the instinct you're growing toward," so an
        # unguarded same-as-played result would dress the student's own move (often
        # their mistake) up as the target. Canonical case: game 2dtDqQgRy2OD move 24,
        # where the +350 top move is the played Kxg2 (not the better Rfe1) → suppress.
        if asp["top_moves"][0]["move"] != r.san:
            r.aspirational = {"elo": target_elo, **asp["top_moves"][0]}


# Within-game SLIDE detection: EVAL-ONLY (no Maia). This replaces the rejected
# Phase 4 maia "skill band" trajectory (calibrated & rejected 2026-06-14 — see
# design/archive/designs/maia3_integration_design_20260613.md §8.4 / §15 FR-3). That method tried
# to spot "moves that look like a weaker player's", and was structurally confounded
# (bad moves are rare at all bands -> read 'level' -> missed). The eval stream is the
# signal that actually survived: a "slide" is a stretch of several mediocre moves
# that together hand the game away with NO single blunder — exactly the pattern the
# move-by-move agenda misses (each move alone isn't agenda-worthy).
#
# Validated on 1,700+ analysed Lichess games across two players (~1500 and ~2000):
# genuine gradual slides occur in ~16-18% of games, and the win%-space thresholds
# below are rating-stable (a 500-Elo gap moved the rate by ~2 points). The win%
# frame is self-normalizing, so "lose 25 net win% over 3+ moves" means the same
# thing at every level. Detector + corpus bench: scripts/scan_eval_slides.py.
SLIDE_PLY_DROP_MIN = 3.0       # a student move "loses" if it sheds >= this much win%
SLIDE_MIN_RUN = 3              # consecutive losing student moves to form a run
SLIDE_MIN_START_WINPCT = 30.0  # the run must START from a non-floored position (had chances)
SLIDE_NET_LOSS_MIN = 25.0      # net (start->end) win% lost across the run
SLIDE_CLIFF_FRACTION = 0.70    # if one move is >= this share of the NET loss it's a single
                               # blunder (the agenda already covers it), not a gradual slide


def _slide_onset_state(start_winpct: float) -> str:
    """Classify the student's standing when a slide began (start win% is gated
    >= SLIDE_MIN_START_WINPCT, so the floor is ~30)."""
    if start_winpct >= 55:
        return "ahead"
    if start_winpct < 45:
        return "worse"
    return "level"


def _slide_queens_on(fen: str) -> bool:
    """Whether queens were still on the board at slide onset (a coarse proxy for
    'sharp, attack-laden' vs 'simplified' positions)."""
    try:
        b = chess.Board(fen)
    except ValueError:
        return False
    return bool(b.pieces(chess.QUEEN, chess.WHITE) or b.pieces(chess.QUEEN, chess.BLACK))


def _detect_slides(review: GameReview) -> None:
    """Flag sustained one-sided eval slides in the student's play: >= SLIDE_MIN_RUN
    consecutive moves each losing win%, starting from a non-floored position, shedding
    a large NET amount, with no single dominating blunder. Eval-only — reads the
    win%-space already on each MoveRecord, needs no engine. Sets review.slides or
    leaves it None.

    NET (start win% before the run minus end win% after it) is the key measure, not
    the sum of per-move drops: summing inflates mutual-blunder churn where the
    opponent keeps handing win% back. A real slide is a NET one-sided collapse."""
    sc = review.student_color
    if sc is None:
        return
    student_white = sc == "white"
    rows = []   # (record, winpct_before, winpct_after, drop)
    for r in review.records:
        if r.color != sc or r.phase == "opening":
            continue
        if r.eval_before_cp is None or r.eval_after_cp is None:
            continue
        cb = r.eval_before_cp if student_white else -r.eval_before_cp
        ca = r.eval_after_cp if student_white else -r.eval_after_cp
        rows.append((r, cp_to_win_pct(cb), cp_to_win_pct(ca), r.drop_pct))

    segments, i, n = [], 0, len(rows)
    while i < n:
        if rows[i][3] < SLIDE_PLY_DROP_MIN or rows[i][1] < SLIDE_MIN_START_WINPCT:
            i += 1
            continue
        j = i
        while j < n and rows[j][3] >= SLIDE_PLY_DROP_MIN:
            j += 1
        seg = rows[i:j]
        net = seg[0][1] - seg[-1][2]
        max_single = max(x[3] for x in seg)
        if (len(seg) >= SLIDE_MIN_RUN and net >= SLIDE_NET_LOSS_MIN
                and max_single < SLIDE_CLIFF_FRACTION * net):
            onset = seg[0][0]
            start_wp = seg[0][1]
            segments.append({
                "start_move": onset.number, "end_move": seg[-1][0].number,
                "plies": len(seg), "start_winpct": round(start_wp, 1),
                "end_winpct": round(seg[-1][2], 1), "net_loss": round(net, 1),
                "max_single": round(max_single, 1),
                # Co-markers: factual co-occurrence captured at slide onset, so the
                # cross-game batch aggregate can describe WHEN slides happen (ahead /
                # queens on / time pressure) — never a causal/psychological claim.
                "onset": _slide_onset_state(start_wp),
                "queens_on": _slide_queens_on(onset.fen_before),
                # clock < 60s if the PGN carried clocks, else None (unknown — never
                # counted against the player in the aggregate).
                "clock_low": (onset.clock_remaining < 60)
                             if onset.clock_remaining is not None else None,
            })
        i = j

    if segments:
        review.slides = segments


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


def _check_created_by_move(before: chess.Board, sequence: list) -> bool:
    """Replay `sequence` (opponent first) as if the student had passed instead of moving: the
    student's move created the final check if a move is illegal there or the check is no check."""
    probe = before.copy(stack=False)
    probe.push(chess.Move.null())
    for move in sequence[:-1]:
        if not probe.is_legal(move):
            return True
        probe.push(move)
    return not probe.is_legal(sequence[-1]) or not probe.gives_check(sequence[-1])


def _allows_new_forcing_sequence(record: MoveRecord) -> bool:
    """#9: the student's move CREATED the opponent's forcing check (high precision, low recall).

    Two shapes, each compared against the pre-move position with the student passing:
    - the opponent's first check within three plies of the punishment line is new
      (ZU8uPxvn 10...f5 opens a2-g8: 11. d4 Qd6 12. Bc4+);
    - the opponent captures, the engine line declines the recapture, and recapturing would
      allow a new checking capture the student cannot take back (YbjNb8wD 16. Re3? Bxd2: 17. Qxd2
      runs into ...Qxa1+, a check the rook used to block; the engine line plays 17. Re2 instead).
    A check that was already there is an ignored threat; in-check moments, mates (king
    safety) and simple recaptures are excluded. Lines without a check are not used: a capture
    that became possible usually means a hanging piece."""
    if not record.punishment_line or _is_simple_capture_punishment(record):
        return False
    if {m.get("motif") for m in record.motifs} & {"checkmate", "back_rank_mate", "smothered_mate"}:
        return False
    try:
        before = chess.Board(record.fen_before)
        after = chess.Board(record.fen_after)
        if before.is_check():
            return False
        line, walk = [], after.copy(stack=False)
        for ply, san in enumerate(record.punishment_line[:3]):
            move = walk.parse_san(san)
            line.append(move)
            if ply % 2 == 0 and walk.gives_check(move):
                return _check_created_by_move(before, line)
            walk.push(move)
    except (ValueError, AssertionError):
        return False
    return _refuted_recapture_is_new(before, after, line)


def _refuted_recapture_is_new(before: chess.Board, after: chess.Board, line: list) -> bool:
    capture = line[0]
    if not after.is_capture(capture):
        return False
    board = after.copy(stack=False)
    board.push(capture)
    if len(line) > 1 and line[1].to_square == capture.to_square and board.is_capture(line[1]):
        return False                      # the engine line recaptures: nothing refuted it
    for recapture in [m for m in board.legal_moves if m.to_square == capture.to_square]:
        reply_board = board.copy(stack=False)
        reply_board.push(recapture)
        for reply in list(reply_board.legal_moves):
            if not (reply_board.is_capture(reply) and reply_board.gives_check(reply)):
                continue
            reply_board.push(reply)
            safe = not any(m.to_square == reply.to_square for m in reply_board.legal_moves)
            reply_board.pop()
            if safe and _check_created_by_move(before, [capture, recapture, reply]):
                return True
    return False


def suggest_error_categories(record: MoveRecord) -> list:
    """Evidence-based diagnosis suggestions: tactics in the line the student
    MISSED map to missed_*; tactics in the opponent's punishment map to what
    the student failed to see coming."""
    suggestions = []
    if _is_simple_capture_punishment(record):
        suggestions.append("bad_trade")
    try:
        played = chess.Move.from_uci(record.uci)
        before = chess.Board(record.fen_before)
        piece = before.piece_at(played.from_square)
        best = before.parse_san(record.best_move_san) if record.best_move_san else None
        best_piece = before.piece_at(best.from_square) if best else None
        if (piece and piece.piece_type == chess.PAWN and best_piece
                and best_piece.piece_type in (chess.ROOK, chess.KNIGHT, chess.BISHOP)
                and not record.is_capture):
            suggestions.append("premature_attack")
    except (ValueError, AssertionError, AttributeError):
        pass
    missed_kinds = {m["motif"] for m in record.best_line_motifs}
    punish_kinds = {m["motif"] for m in record.motifs}

    if any(k.endswith("fork") for k in missed_kinds):
        if "premature_attack" not in suggestions:
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
    elif _allows_new_forcing_sequence(record):
        suggestions.append("allows_forcing_sequence")
    elif (not _is_simple_capture_punishment(record)
          and (any(k.endswith("fork") for k in punish_kinds)
          or punish_kinds & {"pin", "skewer", "discovered_check", "discovered_attack",
                              "double_check", "removal_of_defender"})):
        suggestions.append("ignored_threat")
    elif record.punishment_line and record.punishment_line[0].endswith(("+", "#")):
        # The refutation is an immediate forcing check: whatever else the move
        # was trying to do (e.g. grab material), the root error was ignoring a
        # direct threat — sharper than the is_capture -> bad_trade fallback.
        suggestions.append("ignored_threat")

    if record.time_flag in ("hope_chess", "time_scramble"):
        suggestions.append("time_management")
    deduped = []
    for suggestion in suggestions:
        if suggestion not in deduped:
            deduped.append(suggestion)
    return deduped[:2]


def attach_board_tokens(review: GameReview, state, invert: bool = False) -> None:
    """Pre-mint a real diagram token for each agenda moment so the model never
    has a reason to invent one. Oriented to the student's side."""
    from app.render import render_board_token
    pov = review.student_color or "white"
    for record in review.agenda:
        board = chess.Board(record.fen_before)
        record.board_token = render_board_token(state, board, pov, invert)


def _numbered_line(fen: str, sans: list) -> str:
    """A SAN line with move numbers from `fen`, so it says whose move each is ("20... Nxb5 21. Qxb5").
    Unnumbered, a punishment line starting with the opponent's reply was replayed from the wrong
    side (live evals: "Nxb5 Qxb5" became the illegal "20... Qxb5"). Unparseable moves stay as given."""
    try:
        board = chess.Board(fen)
    except (TypeError, ValueError):
        return " ".join(str(s) for s in sans)
    out = []
    for i, san in enumerate(sans):
        if board.turn == chess.WHITE:
            out.append(f"{board.fullmove_number}. {san}")
        else:
            out.append(f"{board.fullmove_number}... {san}" if i == 0 else str(san))
        try:
            board.push_san(str(san))
        except ValueError:
            out.extend(str(s) for s in sans[i + 1:])
            break
    return " ".join(out)


def _pv_to_san(board: chess.Board, pv) -> list:
    sans, b = [], board.copy(stack=False)
    for m in pv:
        try:
            sans.append(b.san(m))
            b.push(m)
        except (ValueError, AssertionError):
            break
    return sans
