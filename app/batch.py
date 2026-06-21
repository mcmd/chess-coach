"""Batch pattern review: scan many games cheaply, cluster the student's
recurring decision errors deterministically, deep-verify a few representative
moments, and emit a training-plan digest the coach teaches from.

Backend-led by design: clustering uses only verified scan data (no LLM in the
loop), and everything the coach actually teaches goes through the same deep
enrichment as single-game agenda moments — so every grounding guardrail
(board tokens, question ladders, diagnosis governance, explanation grounding)
applies unchanged.

Per-game scans are cached on disk keyed by game identity, so an interrupted
run resumes for free and re-uploaded PGNs only scan the new games.
"""
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import date

import chess
import chess.pgn

from app import trace as tracing
from app.config import DEFAULT_USER_ELO
from app.review import (GameReview, MoveRecord, review_game, _enrich_agenda,
                        detect_student_color, moment_digest_lines)

# Bump when MoveRecord scan fields or classification change incompatibly.
# v2: scans now carry review.slides (eval-only within-game slides + co-markers).
# v3: MoveRecord.prev_move_san (opponent's move into fen_before, for orientation).
# v4: scan prefers the PGN's [%eval] for the win% stream (eval values/drops change);
#     scans record eval_source.
SCAN_VERSION = 4

# A cluster is only a "pattern" with enough evidence to be a fact, not noise.
MIN_PATTERN_MOMENTS = 3
MIN_PATTERN_GAMES = 2

# Cross-game slide habit (eval-only): aggregate review.slides across the batch.
MIN_SLIDE_GAMES = 2            # surface the habit only with >= this many slide games
SLIDE_BASELINE_RATE = 0.17    # corpus base rate of gradual slides (scan_eval_slides.py)
SLIDE_ABOVE_BASELINE = 1.5    # flag "above typical" when the player's rate >= this x baseline
# A situational co-marker (ahead / queens-on / time pressure) is asserted as a
# pattern only with enough slides AND a clear majority sharing it — else it stays
# raw data. Same "refuse weak claims" discipline that sank the Phase-4 method.
SLIDE_MARKER_MIN_GAMES = 3
SLIDE_MARKER_FRACTION = 0.6

PATTERN_TITLES = {
    "defend_against_direct_threat": "Playing on while a forcing threat hangs over the position",
    "left_piece_en_prise": "Leaving the just-moved piece where it can simply be taken",
    "material_temptation_vs_position_priority": "Grabbing material before checking what it costs",
    "threat_scan_under_time_pressure": "Skipping the threat scan when the clock is low",
    "priority_selection": "Choosing a natural move over the position's urgent priority",
}

PATTERN_HABITS = {
    "defend_against_direct_threat": "Before every move, name the opponent's checks and captures first.",
    "left_piece_en_prise": "After choosing a move, ask: can the piece I just moved simply be taken?",
    "material_temptation_vs_position_priority": "Before taking material, name the position's urgent priority.",
    "threat_scan_under_time_pressure": "Keep one fixed scan (checks, captures, threats) even with a low clock.",
    "priority_selection": "Name the position's urgent priority before choosing a natural move.",
}


def game_key_for(headers: dict, movetext_ucis: str) -> str:
    """Stable game identity: the Lichess ID when the Site header has one,
    else a movetext hash (same recipe as local-game lesson resume)."""
    site = (headers or {}).get("Site", "") or ""
    m = re.search(r"lichess\.org/(\w{8})", site)
    if m:
        return m.group(1)
    return "local-" + hashlib.sha1(movetext_ucis.encode()).hexdigest()[:12]


def split_games(pgn_text: str, max_games: int):
    """First `max_games` games with moves from a multi-game PGN.
    Returns (games, truncated)."""
    games = []
    pgn_io = io.StringIO(pgn_text)
    while len(games) < max_games:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return games, False
        if list(game.mainline()):
            games.append(game)
    return games, chess.pgn.read_game(pgn_io) is not None


# ---------------------------------------------------------------------------
# Pass 1: fast scan with per-game disk cache
# ---------------------------------------------------------------------------

def batches_dir(config) -> str:
    base = os.path.dirname(getattr(config, "lesson_file", "") or "") or "."
    return os.path.join(base, "batches")


def _scan_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, "scans", f"{key}.json")


def _load_cached_scan(cache_dir: str, key: str):
    path = _scan_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            scan = json.load(f)
        if scan.get("scan_version") != SCAN_VERSION:
            return None
        return scan
    except (json.JSONDecodeError, OSError):
        return None


def _save_scan(cache_dir: str, key: str, scan: dict) -> None:
    path = _scan_path(cache_dir, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scan, f)
    os.replace(tmp, path)


def scan_games(pgn_text: str, engines, config, cache_dir: str,
               on_progress=None, on_status=None):
    """Fast-scan every game (cache-aware). Returns (scans, skipped, truncated).
    Each scan is a JSON-safe dict with full scan-level MoveRecords."""
    max_games = int(getattr(config, "batch_max_games", 50))
    games, truncated = split_games(pgn_text, max_games)
    scan_eval_time = float(getattr(config, "batch_eval_time", 0.4))
    scans, skipped = [], 0
    n = len(games)

    # Plan the work before scanning so the progress bar (and its ETA) can span the
    # WHOLE PGN instead of resetting per game: resolve student/cache for each game
    # once, and total the plies that will actually be scanned. Cached and skipped
    # games emit no per-ply progress, so they are excluded from the grand total.
    plan, grand_total = [], 0
    for game in games:
        headers = dict(game.headers)
        student_color = detect_student_color(headers, config)
        ucis = " ".join(m.uci() for m in game.mainline_moves())
        key = game_key_for(headers, ucis)
        skip = bool(getattr(config, "player_name", None)) and student_color is None
        cached = None if skip else _load_cached_scan(cache_dir, key)
        n_plies = 0 if (skip or cached is not None) else (ucis.count(" ") + 1 if ucis else 0)
        grand_total += n_plies
        plan.append((game, headers, key, skip, cached, n_plies))

    # review_game reports (ply, game_total); remap onto the batch-wide total so the
    # one bar advances monotonically and never resets between games.
    base = 0

    def batch_progress(done, _game_total):
        if on_progress:
            on_progress(base + done, grand_total or 1)

    for i, (game, headers, key, skip, cached, n_plies) in enumerate(plan, 1):
        if skip:
            skipped += 1
            if on_status:
                on_status(f"game {i}/{n}: skipped (student not in headers)")
            continue
        if cached is not None:
            scans.append(cached)
            if on_status:
                on_status(f"game {i}/{n}: cached ({key})")
            continue
        if on_status:
            on_status(f"game {i}/{n}: scanning ({key})")
        review = review_game(str(game), engines, config,
                             on_progress=(batch_progress if on_progress else None),
                             eval_time=scan_eval_time, with_maia=False, enrich=False)
        if review is None:
            skipped += 1
            base += n_plies
            continue
        scan = {
            "game_key": key,
            "headers": headers,
            "student_color": review.student_color,
            "scan_version": SCAN_VERSION,
            "scanned_with": {"eval_time": scan_eval_time, "with_maia": False},
            "eval_source": review.eval_source,   # "pgn" ([%eval]) or "engine" (local search)
            "records": [asdict(r) for r in review.records],
            "slides": review.slides or [],   # eval-only within-game slides (+ co-markers)
        }
        _save_scan(cache_dir, key, scan)
        scans.append(scan)
        base += n_plies
    return scans, skipped, truncated


# ---------------------------------------------------------------------------
# Pass 1.5: deterministic clustering (no engine time)
# ---------------------------------------------------------------------------

def _light_objective(rec: MoveRecord) -> str:
    """Cheap, honest mistake classification from scan data only. This is a
    GROUPING heuristic — the deep pass attaches the real recommended_diagnosis
    to everything that gets taught."""
    punisher = None
    try:
        board_after = chess.Board(rec.fen_after)
        if rec.punisher_uci:
            mv = chess.Move.from_uci(rec.punisher_uci)
            if mv in board_after.legal_moves:
                punisher = mv
    except ValueError:
        board_after = None
    if punisher is not None and board_after.gives_check(punisher):
        return "defend_against_direct_threat"
    if (punisher is not None and board_after.is_capture(punisher)
            and punisher.to_square == chess.Move.from_uci(rec.uci).to_square):
        return "left_piece_en_prise"
    if rec.is_capture:
        return "material_temptation_vs_position_priority"
    if rec.time_flag in ("hope_chess", "time_scramble"):
        return "threat_scan_under_time_pressure"
    return "priority_selection"


@dataclass
class BatchPattern:
    pattern_id: str          # "<objective>:<phase>"
    objective: str
    phase: str
    title: str               # template text, never LLM-authored
    habit: str
    count: int
    games_seen: int
    total_drop: float
    moments: list            # all member moments: {game_key, move_label, san, drop_pct, fen_before, in_slide}
    representatives: list = field(default_factory=list)  # enriched MoveRecords (deep pass)
    # Knowledge-gap vs execution axis: how many of this cluster's moments fell
    # INSIDE a within-game slide. Mostly-inside => discipline/execution leak (coach
    # the meta-skill); mostly-outside => knowledge gap (teach the pattern).
    in_slide_count: int = 0
    axis: str = None         # "execution" | "knowledge" | "mixed" | None


def _student_mistakes(scan: dict):
    color = scan.get("student_color")
    for d in scan.get("records", []):
        if color and d.get("color") != color:
            continue
        if (d.get("drop_pct") or 0) >= 10:
            yield d


def _move_in_slides(move_number: int, slides: list) -> bool:
    """Whether a student move (by fullmove number) falls inside any of the game's
    within-game slide segments."""
    return any(s["start_move"] <= move_number <= s["end_move"] for s in (slides or []))


def aggregate_patterns(scans: list) -> list:
    """Group student mistakes by (light objective, phase); keep clusters with
    enough cross-game evidence; rank by count then total win% lost. Each moment is
    tagged in_slide (did it fall inside a within-game slide?), and each pattern gets
    a knowledge-vs-execution axis from how many of its moments did."""
    slides_by_game = {s["game_key"]: s.get("slides") or [] for s in scans}
    clusters = {}
    for scan in scans:
        for d in _student_mistakes(scan):
            rec = MoveRecord(**d)
            objective = _light_objective(rec)
            key = (objective, rec.phase or "?")
            entry = clusters.setdefault(key, {"moments": [], "games": set(), "drop": 0.0})
            entry["moments"].append({"game_key": scan["game_key"],
                                     "move_label": rec.move_label,
                                     "san": rec.san,
                                     "drop_pct": rec.drop_pct,
                                     "fen_before": rec.fen_before,
                                     "in_slide": _move_in_slides(rec.number,
                                                                 slides_by_game.get(scan["game_key"])),
                                     "record": d})
            entry["games"].add(scan["game_key"])
            entry["drop"] += rec.drop_pct
    patterns = []
    for (objective, phase), entry in clusters.items():
        if len(entry["moments"]) < MIN_PATTERN_MOMENTS or len(entry["games"]) < MIN_PATTERN_GAMES:
            continue
        title = PATTERN_TITLES.get(objective, objective.replace("_", " "))
        if phase and phase != "?":
            title = f"{title} ({phase})"
        in_slide = sum(1 for m in entry["moments"] if m["in_slide"])
        frac = in_slide / len(entry["moments"])
        axis = "execution" if frac >= 0.6 else ("knowledge" if frac <= 0.4 else "mixed")
        patterns.append(BatchPattern(
            pattern_id=f"{objective}:{phase}", objective=objective, phase=phase,
            title=title, habit=PATTERN_HABITS.get(objective, ""),
            count=len(entry["moments"]), games_seen=len(entry["games"]),
            total_drop=round(entry["drop"], 1),
            moments=sorted(entry["moments"], key=lambda m: -m["drop_pct"]),
            in_slide_count=in_slide, axis=axis,
        ))
    patterns.sort(key=lambda p: (-p.count, -p.total_drop))
    return patterns


def aggregate_slides(scans: list) -> dict:
    """Aggregate the eval-only within-game slides across the batch into a cross-game
    'slide habit' signature, or None if it doesn't clear the min-count gate.

    Reports the player's slide rate against the validated ~17% corpus baseline, and
    surfaces a situational co-marker (ahead / queens-on / time-pressure) only when it
    is shared by a clear majority of a meaningful number of slides — otherwise the
    markers stay raw counts. Factual co-occurrence only; the prompt narrates no cause."""
    games_total = len(scans)
    worst_per_game, all_segs, engine_eval_slides = [], [], 0
    for s in scans:
        segs = s.get("slides") or []
        if not segs:
            continue
        all_segs.extend(segs)
        worst = max(segs, key=lambda x: x["net_loss"])
        worst_per_game.append({"game_key": s["game_key"], **worst})
        if s.get("eval_source", "engine") != "pgn":
            engine_eval_slides += 1   # slide read off a local search, not Lichess evals
    n = len(worst_per_game)
    if n < MIN_SLIDE_GAMES:
        return None

    rate = round(n / games_total, 2) if games_total else None
    total = len(all_segs)
    ahead = sum(1 for x in all_segs if x.get("onset") == "ahead")
    queens = sum(1 for x in all_segs if x.get("queens_on"))
    clk_known = [x for x in all_segs if x.get("clock_low") is not None]
    clk_low = sum(1 for x in clk_known if x.get("clock_low"))

    markers = []
    if n >= SLIDE_MARKER_MIN_GAMES:
        if total and ahead / total >= SLIDE_MARKER_FRACTION:
            markers.append("when AHEAD (consolidating a winning position)")
        if total and queens / total >= SLIDE_MARKER_FRACTION:
            markers.append("with QUEENS still on (under sustained attacking pressure)")
        if clk_known and clk_low / len(clk_known) >= SLIDE_MARKER_FRACTION:
            markers.append("on a LOW CLOCK (time pressure)")

    return {
        "slide_games": n,
        "games_total": games_total,
        "rate": rate,
        "baseline": SLIDE_BASELINE_RATE,
        "above_baseline": rate is not None and rate >= SLIDE_BASELINE_RATE * SLIDE_ABOVE_BASELINE,
        "markers": markers,
        "marker_counts": {"ahead": ahead, "queens_on": queens,
                          "clock_low": clk_low, "clock_known": len(clk_known), "total": total},
        "engine_eval_slides": engine_eval_slides,
        "examples": sorted(worst_per_game, key=lambda x: -x["net_loss"])[:5],
    }


def select_representatives(pattern: BatchPattern, per_pattern: int) -> list:
    """Highest-drop moments, at most one per game — patterns are cross-game
    facts, and the teaching examples should look that way."""
    chosen, used_games = [], set()
    for m in pattern.moments:
        if m["game_key"] in used_games:
            continue
        chosen.append(m)
        used_games.add(m["game_key"])
        if len(chosen) >= per_pattern:
            break
    return chosen


def compute_strengths(scans: list) -> list:
    """Verified credit from pass-1 data: per-phase accuracy and only-moves."""
    per_phase, only_moves = {}, 0
    for scan in scans:
        color = scan.get("student_color")
        for d in scan.get("records", []):
            if color and d.get("color") != color:
                continue
            ph = d.get("phase") or "?"
            good, total = per_phase.get(ph, (0, 0))
            per_phase[ph] = (good + (1 if (d.get("drop_pct") or 0) <= 2 else 0), total + 1)
            if d.get("quality") == "only_move":
                only_moves += 1
    lines = []
    for ph in ("opening", "middlegame", "endgame"):
        if ph in per_phase and per_phase[ph][1]:
            good, total = per_phase[ph]
            lines.append(f"{ph}: {good}/{total} moves lost <=2 win% ({good / total:.0%} accurate)")
    if only_moves:
        lines.append(f"found {only_moves} only-move(s) under pressure across the batch")
    return lines


# ---------------------------------------------------------------------------
# Pass 2: deep verification of representatives + the BatchReview object
# ---------------------------------------------------------------------------

def _backfill_maia(record: MoveRecord, engines, elo: int) -> None:
    """Pass 1 skips Maia; representatives get the full typicality treatment
    here, mirroring the single-game scan's thresholds."""
    try:
        with tracing.suspended():
            maia = engines.maia_predict(record.fen_before, elo_self=elo, elo_oppo=elo)
    except Exception:
        return
    if not maia:
        return
    record.maia_top = maia["top_moves"]
    record.human_win_prob = maia.get("win_probability_for_side_to_move")
    record.human_draw_prob = maia.get("draw_probability")
    record.human_loss_prob = maia.get("loss_probability")
    prob = maia["raw_probs"].get(record.uci)
    record.maia_played_prob = round(float(prob), 3) if prob is not None else None
    natural_uci = max(maia["raw_probs"], key=maia["raw_probs"].get)
    record.was_maia_top = natural_uci == record.uci
    try:
        best_uci = chess.Board(record.fen_before).parse_san(record.best_move_san).uci() \
            if record.best_move_san else None
    except ValueError:
        best_uci = None
    if best_uci:
        best_prob = maia["raw_probs"].get(best_uci)
        record.best_move_human_prob = round(float(best_prob), 3) if best_prob is not None else 0.0
    played = record.maia_played_prob if record.maia_played_prob is not None else 0.0
    if record.quality in ("blunder", "mistake"):
        if played >= 0.30:
            record.error_typicality = "typical_for_level"
        elif played <= 0.08:
            record.error_typicality = "atypical_for_level"


def _severity_label(avg_drop: float) -> str:
    """Map an average win% drop to the move-quality word the student already
    knows (same thresholds as review.classify_move: blunder >=30, mistake >=20,
    inaccuracy >=10). Lets the summary describe a pattern's COST in plain terms
    instead of an unbounded cumulative win% sum."""
    if avg_drop >= 30:
        return "a blunder"
    if avg_drop >= 20:
        return "a mistake"
    if avg_drop >= 10:
        return "an inaccuracy"
    return "a minor slip"


@dataclass
class BatchReview:
    """Duck-typed as a GameReview so state.review = batch works with every
    existing guardrail: answer grading by FEN, diagnosis governance and
    explanation grounding, board tokens, lesson persistence."""
    batch_id: str
    created: str
    games_analyzed: int
    games_skipped: int
    truncated: bool
    moments_found: int
    patterns: list           # BatchPattern, ranked
    strengths: list
    records: list = field(default_factory=list)   # enriched representatives, pattern-major order
    agenda_plies: list = field(default_factory=list)
    rep_pattern_idx: list = field(default_factory=list)  # records[i] belongs to patterns[rep_pattern_idx[i]]
    rep_sources: list = field(default_factory=list)      # records[i] came from this game: {game_key, white, black}
    slide_signature: dict = None    # cross-game slide habit (eval-only), or None below the gate
    student_color: str = None       # mixed across games; per-record color is authoritative
    white_elo: int = None           # student elo (used by enrichment helpers)
    black_elo: int = None
    game = None                     # no single game: the timeline is not loaded in batch mode

    def record_at(self, ply: int):
        return self.records[ply] if 0 <= ply < len(self.records) else None

    @property
    def agenda(self):
        return [self.records[p] for p in self.agenda_plies]

    def _slide_signature_lines(self) -> list:
        """The CROSS-GAME SLIDE PATTERN block (eval-only). Empty when the habit is
        below the min-count gate."""
        sig = self.slide_signature
        if not sig:
            return []
        n, total = sig["slide_games"], sig["games_total"]
        pct = f"{sig['rate']:.0%}" if sig.get("rate") is not None else "?"
        out = ["", "CROSS-GAME SLIDE PATTERN (eval-only; cumulative one-sided losses, no single blunder):"]
        head = (f"The student had a sustained within-game slide in {n}/{total} games (~{pct}, "
                f"vs ~{sig['baseline']:.0%} typical")
        head += "; well ABOVE typical)." if sig.get("above_baseline") else ")."
        out.append(head)
        if sig.get("markers"):
            out.append("These slides cluster " + "; ".join(sig["markers"]) + ".")
        ex = ", ".join(f"game {e['game_key']} moves {e['start_move']}-{e['end_move']} "
                       f"(-{e['net_loss']:.0f}% over {e['plies']})" for e in sig.get("examples", [])[:3])
        if ex:
            out.append(f"Worst examples: {ex}.")
        if sig.get("engine_eval_slides"):
            out.append(f"RELIABILITY CAVEAT: {sig['engine_eval_slides']} of these slide games had NO "
                       "[%eval] annotations, so their win% came from a short LOCAL search, not deep "
                       "analysis — that read is shakier. Present those tentatively (\"this looks like a "
                       "slide\") rather than as established fact.")
        out.append("Coach this as a META-SKILL (consolidating advantages, re-scanning when the "
                   "position turns, holding focus through rising complexity) — it is a factual win% "
                   "pattern across games, NEVER a claim about the student's psychology. State only what "
                   "the evals and the co-markers show; do not mind-read.")
        return out

    def digest(self) -> str:
        lines = [f"=== BATCH TRAINING REPORT (backend-verified across {self.games_analyzed} games) ==="]
        note = []
        if self.games_skipped:
            note.append(f"{self.games_skipped} game(s) skipped (student not found or unparseable)")
        if self.truncated:
            note.append("input had more games than the configured maximum; extras ignored")
        if note:
            lines.append("Note: " + "; ".join(note) + ".")
        lines.append(f"Student mistake moments found (>=10 win% lost): {self.moments_found}.")
        if self.strengths:
            lines.append("Verified strengths (give this credit explicitly): " + "; ".join(self.strengths) + ".")
        lines.extend(self._slide_signature_lines())
        if not self.patterns:
            lines.append("No recurring patterns found (a pattern needs 3+ moments across 2+ games). "
                         "The student's mistakes do not cluster — praise the consistency and "
                         "suggest single-game reviews instead.")
            return "\n".join(lines)
        lines.append("")
        lines.append("RECURRING PATTERNS (deterministic clustering of verified mistakes — "
                     "the counts are facts; never invent additional patterns):")
        total_cost = sum(p.total_drop for p in self.patterns) or 1.0
        rank = {id(p): n for n, p in enumerate(self.patterns, 1)}
        most_frequent = max(self.patterns, key=lambda p: p.count)
        most_damaging = max(self.patterns, key=lambda p: p.total_drop)
        md_avg = most_damaging.total_drop / most_damaging.count if most_damaging.count else 0.0
        lines.append("  TWO WAYS TO READ THESE — how OFTEN a leak happens vs how much it COSTS. "
                     "They can disagree (a rarer habit can bleed more), so present BOTH and let the "
                     "student choose; never imply frequency alone means priority:")
        lines.append(f"    - Most FREQUENT: Pattern {rank[id(most_frequent)]} "
                     f"({most_frequent.count} moments).")
        lines.append(f"    - Most DAMAGING: Pattern {rank[id(most_damaging)]} "
                     f"(~{most_damaging.total_drop / total_cost:.0%} of the win% bled across these "
                     f"patterns; averages {_severity_label(md_avg)} (~{md_avg:.0f} win%) per slip).")
        for n, p in enumerate(self.patterns, 1):
            avg = p.total_drop / p.count if p.count else 0.0
            examples = ", ".join(f"game {m['game_key']}: {m['move_label']}" for m in p.moments[:3])
            lines.append(
                f"[Pattern {n}] {p.title} — appears in {p.games_seen} of {self.games_analyzed} "
                f"games ({p.count} moments). Averages {_severity_label(avg)} (~{avg:.0f} win%) per "
                f"slip; ~{p.total_drop / total_cost:.0%} of the win% bled across these patterns "
                f"(raw cumulative {p.total_drop:.0f} win% — a sum over all {p.count} moments, not "
                f"a percentage). Examples: {examples}.")
            if p.axis == "execution":
                lines.append(f"  Axis: EXECUTION/DISCIPLINE — {p.in_slide_count}/{p.count} of these "
                             "happened DURING a within-game slide, so this is a focus/process leak under "
                             "drift, not a knowledge gap. Coach the meta-skill (re-scan when the position "
                             "turns), not just the motif.")
            elif p.axis == "knowledge":
                lines.append("  Axis: KNOWLEDGE GAP — these occur as isolated mistakes, not during "
                             "slides. Teach the pattern directly.")
            if p.habit:
                lines.append(f"  Habit to build: {p.habit}")
        # Same underlying habit shown in more than one phase (e.g. priority_selection
        # in both opening and middlegame): name it so the student sees one weakness.
        by_obj = {}
        for n, p in enumerate(self.patterns, 1):
            by_obj.setdefault(p.objective, []).append((n, p))
        for obj, group in by_obj.items():
            if len(group) > 1:
                ns = " and ".join(f"Pattern {n}" for n, _ in group)
                tot_m = sum(p.count for _, p in group)
                games = {m["game_key"] for _, p in group for m in p.moments}
                lines.append(f"  SAME HABIT, TWO PHASES: {ns} are the same underlying weakness "
                             f"({obj.replace('_', ' ')}) in different game phases — together {tot_m} "
                             f"moments across {len(games)} games. Frame them as one habit, not two.")
        lines.append("")
        lines.append("TRAINING PLAN PROTOCOL: present the patterns above as a training plan in "
                     "your own words (with the counts and the frequency-vs-impact split — they are "
                     "the evidence). Name BOTH the most frequent and the most damaging leak and let "
                     "the student choose which to train first; do not default to frequency. Do NOT "
                     "begin teaching a position until the student picks a pattern.")
        lines.append("")
        lines.append("TEACHING POSITIONS (deep-verified; teach the chosen pattern's positions ONE "
                     "AT A TIME under the standard game-review protocol):")
        for i, r in enumerate(self.records, 1):
            pat_n = self.rep_pattern_idx[i - 1] + 1 if i - 1 < len(self.rep_pattern_idx) else 1
            src = self.rep_sources[i - 1] if i - 1 < len(self.rep_sources) else {}
            origin = src.get("game_key", "?")
            players = f"{src.get('white', '?')} vs {src.get('black', '?')}"
            site = src.get("site", "") or ""
            link = f", {site}" if "lichess.org" in site else ""
            lines.append(f"-- Pattern {pat_n}, position from game {origin} ({players}{link}) --")
            lines.extend(moment_digest_lines(i, r))
        lines.append("")
        lines.append("Coaching protocol: for each position, set the scene at the FEN BEFORE the move "
                     "(name the game from its Orientation line — the move number, the opponent's last "
                     "move, and the Lichess link if present, so the student can pull the board up), "
                     "ask the Initial question EXACTLY first, use the hint ladder only after an attempt, "
                     "grade answers with evaluate_student_answer, ask the student WHY the move works, then teach using "
                     "the reveal-phase explanation primitives. Log each diagnosed mistake once with "
                     "submit_diagnosis. After finishing a pattern's positions, give a pattern "
                     "synthesis around its habit and offer the next pattern.")
        return "\n".join(lines)


def attach_batch_board_tokens(batch: BatchReview, state, invert: bool = False) -> None:
    """Per-record POV (representatives come from games where the student may
    have played either color)."""
    from app.render import render_board_token
    for r in batch.records:
        board = chess.Board(r.fen_before)
        r.board_token = render_board_token(state, board, r.color or "white", invert)


def run_batch_review(pgn_text: str, engines, config, on_progress=None, on_status=None):
    """The full pipeline. Returns a BatchReview, or {'error': ...}."""
    cache_dir = batches_dir(config)
    scans, skipped, truncated = scan_games(pgn_text, engines, config, cache_dir,
                                           on_progress=on_progress, on_status=on_status)
    if not scans:
        return {"error": "No reviewable games found in the PGN "
                         "(check that player_name matches a header in at least one game)."}
    patterns = aggregate_patterns(scans)
    slide_signature = aggregate_slides(scans)
    moments_found = sum(1 for s in scans for _ in _student_mistakes(s))

    top_n = int(getattr(config, "batch_top_patterns", 3))
    reps_per = int(getattr(config, "batch_reps", 3))
    patterns = patterns[:top_n]

    key_digest = hashlib.sha1("|".join(sorted(s["game_key"] for s in scans)).encode()).hexdigest()[:6]
    batch = BatchReview(
        batch_id=f"{date.today().isoformat()}-{len(scans)}g-{key_digest}",
        created=date.today().isoformat(),
        games_analyzed=len(scans), games_skipped=skipped, truncated=truncated,
        moments_found=moments_found, patterns=patterns,
        strengths=compute_strengths(scans),
        slide_signature=slide_signature,
        white_elo=getattr(config, "user_elo", DEFAULT_USER_ELO),
        black_elo=getattr(config, "user_elo", DEFAULT_USER_ELO),
    )
    headers_by_key = {s["game_key"]: s.get("headers", {}) for s in scans}
    for p_idx, pattern in enumerate(patterns):
        for moment in select_representatives(pattern, reps_per):
            record = MoveRecord(**moment["record"])
            _backfill_maia(record, engines, getattr(config, "user_elo", DEFAULT_USER_ELO))
            pattern.representatives.append(record)
            batch.records.append(record)
            batch.rep_pattern_idx.append(p_idx)
            h = headers_by_key.get(moment["game_key"], {})
            batch.rep_sources.append({"game_key": moment["game_key"],
                                      "white": h.get("White", "?"),
                                      "black": h.get("Black", "?"),
                                      "site": h.get("Site", "")})
    batch.agenda_plies = list(range(len(batch.records)))
    if batch.records:
        if on_status:
            on_status(f"deep-verifying {len(batch.records)} representative moments")
        online = bool(getattr(config, "use_online_databases", True))
        _enrich_agenda(batch, engines, online)
    _save_report(cache_dir, batch)
    return batch


# ---------------------------------------------------------------------------
# Report persistence (reload without rescanning)
# ---------------------------------------------------------------------------

def _report_path(cache_dir: str, batch_id: str) -> str:
    return os.path.join(cache_dir, f"{batch_id}.json")


def _save_report(cache_dir: str, batch: BatchReview) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    data = {
        "batch_id": batch.batch_id, "created": batch.created,
        "games_analyzed": batch.games_analyzed, "games_skipped": batch.games_skipped,
        "truncated": batch.truncated, "moments_found": batch.moments_found,
        "strengths": batch.strengths,
        "slide_signature": batch.slide_signature,
        "rep_pattern_idx": batch.rep_pattern_idx,
        "rep_sources": batch.rep_sources,
        "records": [asdict(r) for r in batch.records],
        "patterns": [{
            "pattern_id": p.pattern_id, "objective": p.objective, "phase": p.phase,
            "title": p.title, "habit": p.habit, "count": p.count,
            "games_seen": p.games_seen, "total_drop": p.total_drop,
            "in_slide_count": p.in_slide_count, "axis": p.axis,
            "moments": [{k: v for k, v in m.items() if k != "record"} for m in p.moments],
        } for p in batch.patterns],
    }
    tmp = _report_path(cache_dir, batch.batch_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _report_path(cache_dir, batch.batch_id))


def _record_from_dict(d: dict) -> MoveRecord:
    # JSON round-trip turns int question-ladder keys into strings.
    r = MoveRecord(**d)
    if r.question_ladder:
        r.question_ladder = {int(k): v for k, v in r.question_ladder.items()}
    return r


def load_latest_batch(config):
    """Most recent saved batch report, reconstructed (board tokens NOT yet
    attached — they are session-scoped)."""
    cache_dir = batches_dir(config)
    if not os.path.isdir(cache_dir):
        return None
    reports = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    if not reports:
        return None
    newest = max(reports, key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)))
    try:
        with open(os.path.join(cache_dir, newest), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    patterns = [BatchPattern(**{**p, "representatives": []}) for p in data.get("patterns", [])]
    batch = BatchReview(
        batch_id=data["batch_id"], created=data.get("created"),
        games_analyzed=data.get("games_analyzed", 0),
        games_skipped=data.get("games_skipped", 0),
        truncated=data.get("truncated", False),
        moments_found=data.get("moments_found", 0),
        patterns=patterns, strengths=data.get("strengths", []),
        slide_signature=data.get("slide_signature"),
        records=[_record_from_dict(d) for d in data.get("records", [])],
        rep_pattern_idx=data.get("rep_pattern_idx", []),
        rep_sources=data.get("rep_sources", []),
        white_elo=getattr(config, "user_elo", DEFAULT_USER_ELO),
        black_elo=getattr(config, "user_elo", DEFAULT_USER_ELO),
    )
    batch.agenda_plies = list(range(len(batch.records)))
    for p_idx, rec in zip(batch.rep_pattern_idx, batch.records):
        if 0 <= p_idx < len(patterns):
            patterns[p_idx].representatives.append(rec)
    return batch


def batch_dossier_summary(batch: BatchReview) -> dict:
    """The distilled facts that enter the long-term dossier."""
    return {
        "batch_id": batch.batch_id,
        "games": batch.games_analyzed,
        "slide_signature": batch.slide_signature,
        "patterns": [{
            "id": p.pattern_id, "title": p.title, "count": p.count,
            "games_seen": p.games_seen, "habit": p.habit, "axis": p.axis,
            "rate": round(p.count / batch.games_analyzed, 2) if batch.games_analyzed else None,
            "representative_positions": [{
                "fen": r.fen_before, "move_label": r.move_label,
                "played": r.san, "better": r.best_move_san,
            } for r in p.representatives[:3]],
        } for p in batch.patterns],
    }
