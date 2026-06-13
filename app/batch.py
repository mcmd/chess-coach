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
from app.review import (GameReview, MoveRecord, review_game, _enrich_agenda,
                        detect_student_color, moment_digest_lines)

# Bump when MoveRecord scan fields or classification change incompatibly.
SCAN_VERSION = 1

# A cluster is only a "pattern" with enough evidence to be a fact, not noise.
MIN_PATTERN_MOMENTS = 3
MIN_PATTERN_GAMES = 2

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
    for i, game in enumerate(games, 1):
        headers = dict(game.headers)
        student_color = detect_student_color(headers, config)
        if getattr(config, "player_name", None) and student_color is None:
            skipped += 1
            if on_status:
                on_status(f"game {i}/{len(games)}: skipped (student not in headers)")
            continue
        ucis = " ".join(m.uci() for m in game.mainline_moves())
        key = game_key_for(headers, ucis)
        cached = _load_cached_scan(cache_dir, key)
        if cached is not None:
            scans.append(cached)
            if on_status:
                on_status(f"game {i}/{len(games)}: cached ({key})")
            continue
        if on_status:
            on_status(f"game {i}/{len(games)}: scanning ({key})")
        review = review_game(str(game), engines, config, on_progress=on_progress,
                             eval_time=scan_eval_time, with_maia=False, enrich=False)
        if review is None:
            skipped += 1
            continue
        scan = {
            "game_key": key,
            "headers": headers,
            "student_color": review.student_color,
            "scan_version": SCAN_VERSION,
            "scanned_with": {"eval_time": scan_eval_time, "with_maia": False},
            "records": [asdict(r) for r in review.records],
        }
        _save_scan(cache_dir, key, scan)
        scans.append(scan)
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
    moments: list            # all member moments: {game_key, move_label, san, drop_pct, fen_before}
    representatives: list = field(default_factory=list)  # enriched MoveRecords (deep pass)


def _student_mistakes(scan: dict):
    color = scan.get("student_color")
    for d in scan.get("records", []):
        if color and d.get("color") != color:
            continue
        if (d.get("drop_pct") or 0) >= 10:
            yield d


def aggregate_patterns(scans: list) -> list:
    """Group student mistakes by (light objective, phase); keep clusters with
    enough cross-game evidence; rank by count then total win% lost."""
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
        patterns.append(BatchPattern(
            pattern_id=f"{objective}:{phase}", objective=objective, phase=phase,
            title=title, habit=PATTERN_HABITS.get(objective, ""),
            count=len(entry["moments"]), games_seen=len(entry["games"]),
            total_drop=round(entry["drop"], 1),
            moments=sorted(entry["moments"], key=lambda m: -m["drop_pct"]),
        ))
    patterns.sort(key=lambda p: (-p.count, -p.total_drop))
    return patterns


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
    student_color: str = None       # mixed across games; per-record color is authoritative
    white_elo: int = None           # student elo (used by enrichment helpers)
    black_elo: int = None
    game = None                     # no single game: the timeline is not loaded in batch mode

    def record_at(self, ply: int):
        return self.records[ply] if 0 <= ply < len(self.records) else None

    @property
    def agenda(self):
        return [self.records[p] for p in self.agenda_plies]

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
        if not self.patterns:
            lines.append("No recurring patterns found (a pattern needs 3+ moments across 2+ games). "
                         "The student's mistakes do not cluster — praise the consistency and "
                         "suggest single-game reviews instead.")
            return "\n".join(lines)
        lines.append("")
        lines.append("RECURRING PATTERNS (deterministic clustering of verified mistakes — "
                     "the counts are facts; never invent additional patterns):")
        for n, p in enumerate(self.patterns, 1):
            examples = ", ".join(f"game {m['game_key']}: {m['move_label']}" for m in p.moments[:3])
            lines.append(f"[Pattern {n}] {p.title} — {p.count} moments across {p.games_seen} games "
                         f"({p.total_drop:.0f} win% lost in total). Examples: {examples}.")
            if p.habit:
                lines.append(f"  Habit to build: {p.habit}")
        lines.append("")
        lines.append("TRAINING PLAN PROTOCOL: present the patterns above as a training plan in "
                     "your own words (with the counts — they are the evidence). Recommend starting "
                     "with Pattern 1, the largest verified leak, but let the student choose. Do NOT "
                     "begin teaching a position until the student picks a pattern.")
        lines.append("")
        lines.append("TEACHING POSITIONS (deep-verified; teach the chosen pattern's positions ONE "
                     "AT A TIME under the standard game-review protocol):")
        for i, r in enumerate(self.records, 1):
            pat_n = self.rep_pattern_idx[i - 1] + 1 if i - 1 < len(self.rep_pattern_idx) else 1
            src = self.rep_sources[i - 1] if i - 1 < len(self.rep_sources) else {}
            origin = src.get("game_key", "?")
            players = f"{src.get('white', '?')} vs {src.get('black', '?')}"
            lines.append(f"-- Pattern {pat_n}, position from game {origin} ({players}) --")
            lines.extend(moment_digest_lines(i, r))
        lines.append("")
        lines.append("Coaching protocol: for each position, set the scene at the FEN BEFORE the move, "
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
        white_elo=getattr(config, "user_elo", 1200),
        black_elo=getattr(config, "user_elo", 1200),
    )
    headers_by_key = {s["game_key"]: s.get("headers", {}) for s in scans}
    for p_idx, pattern in enumerate(patterns):
        for moment in select_representatives(pattern, reps_per):
            record = MoveRecord(**moment["record"])
            _backfill_maia(record, engines, getattr(config, "user_elo", 1200))
            pattern.representatives.append(record)
            batch.records.append(record)
            batch.rep_pattern_idx.append(p_idx)
            h = headers_by_key.get(moment["game_key"], {})
            batch.rep_sources.append({"game_key": moment["game_key"],
                                      "white": h.get("White", "?"),
                                      "black": h.get("Black", "?")})
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
        "rep_pattern_idx": batch.rep_pattern_idx,
        "rep_sources": batch.rep_sources,
        "records": [asdict(r) for r in batch.records],
        "patterns": [{
            "pattern_id": p.pattern_id, "objective": p.objective, "phase": p.phase,
            "title": p.title, "habit": p.habit, "count": p.count,
            "games_seen": p.games_seen, "total_drop": p.total_drop,
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
        records=[_record_from_dict(d) for d in data.get("records", [])],
        rep_pattern_idx=data.get("rep_pattern_idx", []),
        rep_sources=data.get("rep_sources", []),
        white_elo=getattr(config, "user_elo", 1200),
        black_elo=getattr(config, "user_elo", 1200),
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
        "patterns": [{
            "id": p.pattern_id, "title": p.title, "count": p.count,
            "games_seen": p.games_seen, "habit": p.habit,
            "rate": round(p.count / batch.games_analyzed, 2) if batch.games_analyzed else None,
            "representative_positions": [{
                "fen": r.fen_before, "move_label": r.move_label,
                "played": r.san, "better": r.best_move_san,
            } for r in p.representatives[:3]],
        } for p in batch.patterns],
    }
