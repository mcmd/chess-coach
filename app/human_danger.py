"""Bounded Maia-policy danger estimates.

These helpers intentionally produce a provisional internal signal. They map the
top Maia human moves through local Stockfish quick evals so agenda selection can
prefer positions where students are likely to choose bad moves, without changing
the calibrated PGN-eval severity stream.
"""
import math

import chess


MATE_CP = 100000


def _cp_to_win_pct(cp: float) -> float:
    cp = max(-1500, min(1500, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def _quality_from_drop(drop_pct: float) -> str:
    if drop_pct >= 30:
        return "blunder"
    if drop_pct >= 20:
        return "mistake"
    if drop_pct >= 10:
        return "inaccuracy"
    if drop_pct <= 2:
        return "good"
    return "ok"


def _move_drop_from_local_eval(record, board: chess.Board, move: chess.Move, engines,
                               eval_time: float) -> dict:
    mover_white = record.color == "white"
    after = board.copy(stack=False)
    after.push(move)
    score, _ = engines.quick_eval(after.fen(), time_limit=eval_time)
    after_cp_white = score.white().score(mate_score=MATE_CP)
    before_mover = record.eval_before_cp if mover_white else -record.eval_before_cp
    after_mover = after_cp_white if mover_white else -after_cp_white
    drop_pct = round(max(0.0, _cp_to_win_pct(before_mover) - _cp_to_win_pct(after_mover)), 1)
    return {
        "eval_after_cp_white": after_cp_white,
        "drop_pct_mover": drop_pct,
        "quality": _quality_from_drop(drop_pct),
    }


def empty_human_danger(record, reason: str) -> dict:
    return {
        "schema_version": 1,
        "available": False,
        "reason": reason,
        "elo": None,
        "provisional": True,
        "source": "maia_top_moves_plus_local_quick_eval",
        "covered_probability": 0.0,
        "uncovered_probability": 1.0,
        "quality_mass": {},
        "bad_or_worse_mass": 0.0,
        "mistake_or_blunder_mass": 0.0,
        "moves": [],
    }


def human_danger_mass_for_record(record, engines, *, elo: int = None,
                                 eval_time: float = 0.2) -> dict:
    """Estimate how much Maia move probability lies on bad moves.

    The result is deliberately marked provisional: it uses only stored top Maia
    SAN moves plus local quick evals. It is suitable for internal ranking and
    human proposal review, not exact student-facing claims.
    """
    if not getattr(record, "maia_top", None):
        return empty_human_danger(record, "no_maia_top_moves")
    try:
        board = chess.Board(record.fen_before)
    except ValueError:
        return empty_human_danger(record, "invalid_fen")

    quality_mass = {key: 0.0 for key in ("good", "ok", "inaccuracy", "mistake", "blunder", "unknown")}
    moves = []
    covered = 0.0
    for item in record.maia_top:
        san = item.get("move")
        probability = float(item.get("probability") or 0.0)
        covered += probability
        try:
            move = board.parse_san(san)
            local = _move_drop_from_local_eval(record, board, move, engines, eval_time)
            quality = local["quality"]
            moves.append({
                "move": san,
                "uci": move.uci(),
                "probability": round(probability, 3),
                "quality": quality,
                "drop_pct_mover": local["drop_pct_mover"],
                "eval_after_cp_white": local["eval_after_cp_white"],
                "source": "local_quick_eval",
            })
        except Exception as exc:
            quality = "unknown"
            moves.append({
                "move": san,
                "uci": None,
                "probability": round(probability, 3),
                "quality": quality,
                "error": str(exc),
                "source": "parse_or_eval_failed",
            })
        quality_mass[quality] += probability

    quality_mass = {key: round(value, 3) for key, value in quality_mass.items() if value > 0}
    bad_or_worse = sum(quality_mass.get(key, 0.0) for key in ("inaccuracy", "mistake", "blunder"))
    mistake_or_blunder = sum(quality_mass.get(key, 0.0) for key in ("mistake", "blunder"))
    covered = min(covered, 1.0)
    return {
        "schema_version": 1,
        "available": True,
        "reason": None,
        "elo": elo,
        "provisional": True,
        "source": "maia_top_moves_plus_local_quick_eval",
        "source_detail": (
            "Top Maia moves only; local quick eval is mixed-source and must not be narrated "
            "as calibrated truth."
        ),
        "eval_time": eval_time,
        "covered_probability": round(covered, 3),
        "uncovered_probability": round(max(0.0, 1.0 - covered), 3),
        "quality_mass": quality_mass,
        "bad_or_worse_mass": round(bad_or_worse, 3),
        "mistake_or_blunder_mass": round(mistake_or_blunder, 3),
        "moves": moves,
    }


def build_human_danger_mass(review, engines, *, eval_time: float = 0.2) -> dict:
    student_records = [record for record in review.records
                       if review.student_color is None or record.color == review.student_color]
    out = {}
    for record in student_records:
        if record.drop_pct < 5 and record.ply not in review.agenda_plies:
            continue
        elo = review.white_elo if record.color == "white" else review.black_elo
        out[record.ply] = human_danger_mass_for_record(
            record, engines, elo=elo, eval_time=eval_time)
    return out
