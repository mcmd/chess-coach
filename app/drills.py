"""Dossier drill selection (P1-6).

This is the first deterministic layer above the existing /review drill
mechanism: choose the pattern to practice, avoid stale/invalid positions, vary
instances inside a theme, and record what was presented.
"""
from datetime import date
import math
import re

import chess

from app.memory import load_dossier, _atomic_write_json

DRILL_HISTORY_LIMIT = 50
DRILL_OUTCOME_LIMIT = 100
COMMON_MOVE_THRESHOLD = 0.50
FAIRLY_COMMON_MOVE_THRESHOLD = 0.30
DEFAULT_DIFFICULTY = 5.0
DEFAULT_STABILITY_DAYS = 1.0


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").casefold())


def _valid_position(position: dict) -> bool:
    try:
        board = chess.Board(position.get("fen", ""))
    except (TypeError, ValueError):
        return False
    return not board.is_game_over() and any(board.legal_moves)


def _pattern_matches_category(pattern: dict, category: str) -> bool:
    cat = _norm(category)
    if not cat:
        return False
    pid = _norm(pattern.get("id"))
    title = _norm(pattern.get("title"))
    return bool(cat and (pid.startswith(cat) or cat in title))


def popularity_branch(maia_played_prob) -> str:
    """Calibrated GT-7 language: probability gates commonness, never severity."""
    if maia_played_prob is None:
        return ""
    try:
        prob = float(maia_played_prob)
    except (TypeError, ValueError):
        return ""
    pct = round(prob * 100)
    if prob >= COMMON_MOVE_THRESHOLD:
        return f"Human-commonness note: about {pct}% of peers choose this move too; describe it as common but still costly."
    if prob >= FAIRLY_COMMON_MOVE_THRESHOLD:
        return f"Human-commonness note: about {pct}% of peers choose this move; describe it as a fairly common choice."
    return ""


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def retrievability(state: dict, today: date | None = None) -> float:
    """FSRS-lite decay over a theme bucket; lower means more due."""
    if today is None:
        today = date.today()
    last = _parse_day((state or {}).get("last_reviewed") or (state or {}).get("last_seen"))
    if last is None:
        return 0.0
    try:
        stability = max(0.25, float((state or {}).get("stability_days") or DEFAULT_STABILITY_DAYS))
    except (TypeError, ValueError):
        stability = DEFAULT_STABILITY_DAYS
    elapsed = max(0, (today - last).days)
    return round(math.exp(-elapsed / stability), 3)


def _category_retrievability(dossier: dict, category: str) -> float:
    return retrievability((dossier.get("drill_state") or {}).get(category) or {})


def _last_seen_index(history: list, category: str, fen: str) -> int | None:
    for i, entry in enumerate(history):
        if entry.get("error_category") == category and entry.get("fen") == fen:
            return i
    return None


def _choose_varied(candidates: list, history: list) -> dict:
    category = candidates[0].get("error_category")
    seen = {h.get("fen") for h in history if h.get("error_category") == category}
    for pos in candidates:
        if pos.get("fen") not in seen:
            return pos
    return max(candidates, key=lambda pos: _last_seen_index(history, category, pos.get("fen")) or -1)


def select_dossier_drill(dossier: dict) -> dict | None:
    positions = [dict(p) for p in dossier.get("recent_positions", []) if _valid_position(p)]
    if not positions:
        return None
    by_category = {}
    for pos in positions:
        category = pos.get("error_category") or "unknown"
        by_category.setdefault(category, []).append(pos)

    history = list(dossier.get("drill_history") or [])
    pattern_options = []
    for pattern in dossier.get("recurring_patterns", []):
        if pattern.get("status") not in ("active", "improving"):
            continue
        for category, candidates in by_category.items():
            if _pattern_matches_category(pattern, category):
                pattern_options.append((pattern, category, candidates))
    pattern_options.sort(
        key=lambda item: (
            item[0].get("status") != "active",
            _category_retrievability(dossier, item[1]),
            -(item[0].get("evidence_count") or 0),
        )
    )
    if pattern_options:
        pattern, category, candidates = pattern_options[0]
        pick = _choose_varied(candidates, history)
        return {
            "fen": pick["fen"],
            "error_category": category,
            "error_typicality": pick.get("error_typicality"),
            "maia_played_prob": pick.get("maia_played_prob"),
            "pattern_id": pattern.get("id"),
            "pattern_title": pattern.get("title"),
            "source": "recurring_pattern",
            "priority_reason": f"recurring pattern: {pattern.get('title') or pattern.get('id')}",
            "popularity_note": popularity_branch(pick.get("maia_played_prob")),
            "retrievability": _category_retrievability(dossier, category),
        }

    frequencies = dossier.get("error_frequencies") or {}
    ranked_categories = sorted(
        by_category,
        key=lambda c: (_category_retrievability(dossier, c), -int(frequencies.get(c, 0))),
    )
    category = ranked_categories[0]
    pick = _choose_varied(by_category[category], history)
    return {
        "fen": pick["fen"],
        "error_category": category,
        "error_typicality": pick.get("error_typicality"),
        "maia_played_prob": pick.get("maia_played_prob"),
        "pattern_id": None,
        "pattern_title": None,
        "source": "recent_position",
        "priority_reason": "recent diagnosed position",
        "popularity_note": popularity_branch(pick.get("maia_played_prob")),
        "retrievability": _category_retrievability(dossier, category),
    }


def record_drill_presented(dossier_file: str, drill: dict) -> None:
    if not dossier_file or not drill:
        return
    dossier = load_dossier(dossier_file)
    category = drill.get("error_category") or "unknown"
    entry = {
        "date": date.today().isoformat(),
        "fen": drill.get("fen"),
        "error_category": category,
        "pattern_id": drill.get("pattern_id"),
        "source": drill.get("source"),
    }
    if not entry["fen"]:
        return
    history = [entry] + list(dossier.get("drill_history") or [])
    dossier["drill_history"] = history[:DRILL_HISTORY_LIMIT]
    state = dict(dossier.get("drill_state") or {})
    current = dict(state.get(category) or {})
    current["seen_count"] = int(current.get("seen_count") or 0) + 1
    current["last_seen"] = entry["date"]
    current["last_fen"] = entry["fen"]
    if drill.get("pattern_id"):
        current["last_pattern_id"] = drill["pattern_id"]
    state[category] = current
    dossier["drill_state"] = state
    _atomic_write_json(dossier_file, dossier)


def _outcome_from_result(result: dict) -> dict | None:
    if not result or not result.get("reveal_allowed"):
        return None
    verdict = result.get("verdict")
    success = verdict in ("best", "excellent_alternative")
    outcome = {
        "verdict": verdict,
        "success": success,
        "attempts": result.get("attempts_so_far"),
    }
    for key in ("student_move", "win_pct_lost_vs_best", "rank_among_engine_top5"):
        if result.get(key) is not None:
            outcome[key] = result[key]
    return outcome


def _updated_memory_state(current: dict, outcome: dict, today: date) -> dict:
    state = dict(current or {})
    try:
        difficulty = float(state.get("difficulty") or DEFAULT_DIFFICULTY)
    except (TypeError, ValueError):
        difficulty = DEFAULT_DIFFICULTY
    try:
        stability = float(state.get("stability_days") or DEFAULT_STABILITY_DAYS)
    except (TypeError, ValueError):
        stability = DEFAULT_STABILITY_DAYS

    if outcome["success"]:
        difficulty = max(1.0, difficulty - 0.2)
        stability = max(1.0, stability * 2.0)
        state["success_count"] = int(state.get("success_count") or 0) + 1
        state["retrievability"] = 1.0
    else:
        difficulty = min(10.0, difficulty + 0.8)
        stability = max(0.5, stability * 0.5)
        state["lapse_count"] = int(state.get("lapse_count") or 0) + 1
        state["retrievability"] = 0.0

    state["difficulty"] = round(difficulty, 2)
    state["stability_days"] = round(stability, 2)
    state["last_reviewed"] = today.isoformat()
    state["last_result"] = "success" if outcome["success"] else "lapse"
    state["due_after_days"] = max(1, math.ceil(stability))
    return state


def record_drill_outcome(dossier_file: str, drill: dict, result: dict) -> dict | None:
    outcome = _outcome_from_result(result)
    if not dossier_file or not drill or outcome is None:
        return None
    dossier = load_dossier(dossier_file)
    today = date.today()
    category = drill.get("error_category") or "unknown"
    entry = {
        "date": today.isoformat(),
        "fen": drill.get("fen"),
        "error_category": category,
        "pattern_id": drill.get("pattern_id"),
        "source": drill.get("source"),
        **outcome,
    }
    if not entry["fen"]:
        return None

    outcomes = [entry] + list(dossier.get("drill_outcomes") or [])
    dossier["drill_outcomes"] = outcomes[:DRILL_OUTCOME_LIMIT]

    history = list(dossier.get("drill_history") or [])
    for item in history:
        if item.get("fen") == entry["fen"] and item.get("error_category") == category and "outcome" not in item:
            item["outcome"] = "success" if outcome["success"] else "lapse"
            item["verdict"] = outcome.get("verdict")
            item["attempts"] = outcome.get("attempts")
            break
    dossier["drill_history"] = history[:DRILL_HISTORY_LIMIT]

    state = dict(dossier.get("drill_state") or {})
    current = dict(state.get(category) or {})
    if drill.get("pattern_id"):
        current["last_pattern_id"] = drill["pattern_id"]
    current["last_fen"] = entry["fen"]
    state[category] = _updated_memory_state(current, outcome, today)
    dossier["drill_state"] = state
    _atomic_write_json(dossier_file, dossier)
    return entry
