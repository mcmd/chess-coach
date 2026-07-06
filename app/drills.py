"""Dossier drill selection (P1-6).

This is the first deterministic layer above the existing /review drill
mechanism: choose the pattern to practice, avoid stale/invalid positions, vary
instances inside a theme, and record what was presented.
"""
from datetime import date
import re

import chess

from app.memory import load_dossier, _atomic_write_json


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
    if prob >= 0.50:
        return f"Human-commonness note: about {pct}% of peers choose this move too; describe it as common but still costly."
    if prob >= 0.30:
        return f"Human-commonness note: about {pct}% of peers choose this move; describe it as a fairly common choice."
    return ""


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
    patterns = sorted(
        [p for p in dossier.get("recurring_patterns", []) if p.get("status") in ("active", "improving")],
        key=lambda p: (p.get("status") != "active", -(p.get("evidence_count") or 0)),
    )
    for pattern in patterns:
        for category, candidates in by_category.items():
            if _pattern_matches_category(pattern, category):
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
                }

    frequencies = dossier.get("error_frequencies") or {}
    ranked_categories = sorted(by_category, key=lambda c: -int(frequencies.get(c, 0)))
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
    dossier["drill_history"] = history[:50]
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
