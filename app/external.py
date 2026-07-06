"""Cached, failure-tolerant access to Lichess online databases
(opening explorer + Syzygy tablebase).

Design rule: the app is network-ENHANCED, never network-dependent. Every
function returns None on any failure; callers must degrade gracefully to
engine-only behavior. Responses are cached for the session.
"""
from functools import lru_cache

import requests

from app.trace import trace

EXPLORER_MASTERS = "https://explorer.lichess.ovh/masters"
EXPLORER_LICHESS = "https://explorer.lichess.ovh/lichess"
TABLEBASE = "https://tablebase.lichess.ovh/standard"
LICHESS_USER = "https://lichess.org/api/user/{username}"
CHESSCOM_STATS = "https://api.chess.com/pub/player/{username}/stats"


def _fetch_json(url: str, params: dict):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@lru_cache(maxsize=512)
def _get_json_cached(url: str, params_key: tuple):
    trace(f"lichess api fetch: {url.rsplit('/', 1)[-1]} {dict(params_key).get('fen', '')[:30]}")
    return _fetch_json(url, dict(params_key))


def get_json(url: str, params: dict = None):
    """Cached GET. Returns parsed JSON or None on any failure."""
    key = tuple(sorted((params or {}).items()))
    return _get_json_cached(url, key)


def masters_moves(fen: str):
    return get_json(EXPLORER_MASTERS, {"fen": fen})


def lichess_moves(fen: str, ratings: str, speeds: str = "blitz,rapid"):
    return get_json(EXPLORER_LICHESS, {"variant": "standard", "speeds": speeds,
                                       "ratings": ratings, "fen": fen})


def tablebase_probe(fen: str):
    return get_json(TABLEBASE, {"fen": fen})


def lichess_user_profile(username: str):
    if not username:
        return None
    return get_json(LICHESS_USER.format(username=username), {})


def chesscom_player_stats(username: str):
    if not username:
        return None
    return get_json(CHESSCOM_STATS.format(username=username), {})


def clear_cache():
    _get_json_cached.cache_clear()


# --- Tablebase category helpers (pure) ---

def normalize_category(category):
    """Lichess categories (win/cursed-win/maybe-win/draw/...) -> win/draw/loss/None,
    from the perspective of the side to move in the probed position."""
    if not category:
        return None
    if "win" in category:
        return "win"
    if "loss" in category:
        return "loss"
    if category == "draw":
        return "draw"
    return None


def invert_category(category):
    """A move's category is given from the NEXT side to move's perspective;
    invert it to get the mover's outcome."""
    return {"win": "loss", "loss": "win", "draw": "draw"}.get(category)
