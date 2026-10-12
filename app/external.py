"""Cached, failure-tolerant access to Lichess and Chess.com public data
(opening explorer, Syzygy tablebase, game and profile exports).

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
LICHESS_USER_GAMES = "https://lichess.org/api/games/user/{username}"
LICHESS_EXPORT_IDS = "https://lichess.org/api/games/export/_ids"
CHESSCOM_STATS = "https://api.chess.com/pub/player/{username}/stats"
CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/{username}/games/archives"
# Lichess rejects anonymous API requests without an identifying User-Agent
# (lichess-org/api#667); Chess.com's public API asks for the same.
USER_AGENT = "chess-coach (+https://github.com/mcmd/chess-coach)"
PGN_EXPORT_PARAMS = {"evals": "true", "clocks": "true", "opening": "true", "tags": "true"}


def _headers(extra: dict = None) -> dict:
    return {"User-Agent": USER_AGENT, **(extra or {})}


def _is_pgn_response(r) -> bool:
    """A redirect to a login or error page arrives as 200 HTML; never parse it as a game."""
    content_type = (r.headers.get("Content-Type") or "").lower()
    return "html" not in content_type and not r.text.lstrip().startswith("<")


def _fetch_json(url: str, params: dict):
    try:
        r = requests.get(url, params=params, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _fetch_text(url: str, params: dict = None, headers: dict = None):
    try:
        r = requests.get(url, params=params or {}, headers=_headers(headers), timeout=20)
        r.raise_for_status()
        return r.text if _is_pgn_response(r) else None
    except Exception:
        return None


def lichess_game_pgn(game_id: str) -> str | None:
    """One Lichess game as PGN with server evals and clocks. Returns "" when Lichess
    has no such game, and None on any network failure or non-PGN response."""
    try:
        r = requests.post(LICHESS_EXPORT_IDS, params=PGN_EXPORT_PARAMS, data=game_id,
                          headers=_headers({"Accept": "application/x-chess-pgn"}), timeout=15)
        r.raise_for_status()
        return r.text if _is_pgn_response(r) else None
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


def lichess_user_games_pgn(username: str, max_games: int = 50) -> str | None:
    """Recent analysed Lichess games as PGN, or None on any network/API failure."""
    if not username:
        return None
    return _fetch_text(
        LICHESS_USER_GAMES.format(username=username),
        {
            "max": max(1, int(max_games or 1)),
            "analysed": "true",
            "evals": "true",
            "clocks": "true",
            "opening": "true",
        },
        {"Accept": "application/x-chess-pgn"},
    )


def chesscom_user_games_pgn(username: str, max_games: int = 50) -> str | None:
    """Recent Chess.com archive games as PGN, or None on any network/API failure."""
    if not username:
        return None
    archives = get_json(CHESSCOM_ARCHIVES.format(username=username), {}) or {}
    urls = list(archives.get("archives") or [])
    if not urls:
        return None
    pgns = []
    remaining = max(1, int(max_games or 1))
    for url in reversed(urls):
        data = get_json(url, {}) or {}
        for game in reversed(data.get("games") or []):
            pgn = game.get("pgn")
            if pgn:
                pgns.append(pgn)
                remaining -= 1
                if remaining <= 0:
                    return "\n\n".join(reversed(pgns))
    return "\n\n".join(reversed(pgns)) if pgns else None


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
