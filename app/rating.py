"""Optional current-rating enrichment for reviewed games (P0-4d).

Network data is advisory only: conversation_elo remains config.user_elo unless
the user explicitly changes it. Failures return None so reviews degrade silently.
"""
from app import external
from app.config import DEFAULT_USER_ELO
from app.review import configured_player_identities, _normalize_player_identity


OFFER_THRESHOLD = 50


def platform_from_headers(headers: dict) -> str | None:
    site = (headers or {}).get("Site", "") or ""
    site_l = site.lower()
    if "lichess.org" in site_l:
        return "lichess"
    if "chess.com" in site_l:
        return "chesscom"
    return None


def perf_from_time_control(headers: dict) -> str | None:
    tc = ((headers or {}).get("TimeControl") or "").strip()
    if not tc or tc in {"-", "?"}:
        return None
    if "/" in tc:
        return "correspondence"
    try:
        base_s, inc_s = tc.split("+", 1) if "+" in tc else (tc, "0")
        base = int(base_s)
        inc = int(inc_s)
    except ValueError:
        return None
    estimated = base + 40 * inc
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def _configured_handles_for_platform(config, platform: str) -> set[str]:
    handles = set()
    usernames = getattr(config, "usernames", None) or {}
    raw = usernames.get(platform) if isinstance(usernames, dict) else None
    if isinstance(raw, str):
        raw = [raw]
    for handle in raw or []:
        norm = _normalize_player_identity(handle)
        if norm:
            handles.add(norm)
    return handles


def handle_for_platform(headers: dict, student_color: str, config, platform: str) -> str | None:
    if student_color not in ("white", "black"):
        return None
    header_name = (headers or {}).get("White" if student_color == "white" else "Black")
    if not header_name:
        return None
    header_norm = _normalize_player_identity(header_name)
    platform_handles = _configured_handles_for_platform(config, platform)
    if platform_handles and header_norm not in platform_handles:
        return None
    if not platform_handles and header_norm not in configured_player_identities(config):
        return None
    return str(header_name).strip()


def _lichess_rating(username: str, perf: str) -> int | None:
    profile = external.lichess_user_profile(username)
    try:
        rating = profile["perfs"][perf]["rating"]
        return int(rating)
    except (TypeError, KeyError, ValueError):
        return None


def _chesscom_rating(username: str, perf: str) -> int | None:
    stats = external.chesscom_player_stats(username)
    key = {
        "bullet": "chess_bullet",
        "blitz": "chess_blitz",
        "rapid": "chess_rapid",
        "classical": "chess_rapid",
        "correspondence": "chess_daily",
    }.get(perf)
    if not key:
        return None
    try:
        return int(stats[key]["last"]["rating"])
    except (TypeError, KeyError, ValueError):
        return None


def current_rating_enrichment(review, config) -> dict | None:
    """Return an advisory current-rating offer, or None when unavailable/irrelevant."""
    if not getattr(config, "use_online_databases", True):
        return None
    platform = platform_from_headers(getattr(review, "headers", {}) or {})
    if platform not in {"lichess", "chesscom"}:
        return None
    perf = perf_from_time_control(review.headers)
    if not perf:
        return None
    username = handle_for_platform(review.headers, getattr(review, "student_color", None),
                                   config, platform)
    if not username:
        return None

    if platform == "lichess":
        rating = _lichess_rating(username, perf)
    else:
        rating = _chesscom_rating(username, perf)
    if rating is None:
        return None

    configured = int(getattr(config, "user_elo", DEFAULT_USER_ELO) or DEFAULT_USER_ELO)
    game_elo = (getattr(review, "white_elo", None)
                if getattr(review, "student_color", None) == "white"
                else getattr(review, "black_elo", None))
    delta = rating - configured
    offer_update = abs(delta) > OFFER_THRESHOLD
    return {
        "platform": platform,
        "username": username,
        "perf": perf,
        "current_rating": rating,
        "configured_elo": configured,
        "delta": delta,
        "game_elo": game_elo,
        "source": "api",
        "offer_update": offer_update,
    }


def rating_offer_note(enrichment: dict | None) -> str:
    if not enrichment or not enrichment.get("offer_update"):
        return ""
    platform = "Lichess" if enrichment.get("platform") == "lichess" else "Chess.com"
    delta = enrichment.get("delta", 0)
    direction = "above" if delta > 0 else "below"
    return (
        "\n\n[Backend rating enrichment - P0-4d: "
        f"{platform} reports current {enrichment.get('perf')} rating "
        f"~{enrichment.get('current_rating')} for {enrichment.get('username')}, which is "
        f"{abs(delta)} points {direction} the configured coaching level "
        f"~{enrichment.get('configured_elo')}. Do NOT change conversation_elo silently. "
        "Briefly offer to use the current platform rating as the coaching level if the student wants; "
        f"until they explicitly change it, keep coaching pitched at ~{enrichment.get('configured_elo')}. "
        "Move-commonness for this reviewed game still uses the PGN game-time Elo.]"
    )
