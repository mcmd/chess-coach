"""Per-student PGN importer (P1-8).

Local PGNs and remote platform exports share one path: parse games, match the
configured student identity, dedup, reverify through batch review, then update
the dossier and drill banks. Network import remains optional; failures degrade
to a clear ImportResult error without mutating the dossier.
"""
from dataclasses import dataclass
from datetime import date
import io

import chess.pgn

from app import external
from app.batch import batch_dossier_summary, game_key_for, run_batch_review
from app.memory import _atomic_write_json, load_dossier, update_dossier_with_batch
from app.review import configured_player_identities, detect_student_color


@dataclass
class ImportResult:
    games_seen: int = 0
    games_matched: int = 0
    games_new: int = 0
    games_duplicate: int = 0
    games_imported: int = 0
    moments_found: int = 0
    patterns: int = 0
    batch_id: str = None
    message: str = None
    error: str = None


def _read_games(pgn_text: str, max_games: int) -> tuple[list, bool]:
    games = []
    pgn_io = io.StringIO(pgn_text)
    while len(games) < max_games:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            return games, False
        if list(game.mainline_moves()):
            games.append(game)
    return games, chess.pgn.read_game(pgn_io) is not None


def _imported_keys(dossier: dict) -> set:
    return {
        item.get("game_key")
        for item in dossier.get("imported_games", [])
        if item.get("game_key")
    }


def _seed_representative_positions(dossier_file: str, summary: dict) -> int:
    dossier = load_dossier(dossier_file)
    existing = {
        (p.get("fen"), p.get("error_category"))
        for p in dossier.get("recent_positions", [])
    }
    seeded = 0
    for pattern in summary.get("patterns", []):
        pattern_id = pattern.get("id") or ""
        category = pattern_id.split(":", 1)[0] or "unknown"
        for pos in pattern.get("representative_positions", []):
            fen = pos.get("fen")
            key = (fen, category)
            if not fen or key in existing:
                continue
            dossier["recent_positions"].insert(0, {
                "fen": fen,
                "error_category": category,
                "source": "imported_batch_representative",
                "batch_id": summary.get("batch_id"),
                "pattern_id": pattern_id,
                "move_label": pos.get("move_label"),
                "played": pos.get("played"),
                "better": pos.get("better"),
            })
            existing.add(key)
            seeded += 1
    dossier["recent_positions"] = dossier["recent_positions"][:20]
    _atomic_write_json(dossier_file, dossier)
    return seeded


def _record_imported_games(dossier_file: str, keys: list[str], source: str, batch_id: str) -> None:
    if not keys:
        return
    dossier = load_dossier(dossier_file)
    existing = [item for item in dossier.get("imported_games", []) if item.get("game_key") not in keys]
    today = date.today().isoformat()
    entries = [
        {"game_key": key, "source": source, "batch_id": batch_id, "imported_at": today}
        for key in keys
    ]
    dossier["imported_games"] = (entries + existing)[:500]
    _atomic_write_json(dossier_file, dossier)


def import_pgn_text(pgn_text: str, engines, config, *, source: str = "local PGN",
                    on_progress=None, on_status=None) -> ImportResult:
    identities = configured_player_identities(config)
    if not identities:
        return ImportResult(error="No configured player identity. Set player_name or usernames first.")

    max_games = int(getattr(config, "batch_max_games", 50))
    games, truncated = _read_games(pgn_text, max_games)
    dossier = load_dossier(config.dossier_file)
    imported = _imported_keys(dossier)
    new_games, new_keys = [], []
    matched = duplicate = 0
    for game in games:
        headers = dict(game.headers)
        if detect_student_color(headers, config) is None:
            continue
        matched += 1
        ucis = " ".join(m.uci() for m in game.mainline_moves())
        key = game_key_for(headers, ucis)
        if key in imported:
            duplicate += 1
            continue
        new_games.append(game)
        new_keys.append(key)

    result = ImportResult(
        games_seen=len(games),
        games_matched=matched,
        games_new=len(new_games),
        games_duplicate=duplicate,
    )
    if truncated:
        result.message = f"Input was truncated at {max_games} games by batch_max_games."
    if not new_games:
        result.error = "No new matching games to import."
        return result

    filtered_pgn = "\n\n".join(str(game) for game in new_games)
    batch = run_batch_review(filtered_pgn, engines, config,
                             on_progress=on_progress, on_status=on_status)
    if isinstance(batch, dict):
        result.error = batch.get("error") or "Import batch review failed."
        return result

    summary = batch_dossier_summary(batch)
    update_dossier_with_batch(config.dossier_file, summary)
    seeded = _seed_representative_positions(config.dossier_file, summary)
    analyzed_keys = list(getattr(batch, "game_keys", None) or new_keys)
    _record_imported_games(config.dossier_file, analyzed_keys, source, batch.batch_id)
    result.games_imported = batch.games_analyzed
    result.moments_found = batch.moments_found
    result.patterns = len(batch.patterns)
    result.batch_id = batch.batch_id
    suffix = f" Seeded {seeded} drill position(s)." if seeded else ""
    result.message = (result.message or "") + (
        f"Imported {result.games_imported} new game(s), found {result.moments_found} "
        f"mistake moment(s), updated {result.patterns} recurring pattern(s).{suffix}"
    )
    return result


def import_local_pgn_file(path: str, engines, config, *, on_progress=None, on_status=None) -> ImportResult:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            pgn_text = handle.read()
    except OSError as exc:
        return ImportResult(error=f"Could not read PGN file: {exc}")
    return import_pgn_text(pgn_text, engines, config, source=path,
                           on_progress=on_progress, on_status=on_status)


def configured_import_targets(config) -> list[tuple[str, str]]:
    targets = []
    for platform, handles in (getattr(config, "usernames", None) or {}).items():
        key = str(platform or "").casefold()
        if key in ("lichess", "chesscom", "chess.com"):
            for handle in handles or []:
                if handle:
                    targets.append(("chesscom" if key == "chess.com" else key, str(handle)))
    return targets


def should_offer_onboarding_import(config) -> bool:
    if not configured_import_targets(config):
        return False
    dossier = load_dossier(config.dossier_file)
    return not dossier.get("imported_games") and not dossier.get("recurring_patterns")


def _remote_pgn(platform: str, username: str, max_games: int) -> str | None:
    if platform == "lichess":
        return external.lichess_user_games_pgn(username, max_games=max_games)
    if platform == "chesscom":
        return external.chesscom_user_games_pgn(username, max_games=max_games)
    return None


def import_remote_games(platform: str, username: str, engines, config, *, max_games: int = None,
                        on_progress=None, on_status=None) -> ImportResult:
    platform = (platform or "").casefold()
    if platform == "chess.com":
        platform = "chesscom"
    if platform not in ("lichess", "chesscom"):
        return ImportResult(error=f"Unsupported import platform: {platform or 'unknown'}.")
    max_games = int(max_games or getattr(config, "batch_max_games", 50))
    pgn_text = _remote_pgn(platform, username, max_games)
    if not pgn_text:
        return ImportResult(error=f"Could not fetch recent analysed games for {platform}:{username}.")
    return import_pgn_text(
        pgn_text, engines, config, source=f"{platform}:{username}",
        on_progress=on_progress, on_status=on_status,
    )


def import_configured_recent_games(config, engines, *, on_progress=None, on_status=None) -> ImportResult:
    targets = configured_import_targets(config)
    if not targets:
        return ImportResult(error="No supported usernames configured for remote import.")
    errors = []
    for platform, username in targets:
        result = import_remote_games(platform, username, engines, config,
                                     on_progress=on_progress, on_status=on_status)
        if not result.error:
            return result
        errors.append(f"{platform}:{username} -> {result.error}")
    return ImportResult(error="Remote import failed for every configured handle. " + " | ".join(errors))
