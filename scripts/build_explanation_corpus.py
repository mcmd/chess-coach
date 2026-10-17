"""Collect club games for the D9 labelled explanation set (design/DECISIONS.md, C1-C10).

Finds players in a rating band by walking opponents outward from a seed account, checks them in batches,
then exports their rated rapid/classical games and keeps those where both players are in the band.

Lichess API etiquette (https://lichess.org/page/api-tips): one request at a time, a pause between requests,
and a full minute's wait after any HTTP 429. Uses the optional LICHESS_TOKEN from .env and the app's
identifying User-Agent. Read-only endpoints only.

    .venv/bin/python scripts/build_explanation_corpus.py --out design/testing/explanation-set/corpus
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import chess.pgn
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.external import _headers  # noqa: E402

API = "https://lichess.org/api"
PAUSE_S = 2.0          # between requests
BACKOFF_S = 70.0       # after a 429 (Lichess asks for at least one minute)


class Lichess:
    def __init__(self, log):
        self.log = log
        self.requests = 0

    def _call(self, method: str, url: str, headers: dict = None, **kwargs) -> requests.Response | None:
        for attempt in range(3):
            time.sleep(PAUSE_S)
            self.requests += 1
            response = requests.request(method, url, headers=_headers(url, headers), timeout=60, **kwargs)
            if response.status_code == 429:
                self.log(f"429 from {url}; waiting {BACKOFF_S:.0f}s")
                time.sleep(BACKOFF_S)
                continue
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response
        self.log(f"giving up on {url} after repeated 429s")
        return None

    def games(self, username: str, max_games: int) -> list[dict]:
        response = self._call("GET", f"{API}/games/user/{username}", headers={"Accept": "application/x-ndjson"},
                              params={"max": max_games, "rated": "true", "perfType": "rapid,classical",
                                      "clocks": "true", "evals": "true", "opening": "true", "pgnInJson": "true"})
        if response is None:
            return []
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def users(self, usernames: list[str]) -> list[dict]:
        found = []
        for start in range(0, len(usernames), 300):  # the endpoint takes up to 300 ids
            response = self._call("POST", f"{API}/users", data=",".join(usernames[start:start + 300]),
                                  headers={"Content-Type": "text/plain"})
            found += response.json() if response is not None else []
        return found


def in_band(rating, band) -> bool:
    return rating is not None and band[0] <= rating <= band[1]


def game_fits(game: dict, band, min_initial: int, min_moves: int, exclude: set) -> bool:
    """C2/C3: both players in the band, at least `min_initial` seconds per side, standard chess,
    long enough to have a middlegame, and no excluded account (the owner's games are out under C1 C)."""
    players = game.get("players") or {}
    white, black = players.get("white") or {}, players.get("black") or {}
    names = {(p.get("user") or {}).get("id") for p in (white, black)}
    clock = game.get("clock") or {}
    return (game.get("variant") == "standard" and game.get("rated") is True
            and clock.get("initial", 0) >= min_initial
            and in_band(white.get("rating"), band) and in_band(black.get("rating"), band)
            and not (names & exclude) and None not in names
            and len((game.get("moves") or "").split()) >= 2 * min_moves)


def player_fits(user: dict, band, min_games: int) -> bool:
    rapid = ((user.get("perfs") or {}).get("rapid") or {})
    return (not user.get("disabled") and not user.get("tosViolation") and not rapid.get("prov")
            and in_band(rapid.get("rating"), band) and rapid.get("games", 0) >= min_games)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", default="mcmd")
    parser.add_argument("--band", type=int, nargs=2, default=(1050, 1300))
    parser.add_argument("--min-initial", type=int, default=600, help="seconds per side (C3: 10 minutes)")
    parser.add_argument("--min-moves", type=int, default=20)
    parser.add_argument("--min-player-games", type=int, default=50)
    parser.add_argument("--players", type=int, default=12)
    parser.add_argument("--games-per-player", type=int, default=6)
    parser.add_argument("--scan-games", type=int, default=60, help="games read per account while searching")
    parser.add_argument("--exclude", nargs="*", default=["mcmd"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    load_dotenv()
    band, exclude = tuple(args.band), {name.lower() for name in args.exclude}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "search.log"

    def log(message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    api = Lichess(log)
    frontier, seen, chosen = [args.seed.lower()], set(), []
    while frontier and len(chosen) < args.players:
        account = frontier.pop(0)
        if account in seen:
            continue
        seen.add(account)
        games = api.games(account, args.scan_games)
        opponents = []
        for game in games:
            for side in ("white", "black"):
                player = (game.get("players") or {}).get(side) or {}
                name = ((player.get("user") or {}).get("id") or "").lower()
                if name and name not in seen and name not in exclude and in_band(player.get("rating"), band):
                    opponents.append(name)
        opponents = list(dict.fromkeys(opponents))
        checked = {u["id"]: u for u in api.users(opponents)} if opponents else {}
        fitting = [name for name in opponents if name in checked and player_fits(checked[name], band,
                                                                                 args.min_player_games)]
        log(f"{account}: {len(games)} games read, {len(opponents)} opponents in band, {len(fitting)} fit")
        for name in fitting:
            if name not in chosen and len(chosen) < args.players:
                chosen.append(name)
            frontier.append(name)

    log(f"players: {chosen}")
    selected, ids = [], set()
    for name in chosen:
        kept = 0
        for game in api.games(name, args.scan_games):
            if kept >= args.games_per_player or game["id"] in ids:
                continue
            if game_fits(game, band, args.min_initial, args.min_moves, exclude):
                selected.append(game)
                ids.add(game["id"])
                kept += 1
        log(f"{name}: kept {kept} games")

    with (args.out / "games.pgn").open("w", encoding="utf-8") as handle:
        for game in selected:
            handle.write(game["pgn"].strip() + "\n\n")
    manifest = {"band": band, "min_initial_s": args.min_initial, "min_moves": args.min_moves, "seed": args.seed,
                "excluded": sorted(exclude), "players": chosen, "games": [
                    {"id": g["id"], "url": f"https://lichess.org/{g['id']}", "speed": g.get("speed"),
                     "white": g["players"]["white"].get("rating"), "black": g["players"]["black"].get("rating"),
                     "evals": bool(g.get("analysis"))} for g in selected],
                "requests": api.requests, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"saved {len(selected)} games from {len(chosen)} players in {api.requests} requests")
    for game in selected:  # every saved game must parse
        assert chess.pgn.read_game(io.StringIO(game["pgn"])) is not None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
