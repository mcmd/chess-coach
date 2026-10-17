"""Review the D9 corpus games with the coach's own pipeline and keep the teachable moments (design/DECISIONS.md).

For each game the reviewed player is the corpus player in it (scripts/build_explanation_corpus.py), and the
coaching level is that player's rating in the game, so the same pipeline serves any rating band. Every agenda
moment is saved, marked eligible when its key reason is specific rather than "the engine prefers it", so a
coverage report can show how often the coach picks moments it cannot explain. Whether an eligible reason states
a relationship or consequence (not just a square) is judged and verified when moments are selected. Also
collects C10's draws by rule, classified: whether the draw ended the game, and whether it saved a lost position
(D1) or threw away a supported win (D2).

Resumable: games already in moments.jsonl are skipped. Pausable: while the file named by --pause-file exists,
it waits between games (a live eval needs the CPU).

    .venv/bin/python scripts/extract_explanation_moments.py --corpus design/testing/explanation-set/corpus
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def moment_kind(record) -> str:
    """C4's moment families, from the review's own verified evidence."""
    key = next((r for r in ((record.explanation_primitives or {}).get("reasons") or []) if r.get("key")), {})
    if key.get("kind") == "resource_contrast":
        return "move_order_contrast"
    if record.lesson_objective == "defend_against_direct_threat":
        return "defend_threat"
    if record.recommended_diagnosis in ("hanging_piece",) or record.lesson_objective == "check_piece_safety":
        return "hanging_piece"
    if record.recommended_diagnosis == "bad_trade":
        return "bad_trade"
    if record.recommended_diagnosis in ("ignored_threat", "allows_forcing_sequence"):
        return "ignored_threat"
    return record.recommended_diagnosis or "other"


def moment_packet(game_id: str, player: str, record) -> dict:
    """Every agenda moment, eligible or not: the coverage report needs the excluded ones too (how often the
    coach selects a moment it cannot explain is a separate quality measure from grading accuracy)."""
    primitives = record.explanation_primitives or {}
    reasons = primitives.get("reasons") or []
    key = next((r for r in reasons if r.get("key")), None)
    exclusion = ("no_key_reason" if not key else "engine_preference_only" if key.get("generic") else None)
    return {
        "game_id": game_id, "url": f"https://lichess.org/{game_id}#{record.ply + 1}", "player": player,
        "ply": record.ply, "move_label": record.move_label, "played": record.san, "color": record.color,
        "fen_before": record.fen_before, "best_move": record.best_move_san, "best_line": record.best_line,
        "punishment_line": record.punishment_line, "quality": record.quality,
        "eval_before_cp": record.eval_before_cp, "eval_after_cp": record.eval_after_cp,
        "kind": moment_kind(record), "recommended_diagnosis": record.recommended_diagnosis,
        "lesson_objective": record.lesson_objective, "key_reason": key,
        "other_reasons": [r for r in reasons if not r.get("key")],
        "explanation_primitives": primitives, "resource_refutation": record.resource_refutation,
        "maia_played_prob": record.maia_played_prob, "best_move_human_prob": record.best_move_human_prob,
        # Eligibility here is only "a specific key reason exists"; whether it states a relationship or
        # consequence (not just a square) is judged and verified at selection.
        "eligible": exclusion is None, "exclusion": exclusion,
    }


def repetition_cases(game_id: str, review, result: str) -> list:
    """Draws by rule actually reached in the game. Kept apart: whether the draw ended the recorded game or
    the game went on (a claim was available), and whether the mover was lost (D1: a save) or winning (D2: a
    win thrown away only if the position before had a supported winning alternative)."""
    cases = []
    for record in review.records:
        if not record.draw_rule or record.eval_without_draw_rule_cp is None:
            continue
        sign = 1 if record.color == "white" else -1
        mover_cp = sign * record.eval_without_draw_rule_cp
        before_cp = sign * record.eval_before_cp if record.eval_before_cp is not None else None
        ended = record is review.records[-1] and result == "1/2-1/2"
        if mover_cp <= -200:
            decision = "D1_saved_lost_position"
        elif mover_cp >= 200 and before_cp is not None and before_cp >= 200 and record.best_move_san \
                and record.best_move_san != record.san:
            decision = "D2_threw_away_win"
        else:
            continue
        cases.append({"game_id": game_id, "url": f"https://lichess.org/{game_id}#{record.ply + 1}",
                      "move_label": record.move_label, "played": record.san, "draw_rule": record.draw_rule,
                      "mover_cp_without_rule": mover_cp, "mover_cp_before": before_cp,
                      "best_alternative": record.best_move_san, "ended_game": ended,
                      "decision": decision, "fen_before": record.fen_before})
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path, default=Path("/tmp/cc-issues/pause_extract"))
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    from app.config import load_config
    from app.engines import EngineManager
    from app.review import review_game

    os.chdir(ROOT)
    load_dotenv()
    os.environ.setdefault("GEMINI_API_KEY", "unused-by-moment-extraction")
    saved, sys.argv = sys.argv, [sys.argv[0]]
    try:
        config = load_config()
    finally:
        sys.argv = saved
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    players = [p.lower() for p in manifest["players"]]
    out = args.corpus / "moments.jsonl"
    done = {json.loads(line)["game_id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()

    stream = io.StringIO((args.corpus / "games.pgn").read_text(encoding="utf-8"))
    games = []
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    engines = EngineManager(config)
    engines.start()
    try:
        for index, game in enumerate(games, 1):
            game_id = game.headers.get("Site", "").rstrip("/").split("/")[-1]
            if game_id in done:
                continue
            while args.pause_file.exists():
                time.sleep(30)
            names = {"white": game.headers.get("White", "").lower(), "black": game.headers.get("Black", "").lower()}
            side = min((c for c in names if names[c] in players), key=lambda c: players.index(names[c]))
            config.player_name = game.headers.get("White" if side == "white" else "Black")
            config.user_elo = int(game.headers.get("WhiteElo" if side == "white" else "BlackElo"))
            started = time.monotonic()
            try:
                review = review_game(str(game), engines, config)
                moments = [moment_packet(game_id, config.player_name, r) for r in review.agenda]
                entry = {"game_id": game_id, "player": config.player_name, "side": side, "elo": config.user_elo,
                         "result": game.headers.get("Result"), "opening": game.headers.get("Opening"),
                         "agenda": [r.move_label for r in review.agenda], "moments": moments,
                         "repetitions": repetition_cases(game_id, review, game.headers.get("Result", ""))}
            except Exception as exc:  # recorded, never silently dropped
                entry = {"game_id": game_id, "error": f"{type(exc).__name__}: {exc}"}
            entry["seconds"] = round(time.monotonic() - started, 1)
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
            print(f"[{index}/{len(games)}] {game_id}: {sum(m['eligible'] for m in entry.get('moments', []))}/"
                  f"{len(entry.get('moments', []))} eligible moments, "
                  f"{len(entry.get('repetitions', []))} repetitions, {entry['seconds']}s"
                  + (f" ERROR {entry['error']}" if "error" in entry else ""), flush=True)
    finally:
        engines.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
