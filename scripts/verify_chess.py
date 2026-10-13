#!/usr/bin/env python
"""Check a chess claim with the local engines instead of calculating it by hand.

Uses the app's own EngineManager: Stockfish from config.json and the Maia3 human-move
model. Prints JSON. Examples:

    .venv/bin/python scripts/verify_chess.py --fen "<FEN>"                       # best lines
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --move Kh2 --move Kh1  # compare moves
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --line "Bxd2 Qxd2 Qxa1+"
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --elo 1200            # what humans play

Scores are in pawns from White's point of view (mate as "M3" / "-M3"). "deep" is a full
search to --depth (default 30) from a cleared hash with no practical time limit; each result
reports the depth, selective depth, nodes and seconds actually searched, and "complete": false
if the search stopped before --depth. Shallow searches miss refutations (31. Kh2's only appears
at depth 16+), so do not lower --depth to save time. "practical" is a depth-10 search, roughly
what a human calculates: use it only to say what a move looks like at the board, never as the
verdict.
"""
import argparse
import json
import os
import sys
import tempfile

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig  # noqa: E402
from app.engines import EngineManager  # noqa: E402


def _score(pov_score) -> str:
    white = pov_score.white()
    if white.is_mate():
        return f"M{white.mate()}" if white.mate() > 0 else f"-M{-white.mate()}"
    return f"{white.score() / 100:+.2f}"


def _engines(config_path: str) -> EngineManager:
    with open(config_path) as f:
        raw = json.load(f)
    tmp = tempfile.mkdtemp(prefix="verify_chess_")
    keys = ("maia3_deep_model", "maia3_scan_model", "maia3_cache_dir", "maia3_device", "maia3_offline")
    config = AppConfig(
        user_elo=raw.get("user_elo", 1200), language="English", stockfish_path=raw["stockfish_path"],
        gemini_model="unused", api_key="unused", history_file=os.path.join(tmp, "history.json"),
        dossier_file=os.path.join(tmp, "dossier.json"), cli_history_file=os.path.join(tmp, "cli.txt"),
        lesson_file=os.path.join(tmp, "lesson.json"), use_online_databases=False,
        **{k: raw[k] for k in keys if k in raw},
    )
    engines = EngineManager(config)
    engines.start()
    return engines


DEEP_TIME_CAP_S = 600.0   # a safety net only; --depth decides when the search stops


def _lines(engines, board, depth, multipv):
    infos = engines.top_lines(board, multipv=multipv, depth=depth, time_cap=DEEP_TIME_CAP_S,
                              fresh_hash=True)
    return [{"score": _score(info["score"]), "line": board.variation_san(info.get("pv", [])[:12]),
             "depth": info.get("depth"), "seldepth": info.get("seldepth"), "nodes": info.get("nodes"),
             "seconds": round(info.get("time", 0.0), 1), "complete": (info.get("depth") or 0) >= depth}
            for info in infos if info.get("pv")]


def _evaluate(engines, board, depth):
    """Practical and deep evaluation of `board`, plus the deep line."""
    deep = _lines(engines, board, depth, 1) if not board.is_game_over() else []
    result = {"practical": _score(engines.practical_eval(board)) if not board.is_game_over() else None,
              "deep": deep[0]["score"] if deep else None, "deep_line": deep[0]["line"] if deep else None,
              "deep_search": {k: deep[0][k] for k in ("depth", "seldepth", "nodes", "seconds", "complete")}
              if deep else None}
    if board.is_checkmate():
        result["checkmate"] = True
    return result


def verify(engines, fen, moves=(), line=None, elo=None, depth=30, multipv=3):
    board = chess.Board(fen)
    out = {"fen": fen, "side_to_move": "white" if board.turn else "black", "in_check": board.is_check(),
           "depth": depth, "best_lines": _lines(engines, board, depth, multipv)}
    human = None
    if elo:
        pred = engines.human_predict(fen, elo_self=elo, elo_oppo=elo)
        if pred is None:
            out["maia"] = {"error": engines.maia_error or "no prediction"}
        else:
            ranked = sorted(pred.raw_probs.items(), key=lambda kv: kv[1], reverse=True)
            human = {board.san(chess.Move.from_uci(u)): p for u, p in ranked}
            out["maia"] = {"model": pred.model_name, "elo": elo,
                           "top_moves": [{"move": san, "probability": round(p, 3)}
                                         for san, p in list(human.items())[:5]],
                           "wdl_side_to_move": [round(x, 3) if x is not None else None for x in
                                                (pred.win_probability, pred.draw_probability,
                                                 pred.loss_probability)]}
    if moves:
        out["moves"] = []
        for san in moves:
            entry = {"move": san}
            try:
                move = board.parse_san(san)
            except ValueError:
                entry["legal"] = False
                out["moves"].append(entry)
                continue
            entry.update(legal=True, san=board.san(move), gives_check=board.gives_check(move))
            if human is not None:
                entry["maia_probability"] = round(human.get(board.san(move), 0.0), 3)
            after = board.copy(stack=False)
            after.push(move)
            entry.update(_evaluate(engines, after, depth))
            out["moves"].append(entry)
    if line:
        walk, plies = board.copy(stack=False), []
        for san in line.split():
            san = san.rstrip("!?")
            if san[0].isdigit():          # move numbers such as "16." or "16..."
                continue
            try:
                move = walk.parse_san(san)
            except ValueError:
                plies.append({"move": san, "legal": False})
                break
            plies.append({"move": walk.san(move), "legal": True, "gives_check": walk.gives_check(move),
                          "is_capture": walk.is_capture(move)})
            walk.push(move)
        out["line"] = {"plies": plies, "all_legal": all(p["legal"] for p in plies),
                       "final_fen": walk.fen(), **_evaluate(engines, walk, depth)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fen", required=True)
    parser.add_argument("--move", action="append", default=[], help="candidate move (SAN), repeatable")
    parser.add_argument("--line", help="SAN sequence played from --fen, e.g. 'Bxd2 Qxd2 Qxa1+'")
    parser.add_argument("--elo", type=int, help="also ask Maia3 what players of this rating play")
    parser.add_argument("--depth", type=int, default=30,
                        help="search depth; keep >= 30 for verdicts (shallow searches miss refutations)")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"))
    args = parser.parse_args(argv)
    try:
        chess.Board(args.fen)
    except ValueError as exc:
        parser.error(f"invalid FEN: {exc}")
    engines = _engines(args.config)
    try:
        print(json.dumps(verify(engines, args.fen, args.move, args.line, args.elo, args.depth,
                                args.multipv), indent=1))
    finally:
        engines.quit()


if __name__ == "__main__":
    main()
