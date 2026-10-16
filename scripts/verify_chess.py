#!/usr/bin/env python
"""Check a chess claim with the local engines instead of calculating it by hand.

Uses the app's own EngineManager: Stockfish from config.json and the Maia3 human-move
model. Prints JSON. Examples:

    .venv/bin/python scripts/verify_chess.py --fen "<FEN>"                       # best lines
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --move Kh2 --move Kh1  # compare moves
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --line "Bxd2 Qxd2 Qxa1+"
    .venv/bin/python scripts/verify_chess.py --fen "<FEN>" --elo 1200            # what humans play
    .venv/bin/python scripts/verify_chess.py --pgn game.pgn --before "31." --move Kh1 --move Kh2

Game history matters: a bare FEN cannot show a repetition. With --pgn and --before ("31." for
White's 31st move, "30..." for Black's), the position is rebuilt from the game's moves and every
search carries that history, so a candidate that repeats the position a third time scores as the
draw it is. "history" in the output says whether the history was available.

Scores are in pawns from White's point of view (mate as "M3" / "-M3"). "deep" is a full
search to --depth (default 30) from a cleared hash with no practical time limit; each result
reports the depth, selective depth, nodes and seconds actually searched, and "complete": false
if the search stopped before --depth. Shallow searches miss refutations (31. Kh2's only appears
at depth 16+), so do not lower --depth to save time. A candidate move's "deep" comes from a search
of the position before it restricted to that move, and its "deep_line" starts with the move.
"practical" is a depth-10 search, roughly what a human calculates: use it only to say what a move
looks like at the board, never as the verdict. "rules" reports exact draw-rule facts (threefold,
fivefold, fifty-move) for the position reached; these are facts, not engine estimates.
"""
import argparse
import io
import json
import os
import re
import sys
import tempfile

import chess
import chess.engine
import chess.pgn

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


def board_from_pgn(pgn_text: str, before: str) -> chess.Board:
    """The game's position before `before` ("31." = White's 31st move, "30..." = Black's 30th), with the
    game's moves on its stack (from the PGN's own start position, so setup games work)."""
    match = re.fullmatch(r"\s*(\d+)\s*(\.\.\.|\.)\s*", before or "")
    if not match:
        raise ValueError(f"--before must look like '31.' (White) or '30...' (Black), got {before!r}")
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("no game in the PGN")
    board = game.board()
    target = (int(match.group(1)), chess.WHITE if match.group(2) == "." else chess.BLACK)
    for move in game.mainline_moves():
        if (board.fullmove_number, board.turn) == target:
            return board
        board.push(move)
    if (board.fullmove_number, board.turn) == target:
        return board
    raise ValueError(f"the game has no move {before}")


def rule_facts(board: chess.Board) -> dict:
    """Exact draw-rule facts for the position reached, from its move stack (a bare FEN has none)."""
    return {"threefold_repetition": board.is_repetition(3),
            "fivefold_repetition": board.is_fivefold_repetition(),
            "fifty_move_claimable": board.halfmove_clock >= 100,
            "checkmate": board.is_checkmate(), "stalemate": board.is_stalemate()}


def _lines(engines, board, depth, multipv, root_moves=None):
    infos = engines.top_lines(board, multipv=multipv, depth=depth, time_cap=DEEP_TIME_CAP_S,
                              fresh_hash=True, root_moves=root_moves)
    return [{"score": _score(info["score"]), "line": board.variation_san(info.get("pv", [])[:12]),
             "depth": info.get("depth"), "seldepth": info.get("seldepth"), "nodes": info.get("nodes"),
             "seconds": round(info.get("time", 0.0), 1), "complete": (info.get("depth") or 0) >= depth}
            for info in infos if info.get("pv")]


def _summary(lines, board) -> dict:
    deep = lines[0] if lines else None
    return {"deep": deep["score"] if deep else None, "deep_line": deep["line"] if deep else None,
            "deep_search": {k: deep[k] for k in ("depth", "seldepth", "nodes", "seconds", "complete")}
            if deep else None}


def verify(engines, board: chess.Board, moves=(), line=None, elo=None, depth=30, multipv=3,
           history="unavailable"):
    """`board` carries the game's moves on its stack when known; `history` says which."""
    fen = board.fen()
    out = {"fen": fen, "side_to_move": "white" if board.turn else "black", "in_check": board.is_check(),
           "depth": depth, "history": {"status": history, "plies": len(board.move_stack)},
           "best_lines": _lines(engines, board, depth, multipv)}
    human = None
    if elo:
        stack = [m.uci() for m in board.move_stack] or None
        pred = engines.human_predict(fen, elo_self=elo, elo_oppo=elo, history_uci=stack)
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
            after = board.copy()
            after.push(move)
            entry["rules"] = rule_facts(after)
            entry["practical"] = _score(engines.practical_eval(after)) if not after.is_game_over() else None
            entry.update(_summary(_lines(engines, board, depth, 1, root_moves=[move]), board))
            if after.is_checkmate():
                entry["checkmate"] = True
            out["moves"].append(entry)
    if line:
        walk, plies = board.copy(), []
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
        game_over = walk.is_game_over()
        out["line"] = {"plies": plies, "all_legal": all(p["legal"] for p in plies),
                       "final_fen": walk.fen(), "rules": rule_facts(walk),
                       "practical": _score(engines.practical_eval(walk)) if not game_over else None,
                       **(_summary(_lines(engines, walk, depth, 1), walk) if not game_over else
                          {"deep": None, "deep_line": None, "deep_search": None}),
                       **({"checkmate": True} if walk.is_checkmate() else {})}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fen", help="the position (no game history: repetitions are invisible)")
    parser.add_argument("--pgn", help="a PGN file: use the game's position before --before, with its history")
    parser.add_argument("--before", help="with --pgn: '31.' (White's 31st move) or '30...' (Black's 30th)")
    parser.add_argument("--move", action="append", default=[], help="candidate move (SAN), repeatable")
    parser.add_argument("--line", help="SAN sequence played from the position, e.g. 'Bxd2 Qxd2 Qxa1+'")
    parser.add_argument("--elo", type=int, help="also ask Maia3 what players of this rating play")
    parser.add_argument("--depth", type=int, default=30,
                        help="search depth; keep >= 30 for verdicts (shallow searches miss refutations)")
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "config.json"))
    args = parser.parse_args(argv)
    if bool(args.pgn) != bool(args.before):
        parser.error("--pgn and --before go together")
    if not args.pgn and not args.fen:
        parser.error("give --fen, or --pgn with --before")
    try:
        if args.pgn:
            with open(args.pgn, encoding="utf-8", errors="replace") as f:
                board = board_from_pgn(f.read(), args.before)
            if args.fen and chess.Board(args.fen).board_fen() != board.board_fen():
                parser.error(f"--fen does not match the game's position before {args.before}: {board.fen()}")
            history = "complete"
        else:
            board, history = chess.Board(args.fen), "unavailable"
    except ValueError as exc:
        parser.error(str(exc))
    engines = _engines(args.config)
    try:
        print(json.dumps(verify(engines, board, args.move, args.line, args.elo, args.depth, args.multipv,
                                history), indent=1))
    finally:
        engines.quit()


if __name__ == "__main__":
    main()
