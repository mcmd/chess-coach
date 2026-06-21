#!/usr/bin/env python
"""Calibrate the Maia "common move" threshold for the /drill popularity branch (plan Q7).

WHY (plan GT-7, P1-6): when the coach frames a mistake as "a natural trap — most players at your
level fall for this", that claim must rest on a real Maia probability, not a guess. Genuine blunders
are rare at all levels (~1-4%); typical-for-level mistakes are common (~40%+). This script measures the
distribution so you can pick the bar (≈25-30%?) that separates "common/natural" from "rare", instead of
hardcoding a number. The threshold is GLOBAL (Maia is already conditioned on the level), not per-student
— calibrate once on any representative corpus (msnid / Munziyana are fine).

UNLIKE the triage script, this one RUNS MAIA, so it must run inside the repo with the engine stack
available, and it is slow (one Maia "deep" prediction per mistake). Start small:

    PYTHONPATH=. python scripts/calibrate_maia_threshold.py \
        games/calib/msnid_annotated.pgn --profile gm --max-mistakes 200 --out maia_threshold.csv

What it does, per player-side move that lost >= an inaccuracy's worth of win% (read from the PGN's
[%eval], per GT-12): runs Maia at the mover's GAME-TIME rating (PGN header — the diagnostic Elo from
P0-4) and records the probability Maia assigned to the move ACTUALLY PLAYED. Then it prints the
distribution of that probability per severity bucket, so the "common" bar is visible.

NOTE: written against the app API (engines.human_predict -> HumanPrediction.raw_probs) but NOT executed
in this environment (no torch/maia here). Do a --max-mistakes 50 smoke run first.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import chess
import chess.pgn

# Make `app` importable no matter the shell, cwd, or PYTHONPATH: the repo root is the
# parent of scripts/. (Avoids the bash-only `PYTHONPATH=.` prefix, which PowerShell rejects.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.review import cp_to_win_pct, MATE_CP

BLUNDER, MISTAKE, INACCURACY = 30.0, 20.0, 10.0   # win%-drop bands (mirror review.classify_move)
OPENING_CUTOFF_MOVE = 8


def severity(drop: float) -> str:
    if drop >= BLUNDER:
        return "blunder"
    if drop >= MISTAKE:
        return "mistake"
    if drop >= INACCURACY:
        return "inaccuracy"
    return "ok"


def _white_cp(node):
    ev = node.eval()
    return None if ev is None else ev.white().score(mate_score=MATE_CP)


def _header_elo(headers, key, default):
    try:
        return int(headers.get(key, default))
    except (TypeError, ValueError):
        return default


def rating_band(elo: int) -> str:
    if elo < 1300:
        return "<1300"
    if elo < 1600:
        return "1300-1599"
    if elo < 1900:
        return "1600-1899"
    return "1900+"


def bootstrap_engines(profile: str):
    """Reuse the app's own config + engine bootstrap. load_config() reads sys.argv, so hand it a
    clean argv containing only flags it understands."""
    saved = sys.argv
    sys.argv = [saved[0], "--profile", profile]
    try:
        from app.config import load_config
        from app.engines import EngineManager
        config = load_config()
    finally:
        sys.argv = saved
    engines = EngineManager(config)
    engines.start()
    if getattr(engines, "maia", None) is None:
        sys.exit(f"[calib] Maia did not load ({getattr(engines, 'maia_error', 'unknown')}). "
                 f"This script needs the human model.")
    return engines, config


def collect_mistakes(pgn_path, engines, max_mistakes, min_cov):
    rows = []
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while len(rows) < max_mistakes:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                break
            h = game.headers
            welo = _header_elo(h, "WhiteElo", 1500)
            belo = _header_elo(h, "BlackElo", 1500)

            # eval coverage gate (GT-12)
            nodes = list(game.mainline())
            ann = sum(1 for n in nodes if n.eval() is not None)
            if not nodes or ann / len(nodes) < min_cov:
                continue

            history = []          # UCI history for Maia3 (history-aware)
            prev_white_cp = None
            for node in nodes:
                board_before = node.parent.board()
                mover_white = board_before.turn == chess.WHITE
                move_no = board_before.fullmove_number
                played_uci = node.move.uci()
                after_white_cp = _white_cp(node)

                if (prev_white_cp is not None and after_white_cp is not None
                        and move_no > OPENING_CUTOFF_MOVE):
                    sign = 1 if mover_white else -1
                    wb = cp_to_win_pct(sign * prev_white_cp)
                    wa = cp_to_win_pct(sign * after_white_cp)
                    drop = max(0.0, wb - wa)
                    sev = severity(drop)
                    if sev != "ok":
                        mover_elo = welo if mover_white else belo
                        oppo_elo = belo if mover_white else welo
                        try:
                            pred = engines.human_predict(
                                board_before.fen(), elo_self=mover_elo, elo_oppo=oppo_elo,
                                history_uci=list(history), tier="deep")
                            played_prob = float((pred.raw_probs or {}).get(played_uci, 0.0))
                        except Exception as e:
                            print(f"[calib] maia error, skipping one move: {e}", file=sys.stderr)
                            history.append(played_uci)
                            prev_white_cp = after_white_cp
                            continue
                        rows.append({
                            "ref": h.get("Site", "").rsplit("/", 1)[-1] or h.get("GameId", "?"),
                            "move_no": move_no, "side": "white" if mover_white else "black",
                            "played": board_before.san(node.move),
                            "severity": sev, "drop_winpct": round(drop, 1),
                            "mover_elo": mover_elo, "rating_band": rating_band(mover_elo),
                            "played_maia_prob": round(played_prob, 4),
                        })
                        if len(rows) >= max_mistakes:
                            break
                history.append(played_uci)
                prev_white_cp = after_white_cp
    return rows


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[i]


def summarize(rows):
    print(f"\n=== MAIA THRESHOLD CALIBRATION: {len(rows)} mistakes (played-move Maia probability) ===")
    print("Read this for the 'common' bar: blunders should cluster LOW, typical inaccuracies HIGHER.\n")
    by_sev = defaultdict(list)
    for r in rows:
        by_sev[r["severity"]].append(r["played_maia_prob"])
    print(f"  {'severity':<12}{'n':>5}{'mean':>8}{'p10':>8}{'p25':>8}{'median':>8}{'p75':>8}{'p90':>8}")
    for sev in ("inaccuracy", "mistake", "blunder"):
        v = sorted(by_sev.get(sev, []))
        if not v:
            continue
        mean = sum(v) / len(v)
        print(f"  {sev:<12}{len(v):>5}{mean:>8.1%}{pct(v,10):>8.1%}{pct(v,25):>8.1%}"
              f"{pct(v,50):>8.1%}{pct(v,75):>8.1%}{pct(v,90):>8.1%}")

    print("\n  share of mistakes with played-move prob >= candidate bar (per severity):")
    print(f"  {'bar':>6}{'inaccuracy':>12}{'mistake':>10}{'blunder':>10}")
    for bar in (0.15, 0.20, 0.25, 0.30, 0.40):
        cells = []
        for sev in ("inaccuracy", "mistake", "blunder"):
            v = by_sev.get(sev, [])
            cells.append(f"{(sum(1 for x in v if x >= bar) / len(v)):.0%}" if v else "-")
        print(f"  {bar:>6.0%}{cells[0]:>12}{cells[1]:>10}{cells[2]:>10}")

    print("\n  by rating band (median played-move prob):")
    by_band = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_band[r["rating_band"]][r["severity"]].append(r["played_maia_prob"])
    for band in ("<1300", "1300-1599", "1600-1899", "1900+"):
        if band not in by_band:
            continue
        cells = []
        for sev in ("inaccuracy", "mistake", "blunder"):
            v = sorted(by_band[band].get(sev, []))
            cells.append(f"{pct(v,50):.0%}" if v else "-")
        print(f"  {band:<12} inacc {cells[0]:>5}  mistake {cells[1]:>5}  blunder {cells[2]:>5}")

    print("\nInterpretation: pick the bar where 'common' (frame as a natural trap) cleanly separates from "
          "'rare' (just say 'you had a better move'). If bands diverge a lot, make the bar rating-aware.")


def main():
    ap = argparse.ArgumentParser(description="Calibrate the Maia 'common move' threshold (runs Maia).")
    ap.add_argument("pgns", nargs="+")
    ap.add_argument("--profile", required=True, help="Coach profile to load engines from (e.g. gm).")
    ap.add_argument("--out", default="maia_threshold.csv")
    ap.add_argument("--max-mistakes", type=int, default=400, help="Cap (Maia is slow). Start ~50-200.")
    ap.add_argument("--min-eval-coverage", type=float, default=0.6)
    args = ap.parse_args()

    engines, _ = bootstrap_engines(args.profile)
    print(f"[calib] Maia ready. Collecting up to {args.max_mistakes} mistakes "
          f"(>= inaccuracy, from [%eval]) ...")
    rows = []
    try:
        for path in args.pgns:
            remaining = args.max_mistakes - len(rows)
            if remaining <= 0:
                break
            rows += collect_mistakes(path, engines, remaining, args.min_eval_coverage)
    finally:
        try:
            engines.close()
        except Exception:
            pass

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summarize(rows)
    print(f"\nFull per-mistake detail: {args.out}")


if __name__ == "__main__":
    main()