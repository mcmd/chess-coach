#!/usr/bin/env python
"""Eval-only within-game "slide" scanner — the cheap, no-model corpus probe.

NOT a pytest test (lives in scripts/). Uses ZERO Maia and ZERO Stockfish: it reads
the `[%eval]` annotations Lichess already ships in analysed games, so it runs in
seconds over thousands of games and costs no model time and no tokens.

Purpose (see design/drills_and_idle_prep_plan_20260614.md §3.4 and the maia3 FR-3):
the rejected Phase 4 trajectory feature degenerated to "a run of eval-losing moves",
computable from eval alone. Before investing anything further, answer two questions
a 4-game hand sample cannot:
  1. BASE RATE  - how often does a sustained one-sided eval slide (the target
     phenomenon) actually occur in this player's games?
  2. CANDIDATES - dump the real gradual-slide games so an eval-only detector can be
     designed/validated against them.

    python scripts/scan_eval_slides.py --user msnid [--max 3000] [--no-download]

Downloads (once, cached) the player's ANALYSED games via the Lichess games-export
API, then classifies each game's student side into an archetype from eval alone.
"""
import argparse
import os
import sys

import chess
import chess.pgn

from app.review import MATE_CP, cp_to_win_pct

# --- slide-detection thresholds (explicit so a single run is self-documenting) ---
OPENING_CUTOFF_MOVE = 8     # ignore student moves before this move number (theory noise)
PLY_DROP_MIN = 3.0          # a "losing" student move drops >= this many win%
MIN_RUN = 3                 # consecutive losing student moves to be a "run"
MIN_START_WINPCT = 30.0     # run must START from a non-floored position (had real chances)
TOTAL_LOSS_MIN = 25.0       # cumulative win% lost across the run to count as a real slide
CLIFF_FRACTION = 0.70       # if one move is >= this share of the total loss -> a single
                            #   cliff/blunder, NOT a gradual slide
SWING_FLIPS_CHAOS = 3       # >= this many 45<->55 win% crossings (either side) -> "chaos"

LICHESS_EXPORT = "https://lichess.org/api/games/user/{user}"


def download_games(user: str, out_path: str, max_games: int, perf: str,
                   rated: bool = True) -> None:
    import requests  # local import so scan-only runs don't need it

    params = {
        "evals": "true", "clocks": "false", "analysed": "true",
        "max": str(max_games), "perfType": perf, "opening": "false",
        "rated": "true" if rated else "false",
    }
    headers = {"Accept": "application/x-chess-pgn",
               "User-Agent": "chess-coach-calibration (local research)"}
    print(f"Downloading up to {max_games} ANALYSED games for '{user}' "
          f"(perf={perf}) -> {out_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bytes_written = 0
    with requests.get(LICHESS_EXPORT.format(user=user), params=params, headers=headers,
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
    print(f"  downloaded {bytes_written/1024:.0f} KiB")


def student_color(headers, user: str):
    u = user.strip().casefold()
    if u in headers.get("White", "").casefold():
        return chess.WHITE
    if u in headers.get("Black", "").casefold():
        return chess.BLACK
    return None


def student_moves(game, student_white: bool):
    """Return [(move_number, san, winpct_before, winpct_after, drop)] for the
    student's moves past the opening, using only [%eval] annotations. win% is in
    the student's perspective; drop = win% lost on that move (>=0 is a loss)."""
    rows = []
    node = game
    while node.variations:
        parent = node
        node = node.variations[0]
        ply = node.ply()                       # 1 = White's 1st move
        mover_is_white = (ply % 2 == 1)
        if mover_is_white != student_white:
            continue
        move_no = (ply + 1) // 2
        if move_no < OPENING_CUTOFF_MOVE:
            continue
        ev_before, ev_after = parent.eval(), node.eval()
        if ev_before is None or ev_after is None:
            continue
        wcp_b = ev_before.white().score(mate_score=MATE_CP)
        wcp_a = ev_after.white().score(mate_score=MATE_CP)
        scp_b = wcp_b if student_white else -wcp_b
        scp_a = wcp_a if student_white else -wcp_a
        wp_b = cp_to_win_pct(scp_b)
        wp_a = cp_to_win_pct(scp_a)
        rows.append((move_no, parent.board().san(node.move),
                     wp_b, wp_a, wp_b - wp_a))
    return rows


def classify(rows):
    """Label the student's game from eval alone. Returns (label, detail dict)."""
    if not rows:
        return "no_evals", {}

    # losing-run detection: maximal runs of consecutive losing student moves whose
    # first move started from a non-floored position.
    #   summed = sum of the student's per-move drops (can be inflated by opponent
    #            give-backs between the student's moves)
    #   net    = start win% (before run) - end win% (after run) = the ACTUAL one-sided
    #            collapse. A true slide needs a large NET drop, not just summed.
    runs = []
    i, n = 0, len(rows)
    while i < n:
        if rows[i][4] < PLY_DROP_MIN or rows[i][2] < MIN_START_WINPCT:
            i += 1
            continue
        j = i
        while j < n and rows[j][4] >= PLY_DROP_MIN:
            j += 1
        seg = rows[i:j]
        runs.append({
            "start_move": seg[0][0], "end_move": seg[-1][0], "length": len(seg),
            "start_winpct": round(seg[0][2], 1),
            "end_winpct": round(seg[-1][3], 1),
            "summed": round(sum(r[4] for r in seg), 1),
            "net": round(seg[0][2] - seg[-1][3], 1),
            "max_single": round(max(r[4] for r in seg), 1),
        })
        i = j

    # A real gradual slide: >=MIN_RUN losing moves, a large NET one-sided collapse,
    # and not dominated by a single cliff move.
    gradual = [r for r in runs
               if r["length"] >= MIN_RUN and r["net"] >= TOTAL_LOSS_MIN
               and r["max_single"] < CLIFF_FRACTION * r["net"]]
    # Churn: bled a lot (summed) but gave most back (small net) -> mutual chaos, not a slide.
    churn = [r for r in runs if r["summed"] >= TOTAL_LOSS_MIN and r["net"] < TOTAL_LOSS_MIN]
    cliffs = [r for r in rows if r[4] >= TOTAL_LOSS_MIN and r[2] >= MIN_START_WINPCT]

    # chaos: count 45<->55 win% crossings across the student's after-move evals.
    flips, prev = 0, None
    for r in rows:
        side = "hi" if r[3] >= 55 else ("lo" if r[3] <= 45 else None)
        if side and prev and side != prev:
            flips += 1
        if side:
            prev = side

    detail = {"runs": runs, "gradual": gradual, "n_cliffs": len(cliffs), "flips": flips}
    if gradual:
        return "gradual_slide", detail
    if cliffs:
        return "single_cliff", detail
    if churn or flips >= SWING_FLIPS_CHAOS:
        return "chaos", detail
    if any(r[4] >= PLY_DROP_MIN for r in rows):
        return "minor_errors", detail
    return "clean", detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--perf", default="blitz,rapid,classical")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--rated", default=True, action=argparse.BooleanOptionalAction)
    args = ap.parse_args()

    out_path = args.out or f"games/calib/{args.user}_annotated.pgn"
    if not args.no_download and not os.path.exists(out_path):
        download_games(args.user, out_path, args.max, args.perf, args.rated)
    elif os.path.exists(out_path):
        print(f"Using cached {out_path} ({os.path.getsize(out_path)/1024:.0f} KiB)")
    else:
        print(f"No file at {out_path} and --no-download set.")
        return 1

    labels = {}
    candidates = []
    total = parsed = no_student = analyzable = 0
    with open(out_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total += 1
            sc = student_color(game.headers, args.user)
            if sc is None:
                no_student += 1
                continue
            rows = student_moves(game, sc == chess.WHITE)
            label, detail = classify(rows)
            parsed += 1
            if label != "no_evals":
                analyzable += 1
            labels[label] = labels.get(label, 0) + 1
            if label == "gradual_slide":
                g = max(detail["gradual"], key=lambda r: r["net"])
                candidates.append((game.headers.get("Site", "?"),
                                   "white" if sc == chess.WHITE else "black",
                                   g))

    print(f"\nPlayer: {args.user}   file: {out_path}")
    print(f"Games in file: {total} | student found: {parsed} | "
          f"no student (name not matched): {no_student} | with evals: {analyzable}")
    print(f"\nThresholds: OPENING_CUTOFF_MOVE={OPENING_CUTOFF_MOVE} PLY_DROP_MIN={PLY_DROP_MIN} "
          f"MIN_RUN={MIN_RUN} MIN_START_WINPCT={MIN_START_WINPCT} "
          f"TOTAL_LOSS_MIN={TOTAL_LOSS_MIN} CLIFF_FRACTION={CLIFF_FRACTION}")
    print("\nArchetype breakdown (student side):")
    order = ["gradual_slide", "single_cliff", "chaos", "minor_errors", "clean", "no_evals"]
    for k in order:
        if k in labels:
            pct = 100.0 * labels[k] / analyzable if analyzable and k != "no_evals" else 0.0
            suffix = f"  ({pct:.1f}% of analyzable)" if k != "no_evals" else ""
            print(f"  {k:<14} {labels[k]:>5}{suffix}")
    base = 100.0 * labels.get("gradual_slide", 0) / analyzable if analyzable else 0.0
    print(f"\n>>> BASE RATE of gradual one-sided slides: "
          f"{labels.get('gradual_slide', 0)}/{analyzable} = {base:.1f}%")

    if candidates:
        cand_path = f"games/calib/{args.user}_slide_candidates.txt"
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write(f"# gradual-slide candidates for {args.user}\n")
            for site, color, g in candidates:
                line = (f"{site}  ({color})  moves {g['start_move']}-{g['end_move']} "
                        f"({g['length']} moves)  win% {g['start_winpct']}->{g['end_winpct']} "
                        f"(net -{g['net']}, summed {g['summed']}, max single {g['max_single']})")
                f.write(line + "\n")
        print(f"\nWrote {len(candidates)} candidate(s) to {cand_path}. First few:")
        for site, color, g in candidates[:10]:
            print(f"  {site} ({color}) mv {g['start_move']}-{g['end_move']} "
                  f"win% {g['start_winpct']}->{g['end_winpct']} "
                  f"(net -{g['net']}, summed {g['summed']}, max {g['max_single']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
