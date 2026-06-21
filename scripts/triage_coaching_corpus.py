#!/usr/bin/env python
"""Triage local PGN corpora for COACHING usefulness — no model, no engine, no network.

Purpose (see the assessment/plan, P0-5 + P1-6/8 + GT-12): you have 5k+ games across
three local PGNs. Before building the coaching-quality harness and the drill corpus we
need to know *which* games are worth using, and there are three distinct uses:

  - harness_seed   : a clean, eval-annotated game with a small number of clear, diagnosable
                     mistakes — ideal ground-truth fixtures for the P0-5 regression harness.
  - drill_material : a game the *student* played (matches --user) with diagnosable mistakes,
                     for the /drill corpus; the variety matrix tells us if there are enough
                     distinct instances per (error-type x phase) for spaced, vary-the-instance
                     repetition (P1-6).
  - slide_example  : a game containing a sustained one-sided win% slide (cumulative-drift
                     teaching case), detected with the SAME logic as review._detect_slides.

GT-12 (eval-source invariant): win%-space classification trusts the PGN's [%eval] (Lichess
deep eval), NOT local shallow Stockfish. This script therefore reads ONLY [%eval] and reports
eval coverage; games below --min-eval-coverage are flagged low_value (the coach would fall
back to noisy local Stockfish on them).

It deliberately reuses app.review.cp_to_win_pct / MATE_CP / the SLIDE_* constants so its
verdicts match the live coach. If run outside the repo it falls back to the same literals.

Run (inside the repo, with the venv):
    PYTHONPATH=. python scripts/triage_coaching_corpus.py \
        games/calib/msnid_annotated.pgn games/calib/Munziyana_annotated.pgn games/mcmd.pgn \
        --user mcmd --out triage_report.csv

What to send back: the printed SUMMARY (buckets, eval-coverage, the variety matrix) and, if
you like, the CSV. Neither contains positions/FENs — only game refs (Lichess IDs are your own
games) and counts — so the PGNs stay local.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict

import chess
import chess.pgn

# --- Faithful constants: prefer the app's so triage == coaching; fall back to literals. ---
try:
    from app.review import (cp_to_win_pct, MATE_CP, SLIDE_MIN_RUN, SLIDE_MIN_START_WINPCT,
                            SLIDE_NET_LOSS_MIN, SLIDE_CLIFF_FRACTION, SLIDE_PLY_DROP_MIN)
    _SRC = "app.review"
except Exception:  # pragma: no cover - lets the script run outside the repo
    import math
    MATE_CP = 10000
    def cp_to_win_pct(cp: float) -> float:
        return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)
    SLIDE_MIN_RUN = 3
    SLIDE_MIN_START_WINPCT = 30.0
    SLIDE_NET_LOSS_MIN = 25.0
    SLIDE_CLIFF_FRACTION = 0.70
    SLIDE_PLY_DROP_MIN = 3.0
    _SRC = "fallback literals"

# Move-severity bands mirror review.classify_move's win%-drop thresholds. We drop the
# is_best/is_only/is_forced credit logic on purpose: that needs an engine, and triage only
# cares about how bad the mistakes are, not about awarding brilliancy credit.
BLUNDER, MISTAKE, INACCURACY = 30.0, 20.0, 10.0
OPENING_CUTOFF_MOVE = 8        # matches scripts/scan_eval_slides.py: ignore early theory noise
HARNESS_MAX_MISTAKES = 5       # a clean teaching game, not a chaotic blunder-fest
MIN_PLIES = 20                 # skip trivial/aborted games


def severity(drop: float) -> str:
    if drop >= BLUNDER:
        return "blunder"
    if drop >= MISTAKE:
        return "mistake"
    if drop >= INACCURACY:
        return "inaccuracy"
    return "ok"


def phase_of(board: chess.Board, move_no: int) -> str:
    """Coarse phase for the variety matrix (triage only). Endgame ~ <=7 pieces or queens off
    with little material; opening by move number; else middlegame."""
    n = len(board.piece_map())
    queens = board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)
    if n <= 7 or (not queens and n <= 12):
        return "endgame"
    if move_no <= 10:
        return "opening"
    return "middlegame"


def _white_cp(node) -> float:
    """White-POV centipawns from this node's [%eval], or None if unannotated."""
    ev = node.eval()
    if ev is None:
        return None
    return ev.white().score(mate_score=MATE_CP)


def _game_ref(headers) -> str:
    site = headers.get("Site", "")
    if "lichess.org/" in site:
        return site.rsplit("/", 1)[-1]
    gid = headers.get("GameId")
    if gid:
        return gid
    return f"{headers.get('White','?')}-{headers.get('Black','?')}-{headers.get('Date','?')}"


def _detect_slide(rows) -> dict:
    """Faithful port of review._detect_slides on the student's (move_no, wb, wa, drop) rows
    (opening already excluded). Returns the first qualifying slide segment or None."""
    i, n = 0, len(rows)
    while i < n:
        if rows[i][3] < SLIDE_PLY_DROP_MIN or rows[i][1] < SLIDE_MIN_START_WINPCT:
            i += 1
            continue
        j = i
        while j < n and rows[j][3] >= SLIDE_PLY_DROP_MIN:
            j += 1
        seg = rows[i:j]
        net = seg[0][1] - seg[-1][2]
        max_single = max(x[3] for x in seg)
        if (len(seg) >= SLIDE_MIN_RUN and net >= SLIDE_NET_LOSS_MIN
                and max_single < SLIDE_CLIFF_FRACTION * net):
            return {"start_move": seg[0][0], "end_move": seg[-1][0],
                    "plies": len(seg), "net_loss": round(net, 1)}
        i = j
    return None


def analyse_side(game, student_white: bool):
    """Walk one game from one side's POV. Returns a per-(game,side) dict of triage signals."""
    counts = Counter()                 # severity counts (student moves)
    by_phase = defaultdict(Counter)    # phase -> severity -> n
    slide_rows = []                    # (move_no, wb, wa, drop) non-opening student moves
    annotated_plies = total_plies = 0
    swings = 0                         # big win% moves by EITHER side (chaos proxy)

    prev_white_cp = None
    for node in game.mainline():
        board_before = node.parent.board()
        mover_white = board_before.turn == chess.WHITE
        move_no = board_before.fullmove_number
        after_white_cp = _white_cp(node)
        total_plies += 1
        if after_white_cp is not None:
            annotated_plies += 1

        if prev_white_cp is not None and after_white_cp is not None:
            # chaos proxy: any side shedding >=10 win% (White POV magnitude)
            if abs(cp_to_win_pct(prev_white_cp) - cp_to_win_pct(after_white_cp)) >= INACCURACY:
                swings += 1
            if mover_white == student_white:
                sign = 1 if student_white else -1
                wb = cp_to_win_pct(sign * prev_white_cp)
                wa = cp_to_win_pct(sign * after_white_cp)
                drop = max(0.0, wb - wa)
                sev = severity(drop)
                counts[sev] += 1
                ph = phase_of(board_before, move_no)
                by_phase[ph][sev] += 1
                if move_no > OPENING_CUTOFF_MOVE:
                    slide_rows.append((move_no, wb, wa, drop))
        prev_white_cp = after_white_cp

    coverage = (annotated_plies / total_plies) if total_plies else 0.0
    slide = _detect_slide(slide_rows) if slide_rows else None
    diagnosable = counts["mistake"] + counts["blunder"]
    return {
        "coverage": coverage, "total_plies": total_plies, "swings": swings,
        "inaccuracy": counts["inaccuracy"], "mistake": counts["mistake"],
        "blunder": counts["blunder"], "diagnosable": diagnosable,
        "by_phase": by_phase, "slide": slide,
    }


def bucket(sig, is_student_side, min_cov) -> str:
    if sig["total_plies"] < MIN_PLIES or sig["coverage"] < min_cov:
        return "low_value"
    if sig["slide"]:
        return "slide_example"
    clean = 1 <= sig["diagnosable"] <= HARNESS_MAX_MISTAKES
    if is_student_side and sig["diagnosable"] >= 1:
        return "drill_material"
    if clean:
        return "harness_seed"
    if sig["diagnosable"] == 0:
        return "low_value"
    return "review_only"   # has mistakes but chaotic / not the student — usable, lower priority


def target_sides(headers, users):
    """Which side(s) to analyse. If the user played, that side (student=True). Else both
    sides (student=False) so the game can still be a harness/slide candidate."""
    w = (headers.get("White", "") or "").casefold()
    b = (headers.get("Black", "") or "").casefold()
    out = []
    if users:
        if w in users:
            out.append((True, True))
        if b in users:
            out.append((False, True))
        if out:
            return out
    return [(True, False), (False, False)]


def main():
    ap = argparse.ArgumentParser(description="Triage local PGNs for coaching usefulness (eval-only).")
    ap.add_argument("pgns", nargs="+", help="One or more local .pgn paths.")
    ap.add_argument("--user", default="", help="Comma-separated student handle(s), e.g. 'mcmd'.")
    ap.add_argument("--out", default="triage_report.csv", help="CSV output path.")
    ap.add_argument("--min-eval-coverage", type=float, default=0.6,
                    help="Min fraction of [%%eval]-annotated plies to be usable (GT-12).")
    ap.add_argument("--max-games", type=int, default=0, help="Stop after N games (0 = all).")
    args = ap.parse_args()

    users = {u.strip().casefold() for u in args.user.split(",") if u.strip()}
    print(f"[triage] constants from: {_SRC}  |  users={sorted(users) or '(both sides)'}  "
          f"|  min eval coverage={args.min_eval_coverage:.0%}")

    rows = []
    buckets = Counter()
    variety = defaultdict(Counter)        # phase -> severity -> n (student/analyzed sides)
    cov_hist = Counter()
    games_seen = 0

    for path in args.pgns:
        try:
            fh = open(path, encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[triage] SKIP {path}: {e}", file=sys.stderr)
            continue
        with fh:
            while True:
                if args.max_games and games_seen >= args.max_games:
                    break
                try:
                    game = chess.pgn.read_game(fh)
                except Exception as e:                       # malformed game; skip, keep going
                    print(f"[triage] parse error in {path}: {e}", file=sys.stderr)
                    continue
                if game is None:
                    break
                games_seen += 1
                h = game.headers
                ref = _game_ref(h)
                for student_white, is_student in target_sides(h, users):
                    sig = analyse_side(game, student_white)
                    bkt = bucket(sig, is_student, args.min_eval_coverage)
                    buckets[bkt] += 1
                    cov_hist[round(sig["coverage"] * 10) / 10] += 1
                    if bkt in ("drill_material", "harness_seed", "slide_example"):
                        for ph, sevs in sig["by_phase"].items():
                            for sev in ("inaccuracy", "mistake", "blunder"):
                                if sevs.get(sev):
                                    variety[ph][sev] += sevs[sev]
                    rows.append({
                        "ref": ref, "file": path.rsplit("/", 1)[-1],
                        "white": h.get("White", ""), "black": h.get("Black", ""),
                        "white_elo": h.get("WhiteElo", ""), "black_elo": h.get("BlackElo", ""),
                        "result": h.get("Result", ""), "eco": h.get("ECO", ""),
                        "student_side": "white" if student_white else "black",
                        "is_student": is_student, "plies": sig["total_plies"],
                        "eval_coverage": round(sig["coverage"], 3), "swings": sig["swings"],
                        "inaccuracy": sig["inaccuracy"], "mistake": sig["mistake"],
                        "blunder": sig["blunder"], "diagnosable": sig["diagnosable"],
                        "slide": "yes" if sig["slide"] else "",
                        "slide_net_loss": sig["slide"]["net_loss"] if sig["slide"] else "",
                        "bucket": bkt,
                    })

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---------------------------- SUMMARY (paste this back) ----------------------------
    print(f"\n=== TRIAGE SUMMARY: {games_seen} games -> {len(rows)} (game,side) rows ===")
    print("\nBuckets:")
    for b in ("drill_material", "harness_seed", "slide_example", "review_only", "low_value"):
        print(f"  {b:<15} {buckets.get(b, 0)}")

    print("\nEval coverage (GT-12 — only high-coverage games are trustworthy):")
    for c in sorted(cov_hist):
        print(f"  ~{int(c*100):>3}% : {cov_hist[c]}")

    print("\nVariety matrix (phase x severity, drill/harness/slide rows) "
          "— shows whether each pattern has enough distinct instances for P1-6:")
    print(f"  {'phase':<12}{'inaccuracy':>12}{'mistake':>10}{'blunder':>10}")
    for ph in ("opening", "middlegame", "endgame"):
        sv = variety.get(ph, {})
        print(f"  {ph:<12}{sv.get('inaccuracy',0):>12}{sv.get('mistake',0):>10}{sv.get('blunder',0):>10}")

    def top(bkt, n=15):
        cand = [r for r in rows if r["bucket"] == bkt]
        cand.sort(key=lambda r: (-r["diagnosable"], -r["eval_coverage"]))
        return cand[:n]

    print("\nTop harness_seed candidates (ref | side | mistakes/blunders | coverage):")
    for r in top("harness_seed"):
        print(f"  {r['ref']:<14} {r['student_side']:<6} m{r['mistake']}/b{r['blunder']}  {r['eval_coverage']:.0%}")
    print("\nSlide examples (ref | side | net win% lost):")
    for r in [r for r in rows if r["bucket"] == "slide_example"][:15]:
        print(f"  {r['ref']:<14} {r['student_side']:<6} -{r['slide_net_loss']}")

    print(f"\nFull per-game detail: {args.out}")


if __name__ == "__main__":
    main()