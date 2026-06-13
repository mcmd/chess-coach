#!/usr/bin/env python
"""Manual calibration harness for Phase 4 within-game trajectory detection.

NOT a pytest test (lives in scripts/, like smoke_maia2.py) — it loads the real
Stockfish + maia3 models, which unit tests never do.

    python scripts/calibrate_trajectory.py path/to/game.pgn [--profile NAME]

Reviews the game, then dumps the FULL per-ply skill-band table the trajectory
detector sees (p_lo / p_mid / p_hi, band, win% lost, collapse flag) — not just
the digest block, which only shows fired segments. This lets us judge whether
the raw signal is meaningful and re-tune the TRAJECTORY_* thresholds from one
(slow) run. It mirrors review._analyze_trajectory and also calls the real
function for parity.
"""
import sys

from dotenv import load_dotenv

from app import review as R
from app import trace as tracing
from app.config import load_config
from app.engines import EngineManager


def _pct(x):
    """A 0..1 probability as a right-aligned percent, or a dash for None."""
    return f"{x * 100:5.1f}" if x is not None else "    -"


def main():
    # load_config() parses sys.argv (it only knows --profile/--trace/etc). Pull
    # our positional PGN path out first, then hand the remaining flags to it.
    pgn_path, rest = None, []
    for a in sys.argv[1:]:
        if pgn_path is None and not a.startswith("-"):
            pgn_path = a
        else:
            rest.append(a)
    if not pgn_path:
        print("usage: python scripts/calibrate_trajectory.py <game.pgn> [--profile NAME]")
        return 2
    sys.argv = [sys.argv[0]] + rest

    load_dotenv()
    config = load_config()
    tracing.set_enabled(False)

    with open(pgn_path, "r", encoding="utf-8") as f:
        pgn_text = f.read()

    engines = EngineManager(config)
    engines.start()
    try:
        if getattr(engines, "maia", None) is None:
            print(f"No human model loaded ({getattr(engines, 'maia_error', '?')}). "
                  "Trajectory needs maia.")
            return 1
        print(f"maia_version={config.maia_version}  deep={config.maia3_deep_model}  "
              f"scan={config.maia3_scan_model}  history_plies={config.maia_history_plies}")

        review = R.review_game(pgn_text, engines, config, with_maia=True, enrich=False)
        if review is None:
            print("Could not parse a game from the PGN.")
            return 1

        sc = review.student_color
        if sc is None:
            print("No student color resolved — set player_name in config to match a PGN header.")
            return 1
        student_elo = review.white_elo if sc == "white" else review.black_elo
        oppo_elo = (review.black_elo if sc == "white" else review.white_elo) or student_elo
        print(f"\nStudent: {sc} ~{student_elo}   opponent ~{oppo_elo}")

        lo = max(1100, student_elo - R.TRAJECTORY_ELO_GAP)
        hi = min(1900, student_elo + R.TRAJECTORY_ELO_GAP)
        print(f"Sweep anchors: lo={lo}  mid={student_elo}  hi={hi}  "
              f"(thresholds: MIN_PROB={R.TRAJECTORY_MIN_PROB} "
              f"PLY_MIN_DROP={R.TRAJECTORY_PLY_MIN_DROP} MIN_RUN={R.TRAJECTORY_MIN_RUN})")
        edged_out = lo >= student_elo or hi <= student_elo
        if edged_out:
            print(f"!! Student Elo {student_elo} is at the edge of Maia's [1100,1900] range — "
                  "the production guard returns early and NO trajectory is computed. "
                  "Table below is shown anyway, for insight.")

        student_plies = [r for r in review.records
                         if r.color == sc and r.phase != "opening"]
        print(f"Post-opening student plies: {len(student_plies)} "
              f"(need >= {R.TRAJECTORY_MIN_PLIES} to analyze)\n")

        def p_at(r, elo):
            with tracing.suspended():
                pred = engines.maia_predict(r.fen_before, elo_self=elo, elo_oppo=oppo_elo,
                                            history_uci=(r.history_uci or None), tier="scan")
            return pred["raw_probs"].get(r.uci, 0.0) if pred else 0.0

        print(f"{'mv':>4} {'move':<8} {'qual':<10} {'drop%':>6} "
              f"{'p_lo':>6} {'p_mid':>6} {'p_hi':>6} {'band':>6} {'collapse':>8}")
        print("-" * 72)
        rows = []
        for r in student_plies:
            p_lo = p_at(r, lo)
            p_hi = p_at(r, hi)
            p_mid = r.maia_played_prob if r.maia_played_prob is not None else p_at(r, student_elo)
            band = R._skill_band(p_lo, p_mid, p_hi)
            is_collapse = band == "low" and r.drop_pct >= R.TRAJECTORY_PLY_MIN_DROP
            rows.append((r, is_collapse))
            print(f"{r.number:>4} {r.san:<8} {r.quality:<10} {r.drop_pct:>6.1f} "
                  f"{_pct(p_lo)} {_pct(p_mid)} {_pct(p_hi)} {band:>6} "
                  f"{'YES' if is_collapse else '':>8}")

        # Segments under current thresholds — mirrors _analyze_trajectory grouping.
        segs, i, n = [], 0, len(rows)
        while i < n:
            if not rows[i][1]:
                i += 1
                continue
            j = i
            while j < n and rows[j][1]:
                j += 1
            run = [r for r, _ in rows[i:j]]
            if len(run) >= R.TRAJECTORY_MIN_RUN:
                segs.append((run[0].number, run[-1].number, len(run),
                             round(sum(x.drop_pct for x in run) / len(run), 1)))
            i = j

        print()
        if segs:
            print("Segments that WOULD fire under current thresholds:")
            for s in segs:
                print(f"  moves {s[0]}-{s[1]}  ({s[2]} plies)  avg {s[3]} win% lost/move")
        else:
            print("No segment fires under current thresholds.")

        # Parity: run the production path and show what the digest would carry.
        review.trajectory = None
        R._analyze_trajectory(review, engines)
        print(f"\nProduction _analyze_trajectory -> review.trajectory:\n  {review.trajectory}")
        if edged_out and review.trajectory is None:
            print("  (None as expected — range guard fired.)")
    finally:
        engines.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
