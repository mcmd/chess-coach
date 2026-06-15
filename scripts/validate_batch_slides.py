"""One-off live validation of the Phase 5 cross-game slide leverage.

Drives the REAL batch pipeline (Stockfish + Maia3) on a multi-game PGN exactly
as the /batch command does, then prints the slide_signature, the CROSS-GAME
SLIDE block + per-pattern Axis tags from the digest, and the dossier slide_trend
the run produces. All output files (batch reports, dossier) are written to a
throwaway temp dir so the real profile is untouched.

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/validate_batch_slides.py <pgn> [player]
"""
import json
import os
import sys
import tempfile

from app.config import AppConfig
from app.engines import EngineManager
from app.batch import run_batch_review, batch_dossier_summary
from app.memory import load_dossier, update_dossier_with_batch


def main():
    pgn_path = sys.argv[1] if len(sys.argv) > 1 else "games/lichess_mcmd_2026-06-14.pgn"
    player = sys.argv[2] if len(sys.argv) > 2 else None

    with open("config.json") as f:
        raw = json.load(f)
    if player is None:
        player = raw.get("player_name", "mcmd")

    tmp = tempfile.mkdtemp(prefix="batch_validate_")
    cfg = AppConfig(
        user_elo=raw.get("user_elo", 1150),
        language="English",
        stockfish_path=raw["stockfish_path"],
        gemini_model="unused",
        api_key="unused",
        history_file=os.path.join(tmp, "history.json"),
        dossier_file=os.path.join(tmp, "player_dossier.json"),
        cli_history_file=os.path.join(tmp, "cli.txt"),
        lesson_file=os.path.join(tmp, "lesson_state.json"),
        player_name=player,
        use_online_databases=False,  # keep validation offline/deterministic
        **{k: raw.get("engine", {}).get(k) for k in ("batch_eval_time",) if raw.get("engine", {}).get(k)},
    )
    print(f"[validate] player={player!r}  out_dir={tmp}")
    print(f"[validate] pgn={pgn_path}")

    engines = EngineManager(cfg)
    engines.start()
    if engines.maia_error:
        print(f"[validate] WARNING maia load: {engines.maia_error}")

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        pgn_text = f.read()

    batch = run_batch_review(pgn_text, engines, cfg,
                             on_status=lambda s: print(f"  · {s}"))
    if isinstance(batch, dict):
        print(f"[validate] ERROR: {batch.get('error')}")
        return 1

    print("\n" + "=" * 70)
    print(f"games_analyzed={batch.games_analyzed} skipped={batch.games_skipped} "
          f"moments={batch.moments_found} patterns={len(batch.patterns)}")
    print("=" * 70)

    print("\n--- slide_signature (raw) ---")
    print(json.dumps(batch.slide_signature, indent=2, default=str))

    print("\n--- per-pattern axis ---")
    for n, p in enumerate(batch.patterns, 1):
        print(f"  [{n}] {p.title!r}  axis={p.axis}  in_slide={p.in_slide_count}/{p.count}")

    print("\n--- digest: CROSS-GAME SLIDE + Axis lines ---")
    for line in batch.digest().splitlines():
        if any(t in line for t in ("CROSS-GAME SLIDE", "slide", "Slide", "Axis:", "SLIDE")):
            print("  " + line.strip())

    print("\n--- dossier slide_trend after update_dossier_with_batch ---")
    update_dossier_with_batch(cfg.dossier_file, batch_dossier_summary(batch))
    dossier = load_dossier(cfg.dossier_file)
    print(json.dumps(dossier.get("slide_trend"), indent=2, default=str))

    print(f"\n[validate] done. temp dir (delete when finished): {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
