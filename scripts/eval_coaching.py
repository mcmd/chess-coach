"""Coaching-quality regression harness (plan P0-5).

This is the *skeleton* (P0-5a): the deterministic, backend-only assertion library
(plan §6.1 tier 1) plus a thin fixture runner. It does NOT call Gemini — the
model-dependent interaction/pedagogy tiers (tiers 2–3) and the curated-corpus
ground-truth assertions (P0-5b) are seeded/verified by a human and grow over time
(plan §6.1, Q5). What lives here today is everything an agent can assert without a
model: properties of a `GameReview` produced from a `[%eval]`-annotated PGN.

Usage as a library (see tests/test_coaching_harness.py):
    from scripts.eval_coaching import theme_of, same_theme, check_review

Usage as a CLI (requires the real engine stack):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_coaching.py games/your.pgn
"""
from __future__ import annotations


# --- Tier 1: deterministic backend assertions over a GameReview -------------

def themes_of(review) -> list:
    """The verified ReviewTheme clusters (ids)."""
    return [t.id for t in (review.themes or [])]


def theme_of(review, ply: int) -> str | None:
    """The theme id a given agenda ply was clustered into, or None."""
    for t in (review.themes or []):
        if ply in t.plies:
            return t.id
    return None


def same_theme(review, ply_a: int, ply_b: int) -> bool:
    """Were these two agenda plies clustered into the same verified theme?"""
    a = theme_of(review, ply_a)
    return a is not None and a == theme_of(review, ply_b)


def check_review(review, spec: dict) -> list:
    """Run a fixture spec of deterministic assertions against a review. Returns a
    list of failure strings (empty = pass). Supported keys:

      min_themes:        int      — at least this many verified theme clusters
      separate_plies:    [[a,b]]  — each pair must be in DIFFERENT themes
      together_plies:    [[a,b]]  — each pair must be in the SAME theme
    """
    failures = []
    n = len(review.themes or [])
    if "min_themes" in spec and n < spec["min_themes"]:
        failures.append(f"min_themes: expected >= {spec['min_themes']}, got {n}")
    for a, b in spec.get("separate_plies", []):
        if same_theme(review, a, b):
            failures.append(f"separate_plies: plies {a} and {b} share theme {theme_of(review, a)!r}")
    for a, b in spec.get("together_plies", []):
        if not same_theme(review, a, b):
            failures.append(f"together_plies: plies {a} and {b} are not in the same theme "
                            f"({theme_of(review, a)!r} vs {theme_of(review, b)!r})")
    return failures


# --- CLI: run the real pipeline on a PGN and print the theme clustering ------

def _main(argv) -> int:
    import json
    from app.config import load_config
    from app.engines import EngineManager
    from app.review import review_game

    if not argv:
        print("usage: eval_coaching.py <pgn-path> [spec.json]")
        return 2
    pgn_path = argv[0]
    spec = json.load(open(argv[1])) if len(argv) > 1 else {}
    config = load_config()
    engines = EngineManager(config)
    engines.start()
    review = review_game(open(pgn_path, encoding="utf-8", errors="replace").read(), engines, config)

    print(f"eval_source={review.eval_source}  agenda_plies={review.agenda_plies}")
    print("themes:")
    for t in review.themes:
        print(f"  [{t.id}] {t.confidence}: plies {t.plies} — {t.safe_summary}")
    if spec:
        failures = check_review(review, spec)
        print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures))
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
