"""Coaching-quality regression harness (plan P0-5).

This is the *skeleton* (P0-5a): the deterministic, backend-only assertion library
(plan §6.1 tier 1) plus a thin fixture runner. It does NOT call Gemini — the
model-dependent interaction/pedagogy tiers (tiers 2–3) and the curated-corpus
ground-truth assertions (P0-5b) are seeded/verified by a human and grow over time
(plan §6.1, Q5). What lives here today is everything an agent can assert without a
model: properties of a `GameReview` produced from a `[%eval]`-annotated PGN.

P0-5b now adds a source-provenance and human-approval gate around that skeleton.
Engine proposal packets never promote themselves to ground truth: a named human
must approve meaningful coaching assertions before a fixture counts as ready.

Usage as a library (see tests/test_coaching_harness.py):
    from scripts.eval_coaching import theme_of, same_theme, check_review

Usage as a CLI (requires the real engine stack):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_coaching.py games/your.pgn
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path

import chess.pgn


CORPUS_SCHEMA_VERSION = 1
_FIXTURE_STATUSES = {"pending_human", "approved", "rejected"}
_FIXTURE_SIDES = {"white", "black"}
_ASSERTION_KEYS = {
    "min_themes", "separate_plies", "together_plies", "student_color", "eval_source",
    "agenda_must_include", "agenda_must_exclude", "diagnosis_by_ply",
    "key_reason_tokens_by_ply", "slide_count",
}
_PEDAGOGICAL_ASSERTION_KEYS = _ASSERTION_KEYS - {"student_color", "eval_source"}


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
    list of failure strings (empty = pass). Assertions cover source identity,
    agenda inclusion/exclusion, diagnosis and key-reason tokens, slide count, and
    verified theme grouping. The manifest validator rejects unknown assertion keys.
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
    if "student_color" in spec and review.student_color != spec["student_color"]:
        failures.append(
            f"student_color: expected {spec['student_color']!r}, got {review.student_color!r}")
    if "eval_source" in spec and review.eval_source != spec["eval_source"]:
        failures.append(f"eval_source: expected {spec['eval_source']!r}, got {review.eval_source!r}")
    agenda = set(review.agenda_plies)
    for ply in spec.get("agenda_must_include", []):
        if ply not in agenda:
            failures.append(f"agenda_must_include: ply {ply} is absent")
    for ply in spec.get("agenda_must_exclude", []):
        if ply in agenda:
            failures.append(f"agenda_must_exclude: ply {ply} is present")
    for raw_ply, expected in spec.get("diagnosis_by_ply", {}).items():
        ply = int(raw_ply)
        record = review.record_at(ply)
        actual = getattr(record, "recommended_diagnosis", None)
        if actual != expected:
            failures.append(f"diagnosis_by_ply: ply {ply} expected {expected!r}, got {actual!r}")
    for raw_ply, required in spec.get("key_reason_tokens_by_ply", {}).items():
        ply = int(raw_ply)
        record = review.record_at(ply)
        reasons = (getattr(record, "explanation_primitives", None) or {}).get("reasons") or []
        key = next((reason for reason in reasons if reason.get("key")), None)
        available = set((key or {}).get("squares", [])) | set((key or {}).get("keywords", []))
        missing = set(required) - available
        if missing:
            failures.append(
                f"key_reason_tokens_by_ply: ply {ply} missing {sorted(missing)}; got {sorted(available)}")
    if "slide_count" in spec:
        actual = len(review.slides or [])
        if actual != spec["slide_count"]:
            failures.append(f"slide_count: expected {spec['slide_count']}, got {actual}")
    return failures


def review_proposal_packet(review) -> dict:
    """Engine-derived evidence for a human labeler, never ground truth itself."""
    student_records = [record for record in review.records
                       if review.student_color is None or record.color == review.student_color]
    candidates = []
    for record in student_records:
        if record.drop_pct < 5 and record.ply not in review.agenda_plies:
            continue
        key_reasons = [reason for reason in (record.explanation_primitives or {}).get("reasons", [])
                       if reason.get("key")]
        candidates.append({
            "ply": record.ply,
            "move": record.move_label,
            "played_san": record.san,
            "best_move": record.best_move_label,
            "drop_pct": record.drop_pct,
            "quality": record.quality,
            "phase": record.phase,
            "on_current_agenda": record.ply in review.agenda_plies,
            "recommended_diagnosis": record.recommended_diagnosis,
            "lesson_objective": record.lesson_objective,
            "motifs": record.best_line_motifs,
            "key_reasons": key_reasons,
            "initial_question": record.initial_question,
        })
    return {
        "notice": "PROPOSAL ONLY: engine-derived evidence for human review; not an approved label.",
        "eval_source": review.eval_source,
        "student_color": review.student_color,
        "agenda_plies": review.agenda_plies,
        "candidates": candidates,
        "themes": [
            {"id": theme.id, "label": theme.label, "plies": theme.plies,
             "confidence": theme.confidence, "safe_summary": theme.safe_summary}
            for theme in (review.themes or [])
        ],
        "slides": review.slides or [],
    }


# --- P0-5b: curated-corpus provenance and human-approval gate ---------------

def _eval_coverage(game) -> float:
    nodes = list(game.mainline())
    if not nodes:
        return 0.0
    return sum(node.eval() is not None for node in nodes) / len(nodes)


def validate_corpus_manifest(manifest_path) -> dict:
    """Validate fixture provenance and report whether the P0-5b human gate is met.

    Source validity is deterministic. Coaching labels are deliberately not:
    fixtures count toward the gate only after a named human approves assertions.
    """
    path = Path(manifest_path).resolve()
    base = path.parent
    source_errors = []
    label_errors = []
    fixtures = []

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "total": 0, "approved": 0, "minimum_approved": 0,
            "pending": 0, "source_errors": [f"manifest: {exc}"],
            "label_errors": [], "fixtures": [], "ready": False,
        }

    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        source_errors.append(
            f"schema_version: expected {CORPUS_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}")
    minimum = manifest.get("minimum_approved", 10)
    if not isinstance(minimum, int) or minimum < 1:
        source_errors.append("minimum_approved must be a positive integer")
        minimum = 10

    seen = set()
    for index, fixture in enumerate(manifest.get("fixtures") or []):
        fixture_id = str(fixture.get("id") or "").strip()
        prefix = fixture_id or f"fixture[{index}]"
        errors = []
        if not fixture_id:
            errors.append("missing id")
        elif fixture_id in seen:
            errors.append("duplicate id")
        seen.add(fixture_id)

        side = fixture.get("student_color")
        if side not in _FIXTURE_SIDES:
            errors.append(f"student_color must be one of {sorted(_FIXTURE_SIDES)}")
        status = fixture.get("review_status")
        if status not in _FIXTURE_STATUSES:
            errors.append(f"review_status must be one of {sorted(_FIXTURE_STATUSES)}")

        source_name = fixture.get("pgn") or ""
        if not source_name:
            errors.append("missing pgn path")
        source = (base / source_name).resolve()
        path_safe = True
        try:
            source.relative_to(base)
        except ValueError:
            errors.append("pgn path escapes the fixture directory")
            path_safe = False
        raw = None
        game = None
        if source_name and path_safe:
            try:
                raw = source.read_bytes()
                game = chess.pgn.read_game(io.StringIO(raw.decode("utf-8")))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"cannot read PGN: {exc}")
        if game is None and raw is not None:
            errors.append("PGN contains no game")
        if raw is not None:
            actual_hash = hashlib.sha256(raw).hexdigest()
            if actual_hash != fixture.get("sha256"):
                errors.append(f"sha256 mismatch: got {actual_hash}")
        coverage = None
        if game is not None:
            site = game.headers.get("Site", "")
            if fixture_id and fixture_id not in site:
                errors.append(f"Site header does not contain game id ({site!r})")
            student_name = (fixture.get("student_name") or "").strip().casefold()
            header_name = (game.headers.get(side.title(), "") or "").strip().casefold() \
                if side in _FIXTURE_SIDES else ""
            if not student_name or student_name != header_name:
                errors.append(
                    f"student_name does not match {side} header ({header_name!r})")
            coverage = _eval_coverage(game)
            minimum_coverage = float(fixture.get("minimum_eval_coverage", 0.95))
            if coverage < minimum_coverage:
                errors.append(
                    f"eval coverage {coverage:.1%} below required {minimum_coverage:.1%}")

        if status == "approved":
            if not fixture.get("reviewed_by") or not fixture.get("reviewed_on"):
                label_errors.append(f"{prefix}: approved fixture lacks reviewer/date")
            else:
                try:
                    date.fromisoformat(fixture["reviewed_on"])
                except (TypeError, ValueError):
                    label_errors.append(f"{prefix}: reviewed_on must be YYYY-MM-DD")
            expected = fixture.get("expected")
            if not isinstance(expected, dict) or not expected:
                label_errors.append(f"{prefix}: approved fixture lacks expected assertions")
            else:
                unknown = set(expected) - _ASSERTION_KEYS
                if unknown:
                    label_errors.append(f"{prefix}: unknown assertion keys {sorted(unknown)}")
                if expected.get("student_color") != side or expected.get("eval_source") != "pgn":
                    label_errors.append(
                        f"{prefix}: expected must pin student_color={side!r} and eval_source='pgn'")
                if not (set(expected) & _PEDAGOGICAL_ASSERTION_KEYS):
                    label_errors.append(f"{prefix}: approved fixture lacks a pedagogical assertion")

        source_errors.extend(f"{prefix}: {error}" for error in errors)
        fixtures.append({
            "id": fixture_id,
            "status": status,
            "eval_coverage": round(coverage, 4) if coverage is not None else None,
            "source_valid": not errors,
        })

    approved = sum(item["status"] == "approved" and item["source_valid"] for item in fixtures)
    pending = sum(item["status"] == "pending_human" for item in fixtures)
    return {
        "total": len(fixtures),
        "approved": approved,
        "minimum_approved": minimum,
        "pending": pending,
        "source_errors": source_errors,
        "label_errors": label_errors,
        "fixtures": fixtures,
        "ready": not source_errors and not label_errors and approved >= minimum,
    }


# --- CLI: run the real pipeline on a PGN and print the theme clustering ------

def _main(argv) -> int:
    if argv and argv[0] == "--check-corpus":
        if len(argv) != 2:
            print("usage: eval_coaching.py --check-corpus <manifest.json>")
            return 2
        report = validate_corpus_manifest(argv[1])
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    if not argv:
        print("usage: eval_coaching.py <pgn-path> [spec.json]\n"
              "       eval_coaching.py --check-corpus <manifest.json>")
        return 2
    from app.config import load_config
    from app.engines import EngineManager
    from app.review import review_game
    from dotenv import load_dotenv
    import sys

    pgn_path = argv[0]
    spec = json.load(open(argv[1])) if len(argv) > 1 else {}
    load_dotenv()
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        config = load_config()
    finally:
        sys.argv = saved_argv
    engines = EngineManager(config)
    engines.start()
    try:
        review = review_game(open(pgn_path, encoding="utf-8", errors="replace").read(), engines, config)
    finally:
        engines.quit()

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
