"""Coaching-quality regression harness — skeleton (plan P0-5a) + the P0-1a
verified-theming assertions it gates. Deterministic, no model (tier 1)."""
import chess

from app.review import GameReview, MoveRecord, _derive_themes
from scripts.eval_coaching import themes_of, theme_of, same_theme, check_review

START = chess.STARTING_FEN


def _rec(idx, san, diag, phase="middlegame", color="white"):
    return MoveRecord(
        ply=2 * idx, number=idx + 1, color=color, san=san, uci="a2a3",
        fen_before=START, fen_after=START, eval_before_cp=0, eval_after_cp=-300,
        drop_pct=20.0, quality="mistake", recommended_diagnosis=diag, phase=phase)


def _review(records):
    r = GameReview(headers={}, records=records, student_color="white",
                   agenda_plies=list(range(len(records))))
    r.themes = _derive_themes(r)
    return r


def test_derive_themes_separates_distinct_diagnosis_classes():
    # Two material grabs (bad_trade) + a defensive recapture (missed_defensive_resource):
    # the session's exact over-fit — they must NOT collapse into one theme.
    review = _review([
        _rec(0, "Nxb5", "bad_trade"),
        _rec(1, "Qxa7", "bad_trade"),
        _rec(2, "Kxg2", "missed_defensive_resource"),
    ])
    assert len(review.themes) == 2
    assert same_theme(review, 0, 1)            # the two grabs cluster together
    assert not same_theme(review, 0, 2)        # the recapture stands apart
    # confidence: repeated class is high, the singleton is low
    by_id = {t.id: t for t in review.themes}
    assert by_id["bad_trade:middlegame"].confidence == "high"
    assert by_id["missed_defensive_resource:middlegame"].confidence == "low"


def test_harness_check_review_flags_wrong_clustering():
    review = _review([
        _rec(0, "Nxb5", "bad_trade"),
        _rec(1, "Qxa7", "bad_trade"),
        _rec(2, "Kxg2", "missed_defensive_resource"),
    ])
    # The correct expectations pass cleanly.
    assert check_review(review, {"min_themes": 2,
                                 "separate_plies": [[0, 2], [1, 2]],
                                 "together_plies": [[0, 1]]}) == []
    # A spec asserting the grab and the recapture belong together must FAIL.
    assert check_review(review, {"together_plies": [[0, 2]]})


def test_phase_splits_same_diagnosis_into_separate_themes():
    # Same category in different phases is a different teachable thread (mirrors batch).
    review = _review([
        _rec(0, "Bg4", "priority_selection", phase="opening"),
        _rec(1, "Re1", "priority_selection", phase="middlegame"),
    ])
    assert set(themes_of(review)) == {"priority_selection:opening", "priority_selection:middlegame"}
    assert not same_theme(review, 0, 1)
    assert theme_of(review, 0) == "priority_selection:opening"
