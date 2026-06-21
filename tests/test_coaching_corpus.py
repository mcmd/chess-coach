import json
import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import chess.pgn

from scripts.eval_coaching import check_review, review_proposal_packet, validate_corpus_manifest
from scripts.triage_coaching_corpus import analyse_side


CORPUS = Path(__file__).parent / "fixtures" / "coaching" / "manifest.json"


def test_p0_5b_sources_are_valid_but_human_gate_is_explicitly_pending():
    report = validate_corpus_manifest(CORPUS)

    assert report["source_errors"] == []
    assert report["label_errors"] == []
    assert report["total"] == 10
    assert report["pending"] == 10
    assert report["approved"] == 0
    assert report["ready"] is False
    assert all(item["source_valid"] for item in report["fixtures"])


def test_approved_fixture_requires_human_identity_date_and_assertions(tmp_path):
    manifest = json.loads(CORPUS.read_text(encoding="utf-8"))
    fixture = manifest["fixtures"][0]
    source = CORPUS.parent / fixture["pgn"]
    shutil.copyfile(source, tmp_path / fixture["pgn"])
    fixture["review_status"] = "approved"
    manifest["fixtures"] = [fixture]
    manifest["minimum_approved"] = 1
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_corpus_manifest(candidate)

    assert report["source_errors"] == []
    assert len(report["label_errors"]) == 2
    assert report["ready"] is False

    fixture["reviewed_by"] = "Human Reviewer"
    fixture["reviewed_on"] = "2026-06-21"
    fixture["expected"] = {"looks_good": True}
    candidate.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_corpus_manifest(candidate)
    assert any("unknown assertion" in error for error in report["label_errors"])
    assert any("lacks a pedagogical assertion" in error for error in report["label_errors"])


def test_human_selection_and_reason_assertions_are_machine_checkable():
    record = SimpleNamespace(
        ply=4, color="white", move_label="3. Bb5", san="Bb5", best_move_label="3. d4",
        drop_pct=18.0, quality="inaccuracy", phase="opening",
        recommended_diagnosis="priority_selection", lesson_objective="choose_priority",
        best_line_motifs=[], initial_question="What matters most?",
        explanation_primitives={
            "reasons": [{"key": True, "text": "controls d4", "squares": ["d4"],
                         "keywords": ["center"]}]},
    )
    review = SimpleNamespace(
        student_color="white", eval_source="pgn", agenda_plies=[4], records=[record],
        themes=[], slides=[{"start_move": 3}],
        record_at=lambda ply: record if ply == 4 else None,
    )
    spec = {
        "student_color": "white", "eval_source": "pgn",
        "agenda_must_include": [4], "agenda_must_exclude": [6],
        "diagnosis_by_ply": {"4": "priority_selection"},
        "key_reason_tokens_by_ply": {"4": ["d4", "center"]},
        "slide_count": 1,
    }

    assert check_review(review, spec) == []
    assert check_review(review, {"agenda_must_exclude": [4]})
    packet = review_proposal_packet(review)
    assert packet["candidates"][0]["recommended_diagnosis"] == "priority_selection"
    assert packet["notice"].startswith("PROPOSAL ONLY")


def test_triage_slide_phase_filter_matches_production_fixture_outcomes():
    def signal(game_id, student_white):
        path = CORPUS.parent / f"{game_id}.pgn"
        game = chess.pgn.read_game(io.StringIO(path.read_text(encoding="utf-8")))
        return analyse_side(game, student_white)["slide"]

    assert signal("JI23vCkO", True) is None
    assert signal("lXewiepo", True)["net_loss"] == 67.1
    assert signal("zKB100Ki", False)["start_move"] == 11
