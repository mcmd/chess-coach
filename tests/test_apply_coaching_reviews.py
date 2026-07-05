import json

import pytest

from scripts.apply_coaching_reviews import apply_review_decisions


def _write_review(tmp_path, *, slide_count=0, source_hash="abc"):
    manifest = {
        "fixtures": [{
            "id": "game1", "sha256": "abc", "student_color": "white",
            "review_status": "pending_human", "reviewed_by": None,
            "reviewed_on": None, "expected": {},
        }]
    }
    packet = {
        "fixture_id": "game1",
        "source_sha256": source_hash,
        "review": {"student_color": "white", "eval_source": "pgn"},
        "human_decision": {
            "approve_fixture": True,
            "reviewed_by": "Reviewer",
            "reviewed_on": "2026-06-21",
            "agenda_must_include": [14],
            "agenda_must_exclude": [],
            "diagnosis_by_ply": {"14": "ignored_threat"},
            "key_reason_tokens_by_ply": {"14": ["h3"]},
            "together_plies": [],
            "separate_plies": [],
            "slide_count": slide_count,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    proposal_dir = tmp_path / "proposals"
    proposal_dir.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (proposal_dir / "game1.proposal.json").write_text(json.dumps(packet), encoding="utf-8")
    return manifest_path, proposal_dir


def test_apply_review_decisions_promotes_explicit_human_label(tmp_path):
    manifest_path, proposal_dir = _write_review(tmp_path)

    assert apply_review_decisions(manifest_path, proposal_dir) == ["game1"]

    fixture = json.loads(manifest_path.read_text(encoding="utf-8"))["fixtures"][0]
    assert fixture["review_status"] == "approved"
    assert fixture["reviewed_by"] == "Reviewer"
    assert fixture["expected"]["student_color"] == "white"
    assert fixture["expected"]["slide_count"] == 0
    assert fixture["expected"]["agenda_must_include"] == [14]


@pytest.mark.parametrize("slide_count", [None, -1, "0"])
def test_apply_review_decisions_requires_explicit_valid_slide_count(tmp_path, slide_count):
    manifest_path, proposal_dir = _write_review(tmp_path, slide_count=slide_count)

    with pytest.raises(ValueError, match="slide_count"):
        apply_review_decisions(manifest_path, proposal_dir)


def test_apply_review_decisions_rejects_source_drift(tmp_path):
    manifest_path, proposal_dir = _write_review(tmp_path, source_hash="different")

    with pytest.raises(ValueError, match="source hash"):
        apply_review_decisions(manifest_path, proposal_dir)
