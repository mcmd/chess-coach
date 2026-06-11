import chess

from app.pawn_story import (
    analyze_pawn_structure_evolution,
    file_status,
    king_shield_facts,
)
from app.review import GameReview, MoveRecord


M20 = "r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20"
M24 = "r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5Pp1/3R1RK1 w - - 0 24"
M24_AFTER = "r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5PK1/3R1R2 b - - 0 24"
M26 = "1r2r1k1/p4ppp/6q1/2QP4/1P4P1/P7/5PK1/3R1R2 w - - 3 26"


def _record(ply, label_san, before, after):
    return MoveRecord(
        ply=ply, number=(ply // 2) + 1, color="white", san=label_san,
        uci="0000", fen_before=before, fen_after=after,
        eval_before_cp=0, eval_after_cp=0, drop_pct=0, quality="ok",
    )


def _review():
    return GameReview(
        headers={}, student_color="white", agenda_plies=[],
        records=[
            _record(39, "Ne4", M20, M24),
            _record(47, "Kxg2", M24, M24_AFTER),
            _record(51, "Qxa7", M24_AFTER, M26),
        ],
    )


def test_pawn_story_does_not_claim_f_pawn_missing():
    story = analyze_pawn_structure_evolution(_review(), "white")
    assert "f2" in story["final_pawn_snapshot"]["pawns"]
    event_text = " ".join(e["safe_summary"] for e in story["events"]).lower()
    assert "f-pawn" not in event_text
    assert "f2 pawn was gone" not in event_text
    assert "your f-pawn was gone" in story["avoid_language"]


def test_g_file_not_called_open_when_pawns_on_g_file():
    board = chess.Board(M26)
    assert file_status(board, "g") == "blocked_by_pawns"
    story = analyze_pawn_structure_evolution(_review(), "white")
    g_status = story["final_pawn_snapshot"]["file_statuses"]["g"]
    assert g_status == "blocked_by_pawns"
    assert "the g-file was open" in story["avoid_language"]


def test_ne4_not_described_as_blockade():
    story = analyze_pawn_structure_evolution(_review(), "white")
    event_text = " ".join(e["safe_summary"] for e in story["events"]).lower()
    assert "blockade" not in event_text


def test_kingside_story_names_g4_pin_not_total_collapse():
    facts = king_shield_facts(chess.Board(M26), chess.WHITE)
    assert facts["king"] == "g2"
    assert "f2" in facts["friendly_pawns_near_king"]
    assert "g4" in facts["friendly_pawns_near_king"]
    assert "g4" in facts["advanced_shield_pawns"]
    assert "g4" in facts["pinned_shield_pawns"]
    assert "Qxg4+" in facts["forcing_threats"]

    story = analyze_pawn_structure_evolution(_review(), "white")
    text = " ".join(e["safe_summary"] for e in story["events"]).lower()
    assert "g4" in text
    assert "pinned" in text
    assert "collapse" not in text
    assert any("g4" in phrase and "hinge" in phrase for phrase in story["safe_language"])
    assert any("Qxg4+" in phrase for phrase in story["safe_language"])
