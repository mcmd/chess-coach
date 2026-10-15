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
    # The avoid-claim is derived from the board (f2 is present), not hardcoded.
    assert any("f-pawn was gone (it is still on f2)" in a for a in story["avoid_language"])


def test_g_file_not_called_open_when_pawns_on_g_file():
    board = chess.Board(M26)
    assert file_status(board, "g") == "blocked_by_pawns"
    story = analyze_pawn_structure_evolution(_review(), "white")
    g_status = story["final_pawn_snapshot"]["file_statuses"]["g"]
    assert g_status == "blocked_by_pawns"
    assert any(a.startswith("the g-file was open") for a in story["avoid_language"])
    # No transcript residue: nothing about specific piece moves like Ne4.
    assert not any("Ne4" in a for a in story["avoid_language"])


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
    assert any("g4" in pair["vivid"] and "hinge" in pair["vivid"] for pair in story["safe_language"])
    assert any("Qxg4+" in pair["concrete"] for pair in story["safe_language"])


def test_safe_language_uses_vivid_concrete_pairs():
    story = analyze_pawn_structure_evolution(_review(), "white")
    assert story["safe_language"]
    assert all(set(pair) == {"vivid", "concrete"} for pair in story["safe_language"])
    assert any("pinned to the king" in pair["concrete"] for pair in story["safe_language"])


# 2dtDqQgR moves 15-17 (ply, SAN, FEN before, FEN after).
_PLIES_28_33 = [
    (28, "e4", "r4rk1/p4ppp/n1pqp3/1p1p4/1P1P2n1/P1NPPN2/4QPPP/R4RK1 w - - 0 15",
     "r4rk1/p4ppp/n1pqp3/1p1p4/1P1PP1n1/P1NP1N2/4QPPP/R4RK1 b - - 0 15"),
    (29, "dxe4", "r4rk1/p4ppp/n1pqp3/1p1p4/1P1PP1n1/P1NP1N2/4QPPP/R4RK1 b - - 0 15",
     "r4rk1/p4ppp/n1pqp3/1p6/1P1Pp1n1/P1NP1N2/4QPPP/R4RK1 w - - 0 16"),
    (30, "dxe4", "r4rk1/p4ppp/n1pqp3/1p6/1P1Pp1n1/P1NP1N2/4QPPP/R4RK1 w - - 0 16",
     "r4rk1/p4ppp/n1pqp3/1p6/1P1PP1n1/P1N2N2/4QPPP/R4RK1 b - - 0 16"),
    (31, "e5", "r4rk1/p4ppp/n1pqp3/1p6/1P1PP1n1/P1N2N2/4QPPP/R4RK1 b - - 0 16",
     "r4rk1/p4ppp/n1pq4/1p2p3/1P1PP1n1/P1N2N2/4QPPP/R4RK1 w - - 0 17"),
    (32, "d5", "r4rk1/p4ppp/n1pq4/1p2p3/1P1PP1n1/P1N2N2/4QPPP/R4RK1 w - - 0 17",
     "r4rk1/p4ppp/n1pq4/1p1Pp3/1P2P1n1/P1N2N2/4QPPP/R4RK1 b - - 0 17"),
    (33, "cxd5", "r4rk1/p4ppp/n1pq4/1p1Pp3/1P2P1n1/P1N2N2/4QPPP/R4RK1 b - - 0 17",
     "r4rk1/p4ppp/n2q4/1p1pp3/1P2P1n1/P1N2N2/4QPPP/R4RK1 w - - 0 18"),
]


def test_chronology_keeps_order_and_marks_a_reversal():
    # Router run 5kf7rqwj S23: the coach said 15...dxe4 "opened the e-file for you" and skipped
    # 16. dxe4, which blocked it again; the tool's events were right but unordered prose was not.
    records = [MoveRecord(ply=p, number=p // 2 + 1, color="white" if p % 2 == 0 else "black", san=san,
                          uci="0000", fen_before=b, fen_after=a, eval_before_cp=0, eval_after_cp=0,
                          drop_pct=0, quality="ok") for p, san, b, a in _PLIES_28_33]
    story = analyze_pawn_structure_evolution(
        GameReview(headers={}, student_color="white", agenda_plies=[], records=records), "white")
    chronology = story["chronology"]
    opened = next(i for i, s in enumerate(chronology) if s.startswith("15... dxe4") and "e-file" in s)
    blocked = next(i for i, s in enumerate(chronology) if s.startswith("16. dxe4") and "e-file" in s)
    assert opened < blocked
    assert "undoes the change at 15... dxe4" in chronology[blocked]
    assert "in this order" in story["narration_rule"]
