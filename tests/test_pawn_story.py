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


def _game_records(sans, fen=chess.STARTING_FEN):
    """Records for real legal moves from `fen`, as review_game builds them."""
    board = chess.Board(fen)
    records = []
    for ply, san in enumerate(sans):
        move = board.parse_san(san)
        white = board.turn == chess.WHITE
        before, label = board.fen(), board.san(move)
        board.push(move)
        records.append(MoveRecord(ply=ply, number=board.fullmove_number - (0 if white else 1),
                                  color="white" if white else "black", san=label, uci=move.uci(),
                                  fen_before=before, fen_after=board.fen(), eval_before_cp=0,
                                  eval_after_cp=0, drop_pct=0, quality="ok"))
    return records


def _story(records, color):
    return analyze_pawn_structure_evolution(
        GameReview(headers={}, student_color="white", agenda_plies=[], records=records), color)


def _events(story, kind):
    return [e for e in story["events"] if e["event"] == kind]


def _2dt_records():
    from tests.test_verify_chess import PGN_2DT
    sans = [tok for tok in PGN_2DT.split() if not tok[0].isdigit()]
    return _game_records(sans)


def test_existing_passer_advances_without_a_creation_event_and_its_blocker_is_named():
    # §10.3: d5 -> d6 is the same passed pawn moving, not a new passer; passed is not free to run.
    story = _story(_game_records(["d6"], "6k1/3n4/8/3P4/8/8/8/6K1 w - - 0 1"), "white")
    assert not _events(story, "passed_pawn_created")
    [advanced] = _events(story, "passed_pawn_advanced")
    assert (advanced["facts"]["from"], advanced["facts"]["to"]) == ("d5", "d6")
    assert advanced["facts"]["blocked_by"] == "black knight on d7"
    assert "not a new passer" in advanced["safe_summary"]


def test_a_pawn_passed_by_an_opponent_capture_gets_the_event_at_that_move():
    # Black's b7 pawn captures away from the b-file, so a5 has no enemy pawn left ahead of it.
    records = _game_records(["Kh8", "Kg2", "bxc6"], "6k1/1p6/2N5/P7/8/8/8/6K1 b - - 0 1")
    [created] = _events(_story(records, "white"), "passed_pawn_created")
    assert (created["move"], created["facts"]["pawn"], created["facts"]["cause"]) == ("2... bxc6", "a5", "position_change")


def test_en_passant_keeps_the_capturer_identity_and_removes_the_captured_passer():
    records = _game_records(["d5", "exd6"], "6k1/3p4/8/4P3/8/8/8/6K1 b - - 0 1")
    white = _story(records, "white")
    [created] = _events(white, "passed_pawn_created")
    assert (created["move"], created["facts"]["pawn"]) == ("1... d5", "e5")
    [advanced] = _events(white, "passed_pawn_advanced")
    assert (advanced["facts"]["to"], advanced["facts"]["origin_square"]) == ("d6", "e5")
    assert white["final_pawn_snapshot"]["origins"] == {"d6": "e5"}

    black = _story(records, "black")
    assert _events(black, "passed_pawn_created")[0]["facts"]["pawn"] == "d5"
    [captured] = _events(black, "passed_pawn_captured")
    assert (captured["move"], captured["facts"]["pawn"], captured["facts"]["origin_square"]) == ("2. exd6", "d5", "d7")
    assert black["final_pawn_snapshot"]["origins"] == {}


def test_promotion_ends_the_pawn_identity():
    story = _story(_game_records(["b8=Q", "Kh7", "Qb1"], "6k1/1P6/8/8/8/8/8/6K1 w - - 0 1"), "white")
    [promoted] = _events(story, "passed_pawn_promoted")
    assert (promoted["facts"]["from"], promoted["facts"]["promoted_to"]) == ("b7", "queen")
    assert not _events(story, "passed_pawn_created") and not _events(story, "passed_pawn_advanced")
    assert story["final_pawn_snapshot"]["origins"] == {}


def test_a_first_move_file_change_is_kept():
    story = _story(_game_records(["exd5"], "6k1/8/8/3p4/4P3/8/8/6K1 w - - 0 1"), "white")
    files = {e["facts"]["file"]: e["facts"]["status"] for e in _events(story, "file_status_changed")}
    assert files == {"d": "half_open_for_black", "e": "open"}


def test_sparse_records_report_only_their_own_moves():
    records = _2dt_records()
    exf3 = records[43]
    assert exf3.move_label == "22... exf3"
    story = _story([records[14], exf3], "white")   # 8. cxd3, then a gap to 22... exf3
    later = [e for e in story["events"] if e["move"] == "22... exf3"]
    assert [(e["event"], e["facts"].get("file")) for e in later] == [("file_status_changed", "e")]


def test_empty_review_has_an_explicit_result():
    story = _story([], "white")
    assert story["status"] == "no_moves"
    assert story["events"] == [] and story["chronology"] == []


def test_2dt_pawn_story_matches_the_real_game():
    # Live runs n6bmuxpr/jwuw0nd6/bgvvfacf (S23-S24). The tool missed 1. d4 (first record), called
    # Black's e-pawn on f3 and g2 "advanced shield pawns" of the king on g8 (narrated as "advancing
    # their king shield to g2"), never said 24. Kxg2 captured the g2 passer, and repeated "advanced
    # pawn(s) g4" for 23... fxg2 and 24. Kxg2. Its d5 passer, a3/b4 passers and open e-file were right.
    records = _2dt_records()
    white = _story(records, "white")
    assert white["chronology"][0].startswith("1. d4: White's king shield")
    [d5] = [e for e in _events(white, "passed_pawn_created") if e["facts"]["pawn"] == "d5"]
    assert d5["move"] == "18. exd5" and d5["facts"]["is_isolated"]
    assert d5["facts"]["origin_square"] == "c2"
    assert d5["facts"]["blocked_by"] == "black queen on d6"
    assert sorted(e["facts"]["pawn"] for e in _events(white, "passed_pawn_created")
                  if e["move"] == "26. Qxa7") == ["a3", "b4"]
    assert any(c.startswith("22... exf3: The e-file is now open") for c in white["chronology"])
    shield = {e["move"]: e["safe_summary"] for e in _events(white, "king_shield_changed")}
    assert "the g2 pawn was captured" in shield["23... fxg2"]
    assert "24. Kxg2" not in shield
    assert "g4" in shield["25... Qg6"] and "pinned" in shield["25... Qg6"]

    black = _story(records, "black")
    black_shield = [e for e in _events(black, "king_shield_changed")]
    assert not any(p in e["facts"]["friendly_pawns_near_king"] for e in black_shield for p in ("f3", "g2"))
    [g2] = _events(black, "passed_pawn_captured")
    assert (g2["move"], g2["facts"]["pawn"], g2["facts"]["origin_square"]) == ("24. Kxg2", "g2", "e7")
