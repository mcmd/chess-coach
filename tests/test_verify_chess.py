import chess
import pytest

from scripts.verify_chess import board_from_pgn, rule_facts

# Lichess 2dtDqQgR: drawn by threefold repetition right after 31. Kh1.
PGN_2DT = ("1. d4 d5 2. Bf4 Bf5 3. Nc3 e6 4. Nb5 Na6 5. e3 c6 6. Nc3 Nb4 7. Bd3 Bxd3 8. cxd3 Nf6 9. a3 Na6 "
           "10. Nf3 Bd6 11. Bxd6 Qxd6 12. O-O O-O 13. Qe2 Ng4 14. b4 b5 15. e4 dxe4 16. dxe4 e5 17. d5 cxd5 "
           "18. exd5 Nc7 19. Rad1 Rfe8 20. Nxb5 Nxb5 21. Qxb5 e4 22. h3 exf3 23. hxg4 fxg2 24. Kxg2 Rab8 "
           "25. Qc5 Qg6 26. Qxa7 Qxg4+ 27. Kh1 Qh3+ 28. Kg1 Qg4+ 29. Kh1 Qf3+ 30. Kg1 Qg4+ 31. Kh1 1/2-1/2")


def test_game_history_shows_the_repetition_a_bare_fen_cannot():
    before_31 = board_from_pgn(PGN_2DT, "31.")
    assert before_31.fen() == "1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31"
    with_history = before_31.copy()
    with_history.push_san("Kh1")
    bare = chess.Board(before_31.fen())
    bare.push_san("Kh1")

    assert rule_facts(with_history)["threefold_repetition"] is True
    assert rule_facts(bare)["threefold_repetition"] is False
    kh2 = before_31.copy()
    kh2.push_san("Kh2")
    assert rule_facts(kh2)["threefold_repetition"] is False


def test_before_selects_whites_or_blacks_move_and_rejects_moves_the_game_lacks():
    before_black_30 = board_from_pgn(PGN_2DT, "30...")
    assert before_black_30.turn == chess.BLACK and before_black_30.peek().uci() == "h1g1"  # after 30. Kg1
    assert board_from_pgn(PGN_2DT, "1.").move_stack == []
    assert board_from_pgn(PGN_2DT, "31...").is_repetition(3)  # the final position
    for bad in ("32.", "31", "move 31"):
        with pytest.raises(ValueError):
            board_from_pgn(PGN_2DT, bad)
