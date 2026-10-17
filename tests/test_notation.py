import chess

from app.notation import illegal_move_reason

MOVE_24 = "r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5Pp1/3R1RK1 w - - 0 24"


def test_an_illegal_move_gets_the_reason_the_board_gives():
    # Owner decision D5: the reason comes from the board (live runs said "the queen can't jump to h5").
    board = chess.Board(MOVE_24)
    assert illegal_move_reason(board, "Qh5") == "your own pawn on d5 is in the way of your queen on b5"
    assert illegal_move_reason(board, "Kh2") == "your king would be in check on h2"
    assert illegal_move_reason(board, "Nf3") == "you have no knight"
    assert illegal_move_reason(board, "Rfe1") is None  # legal
    pinned = chess.Board("4k3/8/8/4p3/7b/8/5P2/4K3 w - - 0 1")
    after_e4_e5 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    assert illegal_move_reason(after_e4_e5, "exd5") == "a pawn captures only onto an enemy piece, and d5 is empty"
    assert illegal_move_reason(chess.Board(), "exd5") == "none of your pawns can capture on d5"
    assert illegal_move_reason(pinned, "f4") == "your pawn on f2 is pinned to your king by the black bishop on h4"
    assert illegal_move_reason(chess.Board(), "O-O") == "there are pieces between the king and the rook"
    in_check = chess.Board("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1")
    assert illegal_move_reason(in_check, "Kd1").startswith("your king is in check")


def test_an_opponent_reply_is_described_as_theirs():
    probe = chess.Board(MOVE_24)
    probe.push(chess.Move.null())
    assert illegal_move_reason(probe, "Qh5", "Black's") == "Black's queen on d6 does not move like that"
