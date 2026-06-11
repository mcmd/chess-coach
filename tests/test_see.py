"""Adversarial tests for the SEE-based piece-safety logic — including the
pinned-defender false negative found in the 2026-06-10 review."""
import chess

from app.concepts import _get_piece_safety, see_exchange_gain


def test_pinned_defender_means_hanging():
    # White Nd5 is attacked by Rd8; its only "defender" Bc4 is absolutely
    # pinned (Rc8 vs Kc1), so Rxd5 wins a clean knight. The old attacker-count
    # heuristic reported the knight as safe.
    board = chess.Board("2rr2k1/8/8/3N4/2B5/8/8/2K5 b - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    assert any("N on d5" in s for s in safety["hanging"])


def test_undefended_attacked_queen_is_hanging():
    board = chess.Board("4r3/8/8/8/4Q3/8/8/4K3 w - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    assert any("Q on e4" in s for s in safety["hanging"])


def test_defended_pawn_attacked_by_rook_is_safe():
    # Rd8 attacks the d3 pawn, but c2 defends it: Rxd3 cxd3 loses a rook for a pawn.
    board = chess.Board("3rk3/8/8/8/8/3P4/2P5/4K3 b - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    flagged = safety["hanging"] + safety["weakly_defended"]
    assert not any("d3" in s for s in flagged)


def test_queen_in_losing_exchange_is_weakly_defended():
    # Queen on e4 attacked by rook + bishop, defended by rook + knight:
    # Bxe4 wins the exchange sequence but not the full queen.
    board = chess.Board("4r1k1/8/8/5b2/4Q3/8/5N2/4R1K1 w - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    assert any("Q on e4" in s for s in safety["weakly_defended"] + safety["hanging"])


def test_equal_minor_exchange_not_flagged():
    # Knight on d4 defended by the e3 pawn, attacked by a knight: Nxd4 exd4 is
    # an equal trade — no material is won, so nothing should be flagged.
    board = chess.Board("4k3/8/2n5/8/3N4/4P3/8/4K3 b - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    flagged = safety["hanging"] + safety["weakly_defended"]
    assert not any("d4" in s for s in flagged)


def test_see_exchange_gain_simple_win():
    # Black to move, white knight on d5 undefended, Rd8 takes it: +3.
    board = chess.Board("3r2k1/8/8/3N4/8/8/8/2K5 b - - 0 1")
    assert see_exchange_gain(board, chess.D5) == 3


def test_see_exchange_gain_defended_target():
    # Black rook takes a pawn-defended knight: 3 (knight) - 5 (rook lost) -> stand pat 0.
    board = chess.Board("3r2k1/8/8/3N4/2P5/8/8/2K5 b - - 0 1")
    assert see_exchange_gain(board, chess.D5) == 0
