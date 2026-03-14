import pytest
import chess
from app.concepts import (
    _get_pawn_data,
    _get_passed_pawns,
    _find_true_outposts,
    _get_piece_safety,
    _get_bad_bishops,
    extract_board_features
)

# --- 1. TEST PAWN DATA ---
def test_get_pawn_data_isolated():
    # White pawn on e4 with no d or f pawns.
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    data = _get_pawn_data(board, chess.WHITE)
    assert "e4" in data["isolated"]
    assert data["island_count"] == 1

def test_get_pawn_data_doubled():
    # White pawns on e2 and e4.
    board = chess.Board("4k3/8/8/8/4P3/8/4P3/4K3 w - - 0 1")
    data = _get_pawn_data(board, chess.WHITE)
    assert 'e' in data["doubled_files"]

# --- 2. TEST PASSED PAWNS ---
def test_get_passed_pawns_white_success():
    # White pawn on e4. Black has no pawns on d, e, f file above rank 4.
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    passed = _get_passed_pawns(board, chess.WHITE)
    assert "e4" in passed

def test_get_passed_pawns_white_blocked():
    # Black pawn on e5 blocks White e4.
    board = chess.Board("4k3/8/8/4p3/4P3/8/8/4K3 w - - 0 1")
    passed = _get_passed_pawns(board, chess.WHITE)
    assert "e4" not in passed

def test_get_passed_pawns_white_adjacent_blocked():
    # Black pawn on d5 protects the path.
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    passed = _get_passed_pawns(board, chess.WHITE)
    assert "e4" not in passed 

def test_get_passed_pawns_black_success():
     board = chess.Board("4k3/8/8/4p3/8/8/8/4K3 b - - 0 1") # Black pawn on e5 (rank 4)
     passed = _get_passed_pawns(board, chess.BLACK)
     assert "e5" in passed

# --- 3. TEST TRUE OUTPOSTS (The regression fix) ---
def test_find_true_outposts_success():
    # White Knight on d5 anchored by White Pawn on e4 and c4.
    # No black pawns on columns c and e above rank 3 (so cannot kick).
    board = chess.Board("r1bqk2r/pp1p1ppp/8/2pN4/2P1P3/8/PP1P1PPP/R1BQKBNR w KQkq - 0 1")
    outposts = _find_true_outposts(board, chess.WHITE)
    assert "d5" in outposts

def test_find_true_outposts_can_be_kicked():
    # Black pawn on e6 CAN kick a White Knight on d5.
    board = chess.Board("r1bqk1nr/pp1p1ppp/4p3/2pN4/2P1P3/8/PP1P1PPP/R1BQKBNR w KQkq - 0 1")
    outposts = _find_true_outposts(board, chess.WHITE)
    assert "d5" not in outposts

def test_find_true_outposts_asymmetrical_axis_check():
    # White Knight on c5. Anchored by White Pawn on b4.
    # No black pawns on columns b and d above rank 3.
    # This proves x/y axes (file/rank) are not inverted.
    board = chess.Board("r1bqk2r/p3pppp/8/2N5/1P2P3/8/P2P1PPP/R1BQKBNR w KQkq - 0 1")

    outposts = _find_true_outposts(board, chess.WHITE)
    assert "c5" in outposts


# --- 4. TEST PIECE SAFETY ---
def test_get_piece_safety_hanging():
    # White Queen on e4, undefended. Black Rook on e8 attacks it.
    board = chess.Board("4r3/8/8/8/4Q3/8/8/4K3 w - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    assert any("Q on e4" in s for s in safety["hanging"])

def test_get_piece_safety_weak_multi_trade():
    # White Queen on e4. 
    # 2 Attackers: Black Rook on e8, Black Bishop on f5.
    # 2 Defenders: White Rook on e1, White Knight on f2.
    # Triggers Condition D: num_atk == num_def and lowest_atk <= lowest_def (3 <= 3).
    board = chess.Board("4r1k1/8/8/5b2/4Q3/8/5N2/4R1K1 w - - 0 1")
    safety = _get_piece_safety(board, chess.WHITE)
    assert any("Q on e4" in s for s in safety["weakly_defended"])


# --- 5. TEST BAD BISHOPS ---
def test_get_bad_bishops_white():
    # FEN contains a Light-squared Bishop on g2. 
    # White's central pawns on d3 and e4 (both light squares) are physically blocked by Black pawns on d4 and e5.
    board = chess.Board("4k3/8/8/4p3/3pP3/3P4/6B1/4K3 w - - 0 1")
    bad = _get_bad_bishops(board, chess.WHITE)
    assert bad

