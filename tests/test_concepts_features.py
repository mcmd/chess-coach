"""Tests for the second wave of chess concepts: loose pieces, tension,
overload, OCB, mobility, backward pawns, back-rank/luft, pawn breaks,
rook-file join, trapped pieces, rule of the square, wrong-colored bishop."""
import chess

from app.concepts import (
    _get_backward_pawns,
    _get_loose_pieces,
    _get_mobility,
    _get_overloaded_defenders,
    _get_pawn_breaks,
    _get_pawn_tension,
    _get_rooks_on_files,
    _get_trapped_pieces,
    _has_opposite_colored_bishops,
    _is_back_rank_vulnerable,
    _get_endgame_features,
    _wrong_bishop_warning,
    extract_board_features,
)


# --- 1. LOOSE PIECES (LPDO) ---

def test_loose_pieces_detects_undefended_only():
    # Black Qd8 has zero defenders (Kh8 not adjacent); White Nd2/Rd1 are king-defended.
    board = chess.Board("3q3k/8/8/8/8/8/3N4/2KR4 w - - 0 1")
    assert _get_loose_pieces(board, chess.BLACK) == ["q on d8"]
    assert _get_loose_pieces(board, chess.WHITE) == []


def test_loose_pieces_ignores_pawns():
    board = chess.Board()  # rooks in corners are "undefended" only via pawns? No: a1 rook defended by nothing? a2 pawn defends b3...
    # Starting position: a1 rook is defended by nothing (a2/b2 pawns don't defend a1, knight b1 doesn't).
    loose = _get_loose_pieces(board, chess.WHITE)
    assert all(not s.startswith("P") for s in loose)


# --- 2. PAWN TENSION ---

def test_pawn_tension_detected():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    assert _get_pawn_tension(board) == ["e4<->d5"]


def test_no_tension_in_starting_position():
    assert _get_pawn_tension(chess.Board()) == []


# --- 3. OVERLOADED DEFENDERS ---

def test_overloaded_sole_defender_of_two_attacked_rooks():
    # Nf3 is the only defender of both Re5 and Rd4, each attacked by a black rook.
    board = chess.Board("3rr2k/8/8/4R3/3R4/5N2/8/K7 w - - 0 1")
    overloaded = _get_overloaded_defenders(board, chess.WHITE)
    assert len(overloaded) == 1
    assert "N on f3 is the ONLY defender" in overloaded[0]
    assert "R on d4" in overloaded[0] and "R on e5" in overloaded[0]


# --- 4. OPPOSITE-COLORED BISHOPS ---

def test_opposite_colored_bishops_true():
    board = chess.Board("4k3/1b6/8/8/8/8/8/2B1K3 w - - 0 1")  # Bb7 light vs Bc1 dark
    assert _has_opposite_colored_bishops(board)


def test_same_colored_bishops_false():
    board = chess.Board("b3k3/8/8/8/8/8/8/4KB2 w - - 0 1")  # both light
    assert not _has_opposite_colored_bishops(board)


def test_two_bishops_one_side_is_not_ocb():
    assert not _has_opposite_colored_bishops(chess.Board())


# --- 5. MOBILITY ---

def test_mobility_counts_starting_position():
    mob = _get_mobility(chess.Board())
    assert mob["white_total_legal_moves"] == 20
    assert mob["black_total_legal_moves"] == 20
    assert "least_mobile_piece" in " ".join(mob.keys())


# --- 6. BACKWARD PAWNS ---

def test_backward_pawn_detected():
    # Black d6 pawn: its only neighbor (e5) is ahead of it, and the stop
    # square d5 is controlled by White's e4 pawn.
    board = chess.Board("4k3/8/3p4/4p3/4P3/8/8/4K3 b - - 0 1")
    assert _get_backward_pawns(board, chess.BLACK) == ["d6"]
    # White e4 has no neighbors at all -> isolated, not backward.
    assert _get_backward_pawns(board, chess.WHITE) == []


def test_supported_pawn_not_backward():
    # Black e5 has neighbor d6 level-or-behind it -> not backward.
    board = chess.Board("4k3/8/3p4/4p3/4P3/8/8/4K3 b - - 0 1")
    assert "e5" not in _get_backward_pawns(board, chess.BLACK)


# --- 7. BACK-RANK / LUFT ---

def test_back_rank_vulnerable_with_no_luft():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert _is_back_rank_vulnerable(board, chess.BLACK)
    assert not _is_back_rank_vulnerable(board, chess.WHITE)  # open squares in front


def test_luft_removes_back_rank_flag():
    board = chess.Board("6k1/5pp1/7p/8/8/8/8/R5K1 w - - 0 1")  # ...h6 played
    assert not _is_back_rank_vulnerable(board, chess.BLACK)


def test_no_heavy_pieces_no_back_rank_danger():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/B5K1 w - - 0 1")
    assert not _is_back_rank_vulnerable(board, chess.BLACK)


# --- 8. PAWN BREAKS ---

def test_pawn_break_via_double_step():
    # f2-f4 would attack the e5 pawn.
    board = chess.Board("4k3/8/8/4p3/8/8/5P2/4K3 w - - 0 1")
    assert "f2-f4" in _get_pawn_breaks(board, chess.WHITE)


def test_no_breaks_when_no_levers_exist():
    assert _get_pawn_breaks(chess.Board(), chess.WHITE) == []


# --- 9. ROOKS ON OPEN/HALF-OPEN FILES ---

def test_rook_on_open_file_join():
    board = chess.Board("r3k3/p7/8/8/8/8/P7/R3K2R w KQq - 0 1")
    on_open, on_half = _get_rooks_on_files(board, chess.WHITE,
                                           open_files=["h"], half_open_files=[])
    assert on_open == ["h1 (h-file)"]
    assert on_half == []


# --- 10. TRAPPED PIECES ---

def test_cornered_knight_is_trapped():
    # Nh8 is attacked by Kg7; both escapes (Nxf7, Ng6) lose material.
    board = chess.Board("7N/5pk1/8/8/8/8/8/K7 w - - 0 1")
    trapped = _get_trapped_pieces(board, chess.WHITE)
    assert any("N on h8" in t for t in trapped)


def test_safe_piece_not_trapped():
    board = chess.Board()
    assert _get_trapped_pieces(board, chess.WHITE) == []


# --- 11. RULE OF THE SQUARE ---

def test_king_outside_the_square_cannot_catch():
    board = chess.Board("4k3/8/8/P7/8/8/8/K7 w - - 0 1")  # Ke8 vs a5 pawn, White to move
    eg = _get_endgame_features(board)
    passer = eg["white_passed_pawn_details"][0]
    assert passer["enemy_king_inside_the_square"] is False


def test_king_inside_the_square_catches():
    board = chess.Board("3k4/8/8/P7/8/8/8/K7 b - - 0 1")  # Kd8, Black to move
    eg = _get_endgame_features(board)
    passer = eg["white_passed_pawn_details"][0]
    assert passer["enemy_king_inside_the_square"] is True


# --- 12. WRONG-COLORED BISHOP ---

def test_wrong_bishop_rook_pawn_draw_warning():
    # White: Kh1, Bh2 (dark), a2 pawn. Promotion corner a8 is light. Black Ka8.
    board = chess.Board("k7/8/8/8/8/8/P6B/7K w - - 0 1")
    warning = _wrong_bishop_warning(board)
    assert warning is not None
    assert "WRONG-COLORED BISHOP" in warning
    assert "a8" in warning


def test_right_bishop_no_warning():
    # Bishop on g2 is light-squared, matching the a8 corner -> winnable, no warning.
    board = chess.Board("k7/8/8/8/8/8/P5B1/7K w - - 0 1")
    assert _wrong_bishop_warning(board) is None


# --- INTEGRATION: lean packet discipline ---

def test_new_features_omitted_when_empty():
    features = extract_board_features(chess.Board())
    # Starting position has no tension, breaks, backward pawns, trapped or
    # overloaded pieces — the keys must be absent, not empty.
    assert "pawn_tension_points" not in features["pawn_structure"]
    assert "white_available_pawn_breaks" not in features["pawn_structure"]
    assert "white_backward_pawns" not in features["pawn_structure"]
    assert "trapped_white" not in features["piece_safety"]
    assert "overloaded_defenders_white" not in features["piece_safety"]
    assert "opposite_colored_bishops" not in features["material"]
    assert "back_rank_vulnerable_no_luft" not in features["king_safety"]["white"]
    # Mobility is always informative and always present.
    assert features["piece_activity"]["mobility"]["white_total_legal_moves"] == 20


def test_position_facts_include_new_concepts():
    from app.review import position_facts
    # Tension + a loose piece in one position: White Ra1 undefended, e4/d5 tension.
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/R3K3 w Q - 0 1")
    facts = position_facts(board)
    assert "pawn tension: e4<->d5" in facts
    assert "White loose (zero defenders): R on a1" in facts


def test_starting_position_not_back_rank_vulnerable():
    # No open file = no entry route for heavy pieces: the pattern is dormant.
    board = chess.Board()
    assert not _is_back_rank_vulnerable(board, chess.WHITE)
    assert not _is_back_rank_vulnerable(board, chess.BLACK)


# --- KING STATUS (castled vs lost rights — restored from old prompt as backend fact) ---

def test_king_status_distinguishes_castled_from_lost_rights():
    from app.concepts import _king_status
    assert _king_status(chess.Board(), chess.WHITE) == "can still castle"
    castled = chess.Board("rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 5")
    assert _king_status(castled, chess.WHITE).startswith("castled")
    assert _king_status(castled, chess.BLACK).startswith("castled")
    # King wandered to e2: rights gone but NOT castled.
    wandered = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPKPPP/RNBQ1BNR b kq - 1 2")
    assert "cannot castle" in _king_status(wandered, chess.WHITE)
