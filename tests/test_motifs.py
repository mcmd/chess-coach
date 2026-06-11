import chess

from app.motifs import detect_motifs_in_line


def _kinds(motifs):
    return {m["motif"] for m in motifs}


def test_knight_fork_detected():
    board = chess.Board("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Nc7+"])
    assert "knight_fork" in _kinds(motifs)
    detail = next(m["detail"] for m in motifs if m["motif"] == "knight_fork")
    assert "rook on a8" in detail and "king on e8" in detail


def test_pin_detected():
    board = chess.Board("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Bb5"])
    assert "pin" in _kinds(motifs)


def test_skewer_detected():
    board = chess.Board("r3k3/8/8/q7/8/8/8/1R4K1 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Ra1"])
    assert "skewer" in _kinds(motifs)


def test_back_rank_mate_detected():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    motifs = detect_motifs_in_line(board, ["Ra8#"])
    assert "back_rank_mate" in _kinds(motifs)


def test_quiet_line_has_no_motifs():
    motifs = detect_motifs_in_line(chess.Board(), ["e4", "e5", "Nf3", "Nc6"])
    assert motifs == []


def test_illegal_continuation_stops_cleanly():
    motifs = detect_motifs_in_line(chess.Board(), ["e4", "Ke2"])  # Ke2 illegal after e4 (black to move)
    assert isinstance(motifs, list)


# --- SECOND WAVE: discovered attack, double check, zwischenzug, smothered, promotion ---

def test_discovered_attack_detected():
    # Ne4 moves away, unveiling Re1's attack on the queen on e8.
    board = chess.Board("4q2k/8/8/8/4N3/8/8/4R1K1 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Nc3"])
    assert "discovered_attack" in _kinds(motifs)
    detail = next(m["detail"] for m in motifs if m["motif"] == "discovered_attack")
    assert "rook on e1" in detail and "queen on e8" in detail


def test_double_check_detected():
    # Nf6+ checks with the knight AND unveils the e1 rook: double check.
    board = chess.Board("4k3/8/8/8/4N3/8/8/4RK2 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Nf6+"])
    assert "double_check" in _kinds(motifs)


def test_zwischenzug_detected():
    # Black grabs the d4 knight; instead of recapturing at once, White throws
    # in Re8+ first and only then takes back on d4.
    board = chess.Board("3r2k1/8/8/8/3N4/8/8/3RR1K1 b - - 0 1")
    motifs = detect_motifs_in_line(board, ["Rxd4", "Re8+", "Kg7", "Rxd4"])
    assert "zwischenzug" in _kinds(motifs)
    zwischenzug = next(m for m in motifs if m["motif"] == "zwischenzug")
    assert zwischenzug["move"] == "Re8+"


def test_immediate_recapture_is_not_zwischenzug():
    board = chess.Board("3r2k1/8/8/8/3N4/8/8/3RR1K1 b - - 0 1")
    motifs = detect_motifs_in_line(board, ["Rxd4", "Rxd4"])
    assert "zwischenzug" not in _kinds(motifs)


def test_smothered_mate_detected():
    board = chess.Board("6rk/6pp/7N/8/8/8/8/6K1 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["Nf7#"])
    assert "smothered_mate" in _kinds(motifs)
    assert "checkmate" not in _kinds(motifs)  # the specific name wins


def test_promotion_detected():
    board = chess.Board("8/4P3/8/8/8/8/8/k1K5 w - - 0 1")
    motifs = detect_motifs_in_line(board, ["e8=Q"])
    assert "promotion" in _kinds(motifs)
