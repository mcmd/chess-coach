from datetime import date

import chess
import chess.pgn


def sparring_pgn_from_board(board: chess.Board, player_name: str = None) -> str | None:
    """Build a PGN for the current sparring game.

    Sparring starts with the student as White in the current CLI flow. The PGN is
    intentionally ordinary so the existing review pipeline owns all analysis,
    agenda selection, and dossier writes.
    """
    if board is None or not board.move_stack:
        return None
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessCoach sparring"
    game.headers["Site"] = "local"
    game.headers["Date"] = date.today().isoformat().replace("-", ".")
    game.headers["Round"] = "-"
    game.headers["White"] = player_name or "Student"
    game.headers["Black"] = "ChessCoach"
    game.headers["Result"] = board.result(claim_draw=True)

    node = game
    replay = chess.Board()
    for move in board.move_stack:
        node = node.add_variation(move)
        replay.push(move)
    return str(game)
