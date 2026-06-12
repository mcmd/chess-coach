import chess
import chess.engine

from app.engines import EngineManager


class _InfoEngine:
    def __init__(self):
        self.calls = []

    def analyse(self, board, limit, multipv=None):
        self.calls.append({"fen": board.fen(), "limit": limit, "multipv": multipv})
        move = next(iter(board.legal_moves))
        info = {"score": chess.engine.PovScore(chess.engine.Cp(300), board.turn), "pv": [move]}
        if multipv:
            return [info]
        return info


class _Cfg:
    threat_depth = 18
    threat_time = 4.0
    threat_multipv = 3


def test_threat_probe_uses_configured_depth_time_and_multipv():
    manager = EngineManager(_Cfg())
    manager.stockfish = _InfoEngine()

    manager.threat_probe(chess.Board())

    deep_call = manager.stockfish.calls[-1]
    assert deep_call["multipv"] == 3
    assert deep_call["limit"].depth == 18
    assert deep_call["limit"].time == 4.0
