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


class _MaiaCfg:
    maia3_deep_model = "maia3-79m"
    maia3_scan_model = "maia3-79m"
    maia3_cache_dir = None
    maia3_device = "cpu"
    maia3_offline = False
    maia_history_plies = 12


def test_maia3_load_failure_degrades_to_engine_only(monkeypatch):
    # P0-7: a Maia-3 load failure must NOT crash and must NOT silently fall back to
    # Maia-2 (removed) — the coach runs engine-only with maia_error set.
    import app.human_model as hm

    def _boom(*a, **k):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(hm, "Maia3Model", _boom)
    manager = EngineManager(_MaiaCfg())
    manager._start_human_models()

    assert manager.maia is None
    assert manager.human_models == {}
    assert manager.maia_error and "checkpoint unavailable" in manager.maia_error
    assert manager.human_predict(chess.STARTING_FEN) is None  # no human lens, no crash
