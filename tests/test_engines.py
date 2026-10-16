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


def test_threat_lines_come_numbered_from_the_opponent_s_move():
    # Run s7writoz S3: the threat line "e4 Nxe4 Qg6 Qc2 Nxd5" was retold as "Nxe4 ... Qg6 ... Nxd5",
    # dropping White's 22. Qc2; unnumbered lines also replayed from the wrong side before (d00beb2).
    board = chess.Board("r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 3 20")
    probe = board.copy(stack=False)
    probe.push(chess.Move.null())
    line, temp = [], probe.copy(stack=False)
    for san in ("e4", "Nxe4", "Qg6", "Qc2", "Nxd5"):
        line.append(temp.parse_san(san))
        temp.push(line[-1])

    class _LineEngine(_InfoEngine):
        def analyse(self, board, limit, multipv=None, **kwargs):
            if not multipv:          # quick_eval of the current position
                return {"score": chess.engine.PovScore(chess.engine.Cp(0), board.turn), "pv": []}
            return [{"score": chess.engine.PovScore(chess.engine.Cp(500), board.turn), "pv": line}]

    manager = EngineManager(_Cfg())
    manager.stockfish = _LineEngine()
    threat = manager.threat_probe(board)[0]
    assert threat["threat_line"] == "e4 Nxe4 Qg6 Qc2 Nxd5"
    assert threat["numbered_line"] == "20... e4 21. Nxe4 Qg6 22. Qc2 Nxd5"


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
