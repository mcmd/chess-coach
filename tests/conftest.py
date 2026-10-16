import chess
import chess.engine
import pytest

from app.config import AppConfig


class FakeEngines:
    """EngineManager double with scriptable evaluations. No processes, no network.

    evals:   {fen: cp}            quick_eval score, white POV
    bests:   {fen: uci}           quick_eval best move
    multipv: {fen: [(uci, cp)]}   top_lines entries, cp from the MOVER's POV
    maia:    {fen: {uci: prob}}   maia_predict distributions
    """

    def __init__(self, evals=None, bests=None, multipv=None, maia=None,
                 maia_win=None, maia_by_elo=None, threats=None, maia_wdl=None):
        self.evals = evals or {}
        self.bests = bests or {}
        self.multipv = multipv or {}
        self.maia_data = maia or {}
        self.maia_win = maia_win or {}        # fen -> win probability
        self.maia_wdl = maia_wdl or {}        # fen -> (win, draw, loss) fractions (maia3-style)
        self.maia_by_elo = maia_by_elo or {}  # (fen, elo_self) -> probs
        self.threats_data = threats or {}     # fen -> threat_probe result list
        self.restricted = {}                  # fen -> (uci, cp mover POV): best_restricted result
        self.stockfish = object()  # truthy: "engine available"
        self.maia = bool(self.maia_data) or bool(self.maia_by_elo)
        self.maia_error = None
        self.config = None
        self.quick_eval_calls = 0
        self.maia_calls = 0
        self.top_line_calls = []

    def quick_eval(self, fen, time_limit=0.3):
        self.quick_eval_calls += 1
        cp = self.evals.get(fen, 0)
        return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE), self.bests.get(fen)

    def top_lines(self, board, multipv=3, depth=18, time_cap=None):
        self.top_line_calls.append({"fen": board.fen(), "multipv": multipv,
                                    "depth": depth, "time_cap": time_cap})
        entries = self.multipv.get(board.fen())
        infos = []
        if entries:
            for uci, cp in entries[:multipv]:
                infos.append({"score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn),
                              "pv": [chess.Move.from_uci(uci)]})
        else:
            for i, m in enumerate(list(board.legal_moves)[:multipv]):
                infos.append({"score": chess.engine.PovScore(chess.engine.Cp(20 - 10 * i), board.turn),
                              "pv": [m]})
        return infos

    def best_restricted(self, board, moves, depth=18, time_cap=None):
        entry = self.restricted.get(board.fen())
        if not entry:
            return None
        uci, cp = entry
        return chess.engine.PovScore(chess.engine.Cp(cp), board.turn), chess.Move.from_uci(uci)

    def threat_probe(self, board, multipv=2):
        return self.threats_data.get(board.fen(), [])

    def maia_predict(self, fen, elo_self=None, elo_oppo=None, history_uci=None, tier="deep"):
        self.maia_calls += 1
        probs = None
        if elo_self is not None and (fen, elo_self) in self.maia_by_elo:
            probs = self.maia_by_elo[(fen, elo_self)]
        else:
            probs = self.maia_data.get(fen)
        if not probs:
            return None
        board = chess.Board(fen)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        wdl = self.maia_wdl.get(fen)
        return {
            "top_moves": [{"move": board.san(chess.Move.from_uci(u)), "probability": p} for u, p in top],
            "win_probability_for_side_to_move": (wdl[0] if wdl else self.maia_win.get(fen, 0.5)),
            "raw_probs": dict(probs),
            "draw_probability": (wdl[1] if wdl else None),
            "loss_probability": (wdl[2] if wdl else None),
            "used_history": bool(history_uci),
            "model_name": "fake",
        }

    def human_predict(self, fen, elo_self=None, elo_oppo=None, history_uci=None, tier="deep"):
        from app.human_model import HumanPrediction, top_moves_san
        d = self.maia_predict(fen, elo_self=elo_self, elo_oppo=elo_oppo,
                              history_uci=history_uci, tier=tier)
        if d is None:
            return None
        return HumanPrediction(
            raw_probs=d["raw_probs"],
            win_probability=d["win_probability_for_side_to_move"],
            draw_probability=d["draw_probability"],
            loss_probability=d["loss_probability"],
            top_moves=top_moves_san(d["raw_probs"], chess.Board(fen)),
            used_history=d["used_history"],
            model_name=d["model_name"],
        )

    def clear_cache(self):
        pass

    def quit(self):
        pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests must never hit the real Lichess APIs. get_json is the single
    chokepoint; individual tests monkeypatch the higher-level helpers
    (masters_moves / lichess_moves / tablebase_probe) to inject data."""
    from app import external
    monkeypatch.setattr(external, "get_json", lambda url, params=None: None)


@pytest.fixture
def fake_engines():
    return FakeEngines()


@pytest.fixture
def stub_config(tmp_path):
    return AppConfig(
        user_elo=1150,
        language="English",
        stockfish_path="stockfish",
        gemini_model="test-model",
        api_key="test-key",
        history_file=str(tmp_path / "history.json"),
        dossier_file=str(tmp_path / "dossier.json"),
        cli_history_file=str(tmp_path / "cli_history.txt"),
        lesson_file=str(tmp_path / "lesson_state.json"),
        player_name="mcmd",
        # The coach-only flow; router behaviour has its own tests (test_router.py, test_commands.py).
        router_model=None,
    )
