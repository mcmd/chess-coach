import os
from functools import lru_cache

import chess
import chess.engine

from app.trace import trace


class EngineStartError(RuntimeError):
    """Raised when a required engine cannot be started."""


class EngineManager:
    """Owns all engine processes AND all engine policy (limits, multipv, caching).

    Other modules should never touch `self.stockfish` directly — they call the
    methods here so depth/time policy lives in exactly one place.
    """

    QUICK_EVAL_TIME = 0.3      # shallow eval used for per-move game scanning
    DEEP_TIME_CAP = 4.0        # hard wall-clock cap for any single deep analysis
    THREAT_TIME_CAP = 1.5

    def __init__(self, config):
        self.config = config
        self.stockfish = None
        self.maia = None
        self.maia_prepared = None
        self.maia_error = None

    def start(self):
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            threads = max(1, min(8, os.cpu_count() or 4))
            self.stockfish.configure({"Threads": threads, "Hash": 512})
        except Exception as e:
            raise EngineStartError(
                f"Could not start Stockfish at '{self.config.stockfish_path}': {e}"
            )

        try:
            from maia2 import model, inference
            self.maia = model.from_pretrained(type="rapid", device="cpu")
            self.maia_prepared = inference.prepare()
        except Exception as e:
            self.maia = None
            self.maia_error = str(e)

    # --- STOCKFISH ---

    @lru_cache(maxsize=4000)
    def quick_eval(self, fen: str, time_limit: float = QUICK_EVAL_TIME):
        """Cached shallow evaluation. Returns (PovScore, best_move_uci | None)."""
        board = chess.Board(fen)
        info = self.stockfish.analyse(board, chess.engine.Limit(time=time_limit))
        pv = info.get("pv") or []
        score, best = info["score"], (pv[0].uci() if pv else None)
        trace(f"stockfish quick_eval({time_limit}s) -> {score.white()} best={best} (fen {fen.split(' ')[0][:24]}...)")
        return score, best

    def top_lines(self, board: chess.Board, multipv: int = 3, depth: int = 18,
                  time_cap: float = None):
        """Multipv analysis with a wall-clock safety cap. Returns a list of InfoDicts."""
        trace(f"stockfish top_lines(depth={depth}, multipv={multipv}, cap={time_cap or self.DEEP_TIME_CAP}s)")
        limit = chess.engine.Limit(depth=depth, time=time_cap or self.DEEP_TIME_CAP)
        infos = self.stockfish.analyse(board, limit, multipv=multipv)
        return infos if isinstance(infos, list) else [infos]

    def threat_probe(self, board: chess.Board, multipv: int = 2) -> list:
        """What does the opponent threaten if the side to move 'passes'?

        Null-move probe: give the opponent a free move and look at their best
        continuations. Only meaningful threats (mate, or a swing of >= 120cp
        versus the current evaluation) are reported.
        """
        if board.is_check():
            return []
        trace("stockfish threat_probe (null move: what does the opponent threaten if we pass?)")
        probe = board.copy(stack=False)
        try:
            probe.push(chess.Move.null())
            current_score, _ = self.quick_eval(board.fen())
            current_cp_opp = current_score.pov(probe.turn).score(mate_score=100000)
            depth = int(getattr(self.config, "threat_depth", 18))
            time_cap = float(getattr(self.config, "threat_time", self.THREAT_TIME_CAP))
            mpv = int(getattr(self.config, "threat_multipv", multipv))
            infos = self.top_lines(probe, multipv=mpv, depth=depth,
                                   time_cap=time_cap)
        except Exception:
            return []

        threats = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            score = info["score"].pov(probe.turn)
            line, tb = [], probe.copy(stack=False)
            for m in pv[:6]:
                line.append(tb.san(m))
                tb.push(m)
            if score.is_mate() and score.mate() > 0:
                threats.append({"threat": f"mate in {score.mate()} if you do nothing",
                                "threat_line": " ".join(line)})
            else:
                cp = score.score(mate_score=100000)
                if cp - current_cp_opp >= 120:
                    threats.append({"threat": f"opponent gains {(cp - current_cp_opp) / 100.0:+.1f} if you do nothing",
                                    "threat_line": " ".join(line)})
        return threats

    # --- MAIA-2 (human move prediction) ---

    def maia_predict(self, fen: str, elo_self: int = None, elo_oppo: int = None):
        """Maia-2 prediction. Returns {top_moves: [{move, probability}], win_probability..., raw_probs}
        or None if Maia is unavailable."""
        if not self.maia:
            return None
        from maia2 import inference
        elo_self = int(elo_self or self.config.user_elo)
        elo_oppo = int(elo_oppo or self.config.user_elo)
        try:
            move_probs, win_prob = inference.inference_each(
                self.maia, self.maia_prepared, fen, elo_self=elo_self, elo_oppo=elo_oppo
            )
        except Exception:
            return None
        if not move_probs:
            return None
        trace(f"maia-2 predict(elo {elo_self} vs {elo_oppo}) -> top={max(move_probs, key=move_probs.get)} "
              f"({move_probs[max(move_probs, key=move_probs.get)]:.0%})")

        board = chess.Board(fen)
        top = []
        for uci, prob in sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if mv in board.legal_moves:
                top.append({"move": board.san(mv), "probability": round(float(prob), 3)})
        return {
            "top_moves": top,
            "win_probability_for_side_to_move": round(float(win_prob), 3) if win_prob is not None else None,
            "raw_probs": {u: float(p) for u, p in move_probs.items()},
        }

    # --- LIFECYCLE ---

    def clear_cache(self):
        self.quick_eval.cache_clear()

    def quit(self):
        if self.stockfish:
            try:
                self.stockfish.quit()
            except Exception:
                pass
            self.stockfish = None
