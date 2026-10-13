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
        self.human_models = {}      # tier ("deep"/"scan") -> HumanModel (maia3)
        self.maia_error = None      # user-facing load-warning summary (None if a model loaded)

    def start(self):
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            threads = max(1, min(8, os.cpu_count() or 4))
            self.stockfish.configure({"Threads": threads, "Hash": 512})
        except Exception as e:
            # popen may have succeeded before configure() failed — don't leak that
            # live process when the caller bails on EngineStartError.
            if self.stockfish is not None:
                try:
                    self.stockfish.quit()
                except Exception:
                    pass
                self.stockfish = None
            raise EngineStartError(
                f"Could not start Stockfish at '{self.config.stockfish_path}': {e}"
            )

        self._start_human_models()

    # --- HUMAN MODEL (maia3) LIFECYCLE & POLICY ---

    def _make_human_model(self, spec: str, cache: dict):
        """Build (or reuse a cached) maia3 model for a spec. Tiers that name the
        same spec share one loaded instance."""
        if spec in cache:
            return cache[spec]
        from app.human_model import Maia3Model
        model = Maia3Model(
            spec,
            cache_dir=getattr(self.config, "maia3_cache_dir", None),
            device=getattr(self.config, "maia3_device", "cpu"),
            history_plies=getattr(self.config, "maia_history_plies", 12),
            local_files_only=getattr(self.config, "maia3_offline", False),
        )
        cache[spec] = model
        return model

    def _start_human_models(self):
        """Load the maia3 deep/scan tiers (sharing a checkpoint when the same
        spec). On failure, set `maia_error` and leave the model unavailable — the
        coach continues engine-only (no human-move lens), failing loud rather than
        silently degrading. (Maia-2 was removed; see issue #8 / plan P0-7.)"""
        deep_spec = getattr(self.config, "maia3_deep_model", "maia3-79m")
        scan_spec = getattr(self.config, "maia3_scan_model", "maia3-79m")

        cache, errors = {}, []
        for tier, spec in (("deep", deep_spec), ("scan", scan_spec)):
            try:
                self.human_models[tier] = self._make_human_model(spec, cache)
                trace(f"human model: {tier} tier -> {spec}")
            except Exception as e:
                errors.append(f"{tier} ({spec}): {e}")

        if not self.human_models:
            self.maia_error = "; ".join(errors) or "no human model available"
        elif errors:
            trace("human model load warnings: " + "; ".join(errors))

    def _resolve_tier(self, tier: str):
        """Pick the model for a tier, degrading deep->scan->None."""
        return (self.human_models.get(tier)
                or self.human_models.get("deep")
                or self.human_models.get("scan"))

    @property
    def maia(self):
        """Back-compat truthiness: is any human-move model available?"""
        return self._resolve_tier("deep")

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
                  time_cap: float = None, fresh_hash: bool = False):
        """Multipv analysis with a wall-clock safety cap. Returns a list of InfoDicts.
        `fresh_hash` searches as a new game, so earlier searches cannot colour the result."""
        trace(f"stockfish top_lines(depth={depth}, multipv={multipv}, cap={time_cap or self.DEEP_TIME_CAP}s)")
        limit = chess.engine.Limit(depth=depth, time=time_cap or self.DEEP_TIME_CAP)
        infos = self.stockfish.analyse(board, limit, multipv=multipv,
                                       **({"game": object()} if fresh_hash else {}))
        return infos if isinstance(infos, list) else [infos]

    def practical_eval(self, board: chess.Board, depth: int = 10, time_cap: float = 1.0):
        """Evaluation at roughly human calculation depth, from a cleared hash table (a new game).
        The shared hash otherwise carries deep refutations into shallow searches: right after the
        depth-20 agenda search, 31. Kh2 scored -7.46 at depth 10 instead of about -2 (2026-10-13)."""
        trace(f"stockfish practical_eval(depth={depth}, fresh hash)")
        info = self.stockfish.analyse(board, chess.engine.Limit(depth=depth, time=time_cap), game=object())
        return info["score"]

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

    # --- HUMAN MOVE PREDICTION (maia3) ---

    def human_predict(self, fen: str, *, elo_self: int = None, elo_oppo: int = None,
                      history_uci: list = None, tier: str = "deep"):
        """Run the human model for `tier` and return a `HumanPrediction` (rich
        object: full distribution + WDL when available), or None if no model is
        available or the position has no prediction. `history_uci` (the moves
        reaching `fen`) conditions the maia3 prediction."""
        model = self._resolve_tier(tier)
        if model is None:
            return None
        elo_self = int(elo_self or self.config.user_elo)
        elo_oppo = int(elo_oppo or self.config.user_elo)
        try:
            pred = model.predict(fen, elo_self=elo_self, elo_oppo=elo_oppo, history_uci=history_uci)
        except Exception:
            return None
        if pred is None or not pred.raw_probs:
            return None
        top_uci = max(pred.raw_probs, key=pred.raw_probs.get)
        trace(f"{pred.model_name} predict(elo {elo_self} vs {elo_oppo}, tier={tier}, "
              f"hist={'y' if pred.used_history else 'n'}) -> top={top_uci} ({pred.raw_probs[top_uci]:.0%})")
        return pred

    def maia_predict(self, fen: str, elo_self: int = None, elo_oppo: int = None,
                     history_uci: list = None, tier: str = "deep"):
        """Legacy contract over `human_predict`: returns a dict with the original
        keys (top_moves / win_probability_for_side_to_move / raw_probs) plus the
        additive WDL fields, or None. Existing callers need no changes."""
        pred = self.human_predict(fen, elo_self=elo_self, elo_oppo=elo_oppo,
                                  history_uci=history_uci, tier=tier)
        if pred is None:
            return None
        board = chess.Board(fen)
        top = []
        for uci, prob in sorted(pred.raw_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if mv in board.legal_moves:
                top.append({"move": board.san(mv), "probability": round(float(prob), 3)})
        return {
            "top_moves": top,
            "win_probability_for_side_to_move": round(float(pred.win_probability), 3)
                if pred.win_probability is not None else None,
            "raw_probs": {u: float(p) for u, p in pred.raw_probs.items()},
            "draw_probability": round(float(pred.draw_probability), 3)
                if pred.draw_probability is not None else None,
            "loss_probability": round(float(pred.loss_probability), 3)
                if pred.loss_probability is not None else None,
            "used_history": pred.used_history,
            "model_name": pred.model_name,
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
