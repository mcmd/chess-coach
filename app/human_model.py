"""Human-move model behind one capability interface.

The coach predicts "what a human at rating X is likely to play here" via a
`HumanModel`. One implementation:

- `Maia3Model` — maia3 (in-process / Route B): history-aware, with a WDL value
  head. Loaded at a configurable size; the same instance can serve multiple
  tiers (see `app.engines.EngineManager`).

`EngineManager` owns *which* model serves *which* call (tier policy) and adapts
`HumanPrediction` back into the legacy `maia_predict` dict so existing callers
are untouched. Heavy imports (torch / maia3) are deferred to construction time so
importing this module stays cheap and test-friendly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import chess


@dataclass
class HumanPrediction:
    """A human-model prediction for one position.

    `raw_probs` is the full distribution over legal moves ({uci: prob}); it is
    the source of truth. `top_moves` (SAN) is a convenience view. `draw/loss`
    are populated from the WDL value head. `used_history` records whether a
    non-empty game history actually informed the prediction.
    """
    raw_probs: dict[str, float]
    win_probability: float | None = None
    draw_probability: float | None = None
    loss_probability: float | None = None
    top_moves: list[dict] = field(default_factory=list)
    used_history: bool = False
    model_name: str | None = None


def top_moves_san(raw_probs: dict[str, float], board: chess.Board, k: int = 3) -> list[dict]:
    """Top-`k` legal moves as [{move: SAN, probability: float}], highest first."""
    out = []
    for uci, prob in sorted(raw_probs.items(), key=lambda kv: kv[1], reverse=True):
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if mv in board.legal_moves:
            out.append({"move": board.san(mv), "probability": float(prob)})
        if len(out) >= k:
            break
    return out


class Maia3Model:
    """maia3 in-process adapter (Route B).

    Reuses maia3's own loading, move vocabulary, board-mirroring and tokenisation
    (via a composed `Maia3UCIEngine`), but runs a single forward pass and reads
    the *full* policy softmax + the position value head — rather than maia3's
    engine path, which does two passes and truncates to MultiPV. Isolated here so
    upstream internal changes break only this class.
    """

    supports_history = True
    supports_wdl = True

    def __init__(self, spec: str, *, cache_dir: str | None = None,
                 device: str = "cpu", history_plies: int = 12,
                 local_files_only: bool = False):
        import torch  # noqa: F401  (validates torch present at construction)
        from maia3.uci import parse_args, Maia3UCIEngine
        from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask

        argv = ["--model", spec, "--device", device, "--temperature", "0"]
        if cache_dir:
            argv += ["--cache-dir", cache_dir]
        if local_files_only:
            # Use the local HF cache only — no per-boot Hugging Face phone-home
            # (maia3 passes this straight to hf_hub_download(local_files_only=...)).
            argv += ["--local-files-only"]
        cfg = parse_args(argv)
        # History accumulation mode on: predict() supplies the real prior positions.
        cfg.use_uci_history = True

        self._engine = Maia3UCIEngine(cfg)
        self._engine.ensure_model_loaded()   # resolves checkpoint (cached) + loads weights
        self.cfg = cfg
        self.device = device
        self.history_plies = max(1, int(history_plies))
        self.name = getattr(getattr(cfg, "model_spec", None), "name", None) or spec

        # Cache the upstream helpers we call on the hot path.
        self._tokenize_board = tokenize_board
        self._historical_tokens = get_historical_tokens
        self._legal_mask = get_legal_moves_mask

    def _build_history(self, board: chess.Board,
                       history_uci: list[str] | None) -> tuple[deque, bool]:
        """Build the deque of board tokenisations leading up to `board`. Returns
        (history, used_history). Falls back to the single current position
        (positional, no history) when no usable history is supplied."""
        hist: deque = deque(maxlen=self.cfg.history)
        if history_uci:
            plies = history_uci[-self.history_plies:]
            try:
                # Replay from the standard start so each intermediate position is
                # tokenised. (Custom-start games fall back to positional below.)
                tmp = chess.Board()
                tmp_moves = chess.Board()
                for u in history_uci:
                    tmp_moves.push(chess.Move.from_uci(u))
                if tmp_moves.board_fen() == board.board_fen():
                    replay = chess.Board()
                    skip = len(history_uci) - len(plies)
                    for i, u in enumerate(history_uci):
                        if i >= skip:
                            hist.append(self._tokenize_board(replay))
                        replay.push(chess.Move.from_uci(u))
                    hist.append(self._tokenize_board(board))
                    return hist, True
            except (ValueError, AssertionError):
                pass
            hist.clear()
        hist.append(self._tokenize_board(board))
        return hist, False

    def predict(self, fen: str, *, elo_self: int, elo_oppo: int,
                history_uci: list[str] | None = None) -> HumanPrediction | None:
        import torch

        board = chess.Board(fen)
        if board.is_game_over():
            return None
        self._engine.board = board  # _move_from_index reads turn/legality from here

        legal_mask = self._legal_mask(board, self._engine.all_moves_dict)
        if not bool(legal_mask.any()):
            return None

        history, used_history = self._build_history(board, history_uci)
        tokens = self._historical_tokens(
            history, self.cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
        tokens = tokens.unsqueeze(0).to(self.device)
        self_elos = torch.tensor([int(elo_self)], dtype=torch.long, device=self.device)
        oppo_elos = torch.tensor([int(elo_oppo)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits_move, logits_value, _ = self._engine.model(tokens, self_elos, oppo_elos)

        mask = legal_mask.to(self.device)
        logits = logits_move[0].float().masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)

        # Map every legal index back to its (un-mirrored) UCI via the engine helper.
        raw: dict[str, float] = {}
        for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            move = self._engine._move_from_index(idx)
            if move is not None:
                raw[move.uci()] = float(probs[idx].item())
        if not raw:
            return None

        # Position value head → WDL (permille, side-to-move POV) → fractions.
        from maia3.uci import wdl_from_value_logits
        win, draw, loss = wdl_from_value_logits(logits_value[0])
        win_f, draw_f, loss_f = win / 1000.0, draw / 1000.0, loss / 1000.0

        return HumanPrediction(
            raw_probs=raw,
            win_probability=win_f,
            draw_probability=draw_f,
            loss_probability=loss_f,
            top_moves=top_moves_san(raw, board),
            used_history=used_history,
            model_name=self.name,
        )
