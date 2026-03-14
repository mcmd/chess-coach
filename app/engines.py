import os
import chess
import chess.engine
import subprocess
from functools import lru_cache
from app.ui import console
import maia2
from maia2 import inference

class EngineManager:
    def __init__(self, config):
        self.config = config
        self.stockfish = None
        self.maia = None

    def start(self):
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            threads = max(1, min(8, os.cpu_count() or 4))
            self.stockfish.configure({"Threads": threads, "Hash": 512})
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Stockfish not found at {self.config.stockfish_path}.")
            exit(1)

        try:
            from maia2 import model
            # Load Maia-2 model (Unified Model)
            self.maia = model.from_pretrained(type="rapid", device="cpu")
            self.maia_prepared = inference.prepare()
        except Exception as e:
            self.maia = None
            console.print(f"[bold red]Error initializing Maia-2:[/bold red] {e}")

    @lru_cache(maxsize=2000)
    def get_cached_eval(self, fen: str, time_limit: float):
        board = chess.Board(fen)
        info = self.stockfish.analyse(board, chess.engine.Limit(time=time_limit))
        return info["score"]

    def eval_move(self, board, move, time_limit, turn_color):
        board.push(move)
        score_obj = self.get_cached_eval(board.fen(), time_limit)
        board.pop()
        return score_obj.pov(turn_color).score(mate_score=100000)
        
    def quit(self):
        if self.stockfish: self.stockfish.quit()
        