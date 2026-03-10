import os
import chess
import chess.engine
import subprocess
from functools import lru_cache
from app.ui import console

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
            rounded_elo = round(self.config.user_elo / 100) * 100
            maia_target = 1100 if rounded_elo <= 1300 else (1500 if rounded_elo <= 1700 else 1900)
            weights_file = os.path.join(self.config.maia_weights_dir, f"maia-{maia_target}.pb.gz")
            
            if os.path.exists(weights_file):
                self.maia = chess.engine.SimpleEngine.popen_uci(self.config.lc0_path, stderr=subprocess.DEVNULL)
                self.maia.configure({"WeightsFile": weights_file})
            else:
                self.maia = None
                console.print(f"[bold yellow]Warning: Maia weights not found at {weights_file}[/bold yellow]")
        except Exception as e:
            self.maia = None
            console.print(f"[bold red]Error initializing Maia:[/bold red] {e}")

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
        if self.maia: self.maia.quit()
        