import chess
import chess.pgn
from dataclasses import dataclass, field

@dataclass
class SessionState:
    """The Single Source of Truth for the local application."""
    board: chess.Board = field(default_factory=chess.Board)
    loaded_moves: list[chess.Move] = field(default_factory=list)
    move_index: int = 0
    sparring_mode: bool = False
    diagnostics: list[dict] = field(default_factory=list)
    board_visuals: dict = field(default_factory=dict)
    board_counter: int = 0

class TimelineManager:
    """Safely manages all physical board mutations, branching invariants, and PGN metadata."""
    def __init__(self, state):
        self.state = state
        self.pgn_nodes =[]  # Explicitly track original GameNodes for clock data

    def reset(self):
        self.state.board.reset()
        self.state.loaded_moves =[]
        self.state.move_index = 0
        self.state.diagnostics.clear()
        self.pgn_nodes =[]

    def load_game(self, game: chess.pgn.Game):
        if game and len(list(game.mainline_moves())) > 0:
            self.state.board = game.end().board()
            self.state.loaded_moves = list(game.mainline_moves())
            self.state.move_index = len(self.state.loaded_moves)
            self.pgn_nodes = list(game.mainline())

    def play_san(self, move_san: str) -> str:
        """Plays a move, handling timeline truncation if the user branches."""
        move = self.state.board.parse_san(move_san)
        
        # 🐛 FIX: Calculate the SAN string BEFORE mutating the physical board!
        clean_san = self.state.board.san(move)
        
        if self.state.move_index < len(self.state.loaded_moves):
            self.state.loaded_moves = self.state.loaded_moves[:self.state.move_index]
            self.pgn_nodes = self.pgn_nodes[:self.state.move_index]
            
        self.state.board.push(move)
        self.state.loaded_moves.append(move)
        self.pgn_nodes.append(None)  # Manual moves have no PGN node data
        self.state.move_index += 1
        
        return clean_san # Return the safely calculated string!

    def step_forward(self) -> tuple[str, str]:
        """Steps forward and returns the SAN move AND the calculated clock data."""
        if self.state.move_index < len(self.state.loaded_moves):
            next_move = self.state.loaded_moves[self.state.move_index]
            move_san = self.state.board.san(next_move)
            
            # --- CLOCK EXTRACTION ENGINE ---
            time_spent_str = ""
            if self.pgn_nodes and self.state.move_index < len(self.pgn_nodes):
                current_node = self.pgn_nodes[self.state.move_index]
                
                if current_node and current_node.clock() is not None:
                    prev_clock = None
                    # We must look 2 half-moves back to find the SAME player's clock
                    if self.state.move_index >= 2:
                        prev_node = self.pgn_nodes[self.state.move_index - 2]
                        if prev_node and prev_node.clock() is not None:
                            prev_clock = prev_node.clock()
                    
                    if prev_clock is not None:
                        time_spent = prev_clock - current_node.clock()
                        clock_left = int(current_node.clock())
                        
                        if time_spent < 0:
                            time_spent_str = f" | Time spent: < 0s (gained from increment) | Clock remaining: {clock_left}s"
                        else:
                            time_spent_str = f" | Time spent: {int(time_spent)}s | Clock remaining: {clock_left}s"
                    else:
                        time_spent_str = " | Time spent: Unknown | Clock remaining: Unknown"

            self.state.board.push(next_move)
            self.state.move_index += 1
            return move_san, time_spent_str
            
        raise IndexError("End of timeline.")

    def step_backward(self):
        if self.state.move_index > 0:
            self.state.board.pop()
            self.state.move_index -= 1
        else:
            raise IndexError("No moves to undo.")