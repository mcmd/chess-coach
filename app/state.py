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
    review: object = None                       # GameReview of the loaded game
    socratic_attempts: dict = field(default_factory=dict)  # fen -> attempt count
    lesson_metadata: dict = field(default_factory=dict)    # persisted resume/restart boundary
    current_game_id: str = None
    teaching_ledger: dict = field(default_factory=dict)    # P0-8 routing/planning observability
    pending_unmatched_review: dict = field(default_factory=dict)  # P0-4c: review awaiting POV/consent
    review_context: dict = field(default_factory=dict)      # ownership, adopted POV, dossier consent
    active_dossier_drill: dict = field(default_factory=dict)  # current /review drill, if any
    claim_gate_extra_fens: list[str] = field(default_factory=list)  # transient tool-returned line positions
    claim_gate_events: list[dict] = field(default_factory=list)  # structured L1/shadow observability
    # Board diagram theme (set from config at startup; defaults = lichess green).
    board_light_color: str = "#eeeed2"
    board_dark_color: str = "#769656"
    board_scale: int = 2

class TimelineManager:
    """Safely manages all physical board mutations, branching invariants, and PGN metadata."""
    def __init__(self, state):
        self.state = state
        self.pgn_nodes = []  # Original GameNodes, for clock data (None for manual moves)
        self.initial_fen = chess.STARTING_FEN

    def reset(self):
        """VCR rewind: jump to the start of the loaded game WITHOUT forgetting it.
        After reset(), step_forward() replays the loaded moves from move 1."""
        self.state.board = chess.Board(self.initial_fen)
        self.state.move_index = 0

    def clear(self):
        """Full wipe: forget the loaded game, diagnostics, and review entirely."""
        self.initial_fen = chess.STARTING_FEN
        self.state.board = chess.Board()
        self.state.loaded_moves = []
        self.state.move_index = 0
        self.state.diagnostics.clear()
        self.state.review = None
        self.state.socratic_attempts.clear()
        self.state.current_game_id = None
        self.state.pending_unmatched_review.clear()
        self.state.review_context.clear()
        self.state.active_dossier_drill.clear()
        self.state.claim_gate_extra_fens.clear()
        self.state.claim_gate_events.clear()
        self.pgn_nodes = []

    def load_game(self, game: chess.pgn.Game):
        """Loads a game and positions the board at the START so the student can
        step through it chronologically."""
        if game and len(list(game.mainline_moves())) > 0:
            self.initial_fen = game.board().fen()
            self.state.board = game.board()
            self.state.loaded_moves = list(game.mainline_moves())
            self.state.move_index = 0
            self.pgn_nodes = list(game.mainline())

    def go_to_ply(self, ply: int):
        """Jump so that the NEXT step_forward() plays the move at `ply`
        (i.e. the board shows the position BEFORE that move)."""
        if not (0 <= ply <= len(self.state.loaded_moves)):
            raise IndexError("Ply outside the loaded game.")
        self.reset()
        for _ in range(ply):
            self.state.board.push(self.state.loaded_moves[self.state.move_index])
            self.state.move_index += 1

    def play_san(self, move_san: str) -> str:
        """Plays a move, handling timeline truncation if the user branches."""
        move = self.state.board.parse_san(move_san)

        # Calculate the SAN string BEFORE mutating the physical board.
        clean_san = self.state.board.san(move)

        if self.state.move_index < len(self.state.loaded_moves):
            self.state.loaded_moves = self.state.loaded_moves[:self.state.move_index]
            self.pgn_nodes = self.pgn_nodes[:self.state.move_index]

        self.state.board.push(move)
        self.state.loaded_moves.append(move)
        self.pgn_nodes.append(None)  # Manual moves have no PGN node data
        self.state.move_index += 1

        return clean_san

    def step_forward(self) -> tuple[str, str]:
        """Steps forward and returns the SAN move AND the calculated clock data."""
        if self.state.move_index < len(self.state.loaded_moves):
            next_move = self.state.loaded_moves[self.state.move_index]
            move_san = self.state.board.san(next_move)

            # --- CLOCK EXTRACTION ---
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
        if self.state.move_index > 0 and self.state.board.move_stack:
            self.state.board.pop()
            self.state.move_index -= 1
        else:
            raise IndexError("No moves to undo.")
