import pytest
import chess
from app.state import SessionState, TimelineManager
from app.tools import CoachTools

class MockEngines:
    config = type('obj', (object,), {'user_elo': 1500})

@pytest.fixture
def setup():
    state = SessionState()
    state.board = chess.Board("rnb1kbnr/ppppqppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    state.sparring_mode = True
    return CoachTools(state, TimelineManager(state), MockEngines()), state

def test_logic_tutor_guards_illegal_move(setup):
    coach, state = setup
    res = coach.execute_coach_move("a4") 
    assert "error" in res
    assert "Logic Tutor Intervention" in res["error"]
    assert state.board.fen() == "rnb1kbnr/ppppqppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def test_logic_tutor_allows_legal_move(setup):
    coach, state = setup
    res = coach.execute_coach_move("Qe2") 
    assert res.get("success") is True

def test_timeline_truncation_on_branching(setup):
    coach, state = setup
    state.board.reset() 
    state.move_index = 0
    state.loaded_moves = []
    coach.timeline.pgn_nodes = [] 
    
    coach.timeline.play_san("e4")
    coach.timeline.play_san("e5")
    assert state.move_index == 2
    assert len(state.loaded_moves) == 2
    assert len(coach.timeline.pgn_nodes) == 2
    
    coach.timeline.step_backward()
    assert state.move_index == 1
    
    coach.timeline.play_san("c5") 
    
    assert state.move_index == 2
    assert len(state.loaded_moves) == 2
    assert len(coach.timeline.pgn_nodes) == 2
    assert state.board.piece_at(chess.C5) is not None
    assert state.board.piece_at(chess.E5) is None