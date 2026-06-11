"""Tests for the agent loop refactor: the spinner wraps MODEL calls only, so
tools run outside any live display and can show their own progress bars."""
from types import SimpleNamespace

from app.agent_loop import run_agent_turn
from app.state import SessionState, TimelineManager
from app.tools import CoachTools
from tests.conftest import FakeEngines


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send_message(self, payload):
        self.sent.append(payload)
        return self.responses.pop(0)


def _text_response(text):
    part = SimpleNamespace(text=text, function_call=None, functionCall=None)
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], function_calls=None)


def _tool_response(name, args=None):
    call = SimpleNamespace(name=name, args=args or {})
    return SimpleNamespace(candidates=[], function_calls=[call])


def test_agent_loop_executes_tool_and_renders_final_text(stub_config, capsys):
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    chat = FakeChat([
        _tool_response("get_game_review_summary"),  # no review loaded -> error dict, fine
        _text_response("Lesson text here."),
    ])
    run_agent_turn(chat, tools, "hello coach")
    assert len(chat.sent) == 2                # message + tool results
    out = capsys.readouterr().out
    assert "Lesson text here." in out


def test_agent_loop_progress_callback_reaches_review(stub_config):
    """The fetch tool passes the shared progress adapter into review_game."""
    from tests.test_review import PGN, _scripted_engines
    ticks = []
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), _scripted_engines(), stub_config,
                       on_progress=lambda done, total: ticks.append((done, total)))
    result = tools._review_and_load(PGN)
    assert "review_digest" in result
    assert ticks and ticks[-1] == (6, 6)      # one tick per ply, completed


# --- MOVE-STYLING NORMALIZATION (bold/bare moves -> consistent code spans) ---

def test_bold_move_with_number_and_glyph_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("You played **17. d5?!** here") == "You played `17. d5?!` here"


def test_bare_numbered_move_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("Black has just played 16... e5, challenging") == \
        "Black has just played `16... e5`, challenging"


def test_already_backticked_moves_untouched():
    from app.agent_loop import normalize_move_styling
    text = "The reply `16... e5` was strong, and `Nf3` too."
    assert normalize_move_styling(text) == text


def test_italic_move_becomes_code():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("consider *Nf3* instead") == "consider `Nf3` instead"


def test_code_fences_untouched():
    from app.agent_loop import normalize_move_styling
    board = "```text\n1. e4 e5 2. Nf3\n```"
    assert normalize_move_styling(board) == board


def test_ordinary_numbered_list_untouched():
    from app.agent_loop import normalize_move_styling
    text = "1. Look at your rooks. 2. Scan for checks."
    assert normalize_move_styling(text) == text


def test_castling_and_bold_emphasis_on_non_moves_untouched():
    from app.agent_loop import normalize_move_styling
    assert normalize_move_styling("**O-O** was right") == "`O-O` was right"
    assert normalize_move_styling("this is **really important**") == "this is **really important**"
