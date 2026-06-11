"""Regression tests for the invented-board-token bug observed in the first
live session: the model fabricated [[BOARD_xxxxxxxx]] IDs for agenda moments
because the digest gave it FENs without tokens."""
from app.agent_loop import substitute_board_tokens, strip_think_blocks
from app.review import review_game, attach_board_tokens
from app.state import SessionState
from tests.test_review import PGN, _scripted_engines, _Cfg


def test_agenda_moments_get_preminted_tokens():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    state = SessionState()
    attach_board_tokens(review, state, invert=False)
    record = review.agenda[0]
    assert record.board_token is not None
    assert record.board_token in state.board_visuals
    # The digest hands the model a ready-made token so it never invents one.
    assert record.board_token in review.digest()


def test_known_tokens_are_substituted():
    visuals = {"[[BOARD_a1b2c3d4]]": "```text\nDIAGRAM\n```"}
    text, scrubbed = substitute_board_tokens("Look: [[BOARD_a1b2c3d4]] here", visuals)
    assert "DIAGRAM" in text
    assert "[[BOARD_" not in text
    assert scrubbed == 0


def test_invented_tokens_are_scrubbed_not_shown_raw():
    visuals = {"[[BOARD_a1b2c3d4]]": "```text\nDIAGRAM\n```"}
    text, scrubbed = substitute_board_tokens(
        "Real: [[BOARD_a1b2c3d4]] and fake: [[BOARD_22340356]] and `[[BOARD_9f562cff]]`",
        visuals)
    assert "DIAGRAM" in text
    assert "[[BOARD_" not in text  # nothing raw leaks to the student
    assert scrubbed == 2
    assert "diagram unavailable" in text


def test_think_blocks_stripped():
    assert strip_think_blocks("<think>internal</think>Lesson text") == "Lesson text"
