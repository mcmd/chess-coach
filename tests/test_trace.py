import pytest

from app import trace
from app.agent_loop import _summarize_result
from app.trace import fmt_args


@pytest.fixture(autouse=True)
def reset_trace_state():
    yield
    trace.set_enabled(False)


def test_trace_disabled_by_default(capsys):
    trace.trace("should not appear")
    assert capsys.readouterr().out == ""


def test_trace_prints_when_enabled(capsys):
    trace.set_enabled(True)
    trace.trace("stockfish quick_eval")
    assert "stockfish quick_eval" in capsys.readouterr().out


def test_suspended_silences_and_restores(capsys):
    trace.set_enabled(True)
    with trace.suspended():
        trace.trace("hidden bulk-scan line")
    trace.trace("visible again")
    out = capsys.readouterr().out
    assert "hidden bulk-scan line" not in out
    assert "visible again" in out


def test_fmt_args_truncates_long_values():
    text = fmt_args({"fen": "x" * 200, "depth": 18})
    assert "depth=18" in text
    assert len(text) <= 180


def test_summarize_result_prefers_interesting_keys():
    assert "verdict=wrong" in _summarize_result({"verdict": "wrong", "reveal_allowed": False})
    assert _summarize_result({"error": "boom"}).startswith("ERROR")
    assert _summarize_result({"a": 1, "b": 2}).startswith("keys:")
