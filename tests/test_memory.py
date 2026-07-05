import json
from types import SimpleNamespace

from app.memory import load_dossier, record_session_takeaway, update_dossier, save_history


def test_dossier_aggregates_taxonomy_categories(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    diagnostics = [
        {"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "Count defenders."},
        {"fen": "f2", "error_category": "hanging_piece", "preventative_habit": "Count defenders."},
        {"fen": "f3", "error_category": "ignored_threat", "preventative_habit": "Ask what changed."},
    ]
    update_dossier(diagnostics, dossier_file)
    d = load_dossier(dossier_file)
    assert d["total_blunders_analyzed"] == 3
    assert d["error_frequencies"]["hanging_piece"] == 2
    assert d["error_frequencies"]["ignored_threat"] == 1
    assert len(d["recent_positions"]) == 3
    assert d["recent_positions"][0]["fen"] == "f3"  # most recent first

    # Aggregation must survive a second session (the F2 fix).
    update_dossier([{"fen": "f4", "error_category": "hanging_piece", "preventative_habit": "x"}], dossier_file)
    d = load_dossier(dossier_file)
    assert d["error_frequencies"]["hanging_piece"] == 3


def test_recent_positions_capped_at_20(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    diagnostics = [{"fen": f"f{i}", "error_category": "missed_tactic", "preventative_habit": "h"} for i in range(30)]
    update_dossier(diagnostics, dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["recent_positions"]) == 20


def test_load_dossier_missing_file_returns_defaults(tmp_path):
    d = load_dossier(str(tmp_path / "nope.json"))
    assert d["total_blunders_analyzed"] == 0
    assert d["recent_positions"] == []


class _FakeChat:
    def __init__(self, messages):
        self._messages = messages

    def get_history(self):
        return self._messages


def _msg(role, text):
    return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])


def test_save_history_strips_think_blocks(tmp_path):
    history_file = str(tmp_path / "hist.json")
    chat = _FakeChat([
        _msg("user", "hello"),
        _msg("model", "<think>secret scratchpad</think>The d4 pawn is undefended."),
        _msg("model", "<think>only scratchpad</think>"),
    ])
    save_history(chat, history_file)
    data = json.load(open(history_file))
    assert len(data) == 2  # think-only message dropped entirely
    assert data[1]["parts"][0]["text"] == "The d4 pawn is undefended."
    assert "secret" not in json.dumps(data)


def test_save_history_caps_length(tmp_path):
    history_file = str(tmp_path / "hist.json")
    chat = _FakeChat([_msg("user", f"msg {i}") for i in range(200)])
    save_history(chat, history_file)
    data = json.load(open(history_file))
    assert len(data) == 80
    assert data[-1]["parts"][0]["text"] == "msg 199"


def test_typicality_trend_recorded_and_capped(tmp_path):
    dossier_file = str(tmp_path / "dossier.json")
    update_dossier([
        {"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "h",
         "error_typicality": "typical_for_level"},
        {"fen": "f2", "error_category": "king_safety", "preventative_habit": "h",
         "error_typicality": "atypical_for_level"},
        {"fen": "f3", "error_category": "missed_tactic", "preventative_habit": "h"},
    ], dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["typicality_history"]) == 1
    entry = d["typicality_history"][0]
    assert entry["typical_errors"] == 1
    assert entry["atypical_errors"] == 1
    assert entry["total_diagnosed"] == 3
    assert d["recent_positions"][2]["error_typicality"] == "typical_for_level"

    # A second session appends a second entry.
    update_dossier([{"fen": "f4", "error_category": "hanging_piece", "preventative_habit": "h",
                     "error_typicality": "atypical_for_level"}], dossier_file)
    d = load_dossier(dossier_file)
    assert len(d["typicality_history"]) == 2


def test_dossier_context_shows_typicality_trend(tmp_path):
    from app.prompts import get_dossier_context
    dossier_file = str(tmp_path / "dossier.json")
    update_dossier([{"fen": "f1", "error_category": "hanging_piece", "preventative_habit": "h",
                     "error_typicality": "typical_for_level"}], dossier_file)
    context = get_dossier_context(dossier_file)
    assert "Error-typicality trend" in context
    assert "1 peer-common" in context


def test_session_takeaway_recorded_and_added_to_context(tmp_path):
    from app.prompts import get_dossier_context
    dossier_file = str(tmp_path / "dossier.json")
    record_session_takeaway(dossier_file, {
        "game_id": "g1",
        "theme_id": "ignored_threat:opening",
        "text": "Session takeaway: a single ignored threat moment in the opening: 1. e4.",
        "source": "review_theme.safe_summary",
    })
    record_session_takeaway(dossier_file, {
        "game_id": "g1",
        "theme_id": "ignored_threat:opening",
        "text": "Session takeaway: a single ignored threat moment in the opening: 1. e4.",
        "source": "review_theme.safe_summary",
    })

    dossier = load_dossier(dossier_file)
    assert len(dossier["session_takeaways"]) == 1
    assert dossier["session_takeaways"][0]["theme_id"] == "ignored_threat:opening"
    context = get_dossier_context(dossier_file)
    assert "Recent closing takeaways" in context
    assert "ignored threat moment" in context
