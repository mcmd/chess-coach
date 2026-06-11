"""Tests for online-database integration: opening deviation detection,
tablebase verification of agenda moments, tablebase-aware answer grading,
the enriched tablebase tool, caching, and offline degradation."""
import chess

from app import external
from app.review import review_game
from app.state import SessionState, TimelineManager
from app.tools import CoachTools
from tests.conftest import FakeEngines
from tests.test_review import PGN, _fens_of, _scripted_engines, _Cfg

ENDGAME_PGN = """[Event "Test"]
[White "mcmd"]
[Black "Opp"]
[SetUp "1"]
[FEN "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1"]
[Result "*"]

1. e4 Kxe4 *
"""


def _mk(name, white, draws, black, uci=None):
    return {"san": name, "uci": uci or "", "white": white, "draws": draws, "black": black}


# --- 1. OPENING DEVIATION DETECTION ---

def test_opening_deviation_found_and_book_tagged(monkeypatch):
    fens = _fens_of(PGN)
    masters = {
        fens[0]: {"opening": None,
                  "moves": [_mk("e4", 600, 300, 100), _mk("d4", 500, 200, 100)]},
        fens[1]: {"opening": {"name": "King's Pawn Game", "eco": "B00"},
                  "moves": [_mk("e5", 400, 150, 50), _mk("c5", 500, 200, 100)]},
        fens[2]: {"opening": {"name": "King's Pawn Game", "eco": "B00"},
                  "moves": [_mk("Nf3", 500, 200, 100), _mk("Bc4", 150, 30, 20)]},  # no Qh5
    }
    monkeypatch.setattr(external, "masters_moves", lambda fen: masters.get(fen))
    monkeypatch.setattr(external, "lichess_moves",
                        lambda fen, ratings, speeds="blitz,rapid":
                        {"moves": [_mk("Qh5", 30, 5, 25), _mk("Nf3", 50, 20, 20)]})

    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert review.opening_name == "King's Pawn Game"
    assert review.opening_eco == "B00"
    assert review.records[0].book_share is not None   # 1. e4 is book
    assert review.records[1].book_share is not None   # 1... e5 is book
    assert review.records[2].book_share is None       # 2. Qh5 left theory
    d = review.opening_deviation
    assert d is not None and d["label"] == "2. Qh5" and d["color"] == "white"
    digest = review.digest()
    assert "Opening report" in digest
    assert "left established theory at 2. Qh5" in digest
    assert "Nf3" in digest                            # theory moves shown
    assert "STUDENT'S" in digest                      # deviation attribution
    assert "Peers at the student's rating play" in digest


def test_theory_note_protects_book_moves_in_agenda():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    review.records[4].book_share = 0.31               # pretend the blunder was book
    assert "established master theory" in review.digest()
    assert "do NOT scold it" in review.digest()


def test_offline_review_has_no_opening_report_and_no_crash():
    # autouse no_network fixture: get_json returns None everywhere.
    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert review.opening_name is None
    assert review.opening_deviation is None
    assert "Opening report" not in review.digest()


def test_online_databases_can_be_disabled(monkeypatch):
    def boom(fen):
        raise AssertionError("network helper called despite use_online_databases=False")
    monkeypatch.setattr(external, "masters_moves", boom)
    monkeypatch.setattr(external, "tablebase_probe", boom)

    class OfflineCfg(_Cfg):
        use_online_databases = False

    review = review_game(PGN, _scripted_engines(), OfflineCfg())
    assert review.opening_deviation is None


# --- 2. TABLEBASE VERIFICATION OF AGENDA MOMENTS ---

def _endgame_engines():
    fens = _fens_of(ENDGAME_PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[1]] = -300   # after 1. e4?? — a clear mistake -> agenda
    evals[fens[2]] = -300
    return FakeEngines(evals=evals), fens


def test_tablebase_truth_attached_to_agenda_moment(monkeypatch):
    engines, fens = _endgame_engines()
    tb = {
        fens[0]: {"category": "draw", "moves": [
            {"san": "e3", "uci": "e2e3", "category": "draw"},
            {"san": "Kd1", "uci": "e1d1", "category": "draw"},
            {"san": "e4", "uci": "e2e4", "category": "win"},   # black wins after e4
        ]},
        fens[1]: {"category": "win"},  # black to move: proven win for black
    }
    monkeypatch.setattr(external, "tablebase_probe", lambda fen: tb.get(fen))

    review = review_game(ENDGAME_PGN, engines, _Cfg())
    assert review.agenda_plies == [0]
    note = review.records[0].tablebase_note
    assert note is not None
    assert "TABLEBASE TRUTH" in note
    assert "proven DRAW" in note and "proven LOSS" in note
    assert "2 of 3 legal moves preserved the draw" in note
    assert "TABLEBASE TRUTH" in review.digest()


# --- 3. TABLEBASE-AWARE ANSWER GRADING ---

def _endgame_tools(stub_config, monkeypatch, tb_data):
    monkeypatch.setattr(external, "tablebase_probe", lambda fen: tb_data.get(fen))
    state = SessionState()
    return CoachTools(state, TimelineManager(state), FakeEngines(), stub_config), state


def test_answer_that_throws_tablebase_draw_is_wrong(stub_config, monkeypatch):
    fen = "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1"
    tools, _ = _endgame_tools(stub_config, monkeypatch, {fen: {
        "category": "draw",
        "moves": [{"san": "e3", "uci": "e2e3", "category": "draw"},
                  {"san": "e4", "uci": "e2e4", "category": "win"}],
    }})
    result = tools.evaluate_student_answer(fen, "e4")
    assert result["verdict"] == "wrong"
    assert "turns a proven draw into a proven loss" in result["tablebase_verdict"]


def test_answer_that_keeps_tablebase_result_is_best(stub_config, monkeypatch):
    fen = "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1"
    tools, _ = _endgame_tools(stub_config, monkeypatch, {fen: {
        "category": "draw",
        "moves": [{"san": "e3", "uci": "e2e3", "category": "draw"},
                  {"san": "e4", "uci": "e2e4", "category": "win"}],
    }})
    result = tools.evaluate_student_answer(fen, "e3")
    assert result["verdict"] == "best"
    assert result["reveal_allowed"] is True
    assert "KEEPS the proven draw" in result["tablebase_verdict"]


# --- 4. RICHER TABLEBASE TOOL ---

def test_tablebase_tool_reports_margin_for_error(stub_config, monkeypatch):
    fen = "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1"
    tools, _ = _endgame_tools(stub_config, monkeypatch, {fen: {
        "category": "draw", "dtz": 0, "dtm": None,
        "moves": [{"san": "e3", "uci": "e2e3", "category": "draw"},
                  {"san": "Kd1", "uci": "e1d1", "category": "draw"},
                  {"san": "e4", "uci": "e2e4", "category": "win"}],
    }})
    result = tools.query_endgame_tablebase(fen)
    assert result["move_outcomes_for_side_to_move"] == {"winning": 0, "drawing": 2, "losing": 1}
    assert result["margin_for_error"] == "2 of 3 legal moves preserve the proven draw"
    assert result["best_move"] == "e3"


def test_tablebase_tool_degrades_gracefully_offline(stub_config):
    fen = "8/8/8/8/3k4/8/4P3/4K3 w - - 0 1"
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), FakeEngines(), stub_config)
    result = tools.query_endgame_tablebase(fen)
    assert "error" in result and "unavailable" in result["error"]


# --- CACHING & CATEGORY HELPERS ---

def test_get_json_caches_responses(monkeypatch):
    calls = []

    def fake_fetch(url, params):
        calls.append(url)
        return {"ok": True}

    monkeypatch.setattr(external, "_fetch_json", fake_fetch)
    external.clear_cache()
    key = (("fen", "test-fen"),)
    assert external._get_json_cached("http://example/x", key) == {"ok": True}
    assert external._get_json_cached("http://example/x", key) == {"ok": True}
    assert len(calls) == 1
    external.clear_cache()


def test_category_normalization_and_inversion():
    assert external.normalize_category("cursed-win") == "win"
    assert external.normalize_category("blessed-loss") == "loss"
    assert external.normalize_category("draw") == "draw"
    assert external.normalize_category("unknown") is None
    assert external.normalize_category(None) is None
    assert external.invert_category("win") == "loss"
    assert external.invert_category("draw") == "draw"
