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


def test_lichess_user_games_pgn_requests_analysed_evals(monkeypatch):
    calls = []
    monkeypatch.setattr(
        external,
        "_fetch_text",
        lambda url, params=None, headers=None: calls.append((url, params, headers)) or "pgn",
    )

    assert external.lichess_user_games_pgn("mcmd", max_games=12) == "pgn"

    url, params, headers = calls[0]
    assert url.endswith("/mcmd")
    assert params["max"] == 12
    assert params["analysed"] == "true"
    assert params["evals"] == "true"
    assert params["clocks"] == "true"
    assert headers["Accept"] == "application/x-chess-pgn"


class _HttpResponse:
    def __init__(self, text, status_code=200, content_type="application/x-chess-pgn"):
        self.text = text
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {}


def test_lichess_game_pgn_uses_bulk_export_with_identifying_user_agent(monkeypatch):
    calls = []

    def post(url, params=None, data=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "data": data, "headers": headers})
        return _HttpResponse('[Event "casual rapid game"]\n\n1. d4 { [%eval 0.15] } 1-0\n')

    monkeypatch.setattr(external.requests, "post", post)
    assert external.lichess_game_pgn("2dtDqQgR").startswith('[Event "casual rapid game"]')
    call = calls[0]
    assert call["url"] == "https://lichess.org/api/games/export/_ids"
    assert call["data"] == "2dtDqQgR"
    assert call["params"]["evals"] == "true" and call["params"]["clocks"] == "true"
    assert "github.com/mcmd/chess-coach" in call["headers"]["User-Agent"]


def test_lichess_game_pgn_distinguishes_missing_game_from_non_pgn_page(monkeypatch):
    monkeypatch.setattr(external.requests, "post", lambda *a, **k: _HttpResponse(""))
    assert external.lichess_game_pgn("zzzzzzzz") == ""
    signup_page = _HttpResponse("<!DOCTYPE html><html>Sign up</html>", content_type="text/html; charset=utf-8")
    monkeypatch.setattr(external.requests, "post", lambda *a, **k: signup_page)
    assert external.lichess_game_pgn("2dtDqQgR") is None


def test_lichess_text_requests_send_identifying_user_agent(monkeypatch):
    seen = []

    def get(url, params=None, headers=None, timeout=None):
        seen.append(headers or {})
        return _HttpResponse('[Event "x"]\n\n1. e4 *\n')

    monkeypatch.setattr(external.requests, "get", get)
    external.lichess_user_games_pgn("mcmd", max_games=1)
    external._fetch_json("https://lichess.org/api/user/mcmd", {})
    assert all("github.com/mcmd/chess-coach" in h.get("User-Agent", "") for h in seen) and len(seen) == 2


def test_chesscom_user_games_pgn_reads_recent_archives(monkeypatch):
    data = {
        "archives": {"archives": ["https://api.chess.com/a/2026/05", "https://api.chess.com/a/2026/06"]},
        "https://api.chess.com/a/2026/06": {"games": [{"pgn": "newer"}, {"pgn": "newest"}]},
        "https://api.chess.com/a/2026/05": {"games": [{"pgn": "older"}]},
    }
    monkeypatch.setattr(
        external,
        "get_json",
        lambda url, params=None: data["archives"] if url.endswith("/archives") else data.get(url),
    )

    assert external.chesscom_user_games_pgn("mcmd", max_games=2) == "newer\n\nnewest"


def test_category_normalization_and_inversion():
    assert external.normalize_category("cursed-win") == "win"
    assert external.normalize_category("blessed-loss") == "loss"
    assert external.normalize_category("draw") == "draw"
    assert external.normalize_category("unknown") is None
    assert external.normalize_category(None) is None
    assert external.invert_category("win") == "loss"
    assert external.invert_category("draw") == "draw"


# --- TABLEBASE CATEGORY-FLIP AGENDA FORCING + ENTRY MARKER ---

def test_category_flip_forced_onto_agenda_despite_small_eval_drop(monkeypatch):
    fens = _fens_of(ENDGAME_PGN)
    # Evals all equal: NO move qualifies for the agenda by win% swing.
    engines = FakeEngines(evals={fen: 0 for fen in fens})
    tb = {
        fens[0]: {"category": "draw", "moves": [
            {"san": "e3", "uci": "e2e3", "category": "draw"},
            {"san": "Kd1", "uci": "e1d1", "category": "draw"},
            {"san": "e4", "uci": "e2e4", "category": "win"},
        ]},
        fens[1]: {"category": "win"},   # after 1. e4: proven win for Black = loss for the student
        fens[2]: {"category": "loss"},  # after Kxe4 (White to move, lost)
    }
    monkeypatch.setattr(external, "tablebase_probe", lambda fen: tb.get(fen))

    review = review_game(ENDGAME_PGN, engines, _Cfg())
    assert review.records[0].drop_pct < 10          # the eval lens saw nothing
    assert 0 in review.agenda_plies                 # the tablebase lens forced it
    assert "TABLEBASE TRUTH" in review.records[0].tablebase_note
    assert review.tablebase_entry is not None
    assert review.tablebase_entry["category"] == "draw"
    assert "Tablebase entry" in review.digest()
    assert "pure technique" in review.digest()


def test_opponent_category_flip_not_forced(monkeypatch):
    fens = _fens_of(ENDGAME_PGN)
    engines = FakeEngines(evals={fen: 0 for fen in fens})
    tb = {
        fens[0]: {"category": "draw", "moves": []},
        fens[1]: {"category": "draw"},   # 1. e4 keeps the draw (no student flip)
        fens[2]: {"category": "win"},    # Black's Kxe4 flips it — but Black isn't the student
    }
    monkeypatch.setattr(external, "tablebase_probe", lambda fen: tb.get(fen))
    review = review_game(ENDGAME_PGN, engines, _Cfg())
    assert review.agenda_plies == []                # student (White) flipped nothing


# --- OPENING DEVIATION AS A STRUCTURED AGENDA MOMENT ---

def _deviation_masters(fens):
    return {
        fens[0]: {"opening": None,
                  "moves": [_mk("e4", 600, 300, 100), _mk("d4", 500, 200, 100)]},
        fens[1]: {"opening": {"name": "King's Pawn Game", "eco": "B00"},
                  "moves": [_mk("e5", 400, 150, 50), _mk("c5", 500, 200, 100)]},
        fens[2]: {"opening": {"name": "King's Pawn Game", "eco": "B00"},
                  "moves": [_mk("Nf3", 500, 200, 100), _mk("Bc4", 150, 30, 20)]},  # no Qh5
    }


def test_costly_deviation_becomes_structured_agenda_moment(monkeypatch):
    fens = _fens_of(PGN)
    monkeypatch.setattr(external, "masters_moves", lambda fen: _deviation_masters(fens).get(fen))
    monkeypatch.setattr(external, "lichess_moves",
                        lambda fen, ratings, speeds="blitz,rapid":
                        {"moves": [_mk("Qh5", 30, 5, 25), _mk("Nf3", 50, 20, 20)]})
    # 2. Qh5 costs ~6 win% — too small for the swing agenda, enough to teach theory.
    evals = {fen: 0 for fen in fens}
    evals[fens[3]] = -70
    review = review_game(PGN, FakeEngines(evals=evals), _Cfg())

    record = review.records[2]
    assert 5 <= record.drop_pct < 10
    assert 2 in review.agenda_plies                       # forced despite small drop
    assert record.is_opening_deviation
    assert record.recommended_diagnosis == "opening_theory"
    assert record.lesson_objective == "opening_theory_knowledge"
    assert "theory" in record.question_ladder[1].lower()
    assert "scores" in record.opening_theory_brief         # W/D/L data, not just popularity
    digest = review.digest()
    assert "OPENING DEVIATION moment" in digest
    assert "study item" in digest
    assert "theory here: Nf3" in digest                    # withheld theory answer


def test_harmless_deviation_stays_out_of_agenda(monkeypatch):
    fens = _fens_of(PGN)
    monkeypatch.setattr(external, "masters_moves", lambda fen: _deviation_masters(fens).get(fen))
    monkeypatch.setattr(external, "lichess_moves",
                        lambda fen, ratings, speeds="blitz,rapid": None)
    review = review_game(PGN, _scripted_engines(), _Cfg())  # Qh5 drop = 0 here
    assert review.opening_deviation is not None
    assert 2 not in review.agenda_plies
    assert not review.records[2].is_opening_deviation
