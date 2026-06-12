"""Batch pattern review: fast-scan knobs, cache, deterministic clustering,
the duck-typed BatchReview, dossier evolution, and the /batch command."""
import io
import json
import os

import chess
import chess.pgn

from app.batch import (BatchPattern, aggregate_patterns, batches_dir, game_key_for,
                       load_latest_batch, run_batch_review, scan_games,
                       select_representatives, split_games, _light_objective)
from app.memory import load_dossier, update_dossier_with_batch
from app.review import MoveRecord, review_game
from app.state import SessionState, TimelineManager
from tests.conftest import FakeEngines


# Three fool's-mate variants: mcmd (White) blunders on move 2, and the
# punisher is the forcing check Qh4# — a recurring "ignored direct threat"
# pattern across three distinct games.
_GAMES = [
    ("1. f3 e5 2. g4 Qh4#", "opp1"),
    ("1. f4 e6 2. g4 Qh4#", "opp2"),
    ("1. g4 e5 2. f3 Qh4#", "opp3"),
]


def _fixture():
    """Returns (pgn_text, engines) scripted so each game's move-2 blunder
    evaluates at -900 with best reply d8h4 (check)."""
    parts, evals, bests, multipv = [], {}, {}, {}
    for moves, opp in _GAMES:
        pgn = (f'[Event "T"]\n[Site "?"]\n[White "mcmd"]\n[Black "{opp}"]\n'
               f'[Result "0-1"]\n\n{moves} 0-1\n')
        parts.append(pgn)
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for i, node in enumerate(game.mainline()):
            board.push(node.move)
            if i == 2:  # after White's second move: the blunder
                evals[board.fen()] = -900
                bests[board.fen()] = "d8h4"
                multipv[board.fen()] = [("d8h4", 900)]
    return "\n".join(parts), FakeEngines(evals=evals, bests=bests, multipv=multipv)


# --- review_game fast-scan knobs ---

def test_scan_knobs_skip_maia_and_enrichment_and_record_punisher(stub_config):
    pgn, engines = _fixture()
    review = review_game(pgn, engines, stub_config, with_maia=False, enrich=False)
    assert engines.maia_calls == 0
    assert review.agenda_plies == []
    assert review.records[2].punisher_uci == "d8h4"
    assert review.records[2].quality == "blunder"


def test_scan_knob_defaults_keep_maia_on(stub_config):
    pgn, engines = _fixture()
    review_game(pgn, engines, stub_config, enrich=False)
    assert engines.maia_calls > 0


# --- game identity & multi-game parsing ---

def test_game_key_prefers_lichess_site_header():
    headers = {"Site": "https://lichess.org/2dtDqQgRy2OD"}
    assert game_key_for(headers, "e2e4 e7e5") == "2dtDqQgR"


def test_game_key_falls_back_to_movetext_hash():
    key = game_key_for({"Site": "?"}, "e2e4 e7e5")
    assert key.startswith("local-") and len(key) == len("local-") + 12
    assert key == game_key_for({}, "e2e4 e7e5")


def test_split_games_caps_and_reports_truncation():
    pgn, _ = _fixture()
    games, truncated = split_games(pgn, 50)
    assert len(games) == 3 and truncated is False
    games, truncated = split_games(pgn, 2)
    assert len(games) == 2 and truncated is True


# --- scan cache ---

def test_scan_cache_reused_on_second_run(stub_config, tmp_path):
    pgn, engines = _fixture()
    cache = str(tmp_path / "batches")
    scans, skipped, _ = scan_games(pgn, engines, stub_config, cache)
    assert len(scans) == 3 and skipped == 0
    fresh = FakeEngines()
    scans2, _, _ = scan_games(pgn, fresh, stub_config, cache)
    assert len(scans2) == 3
    assert fresh.quick_eval_calls == 0  # everything came from cache


def test_corrupt_cache_file_rescans(stub_config, tmp_path):
    pgn, engines = _fixture()
    cache = str(tmp_path / "batches")
    scan_games(pgn, engines, stub_config, cache)
    victim = os.listdir(os.path.join(cache, "scans"))[0]
    with open(os.path.join(cache, "scans", victim), "w") as f:
        f.write("{not json")
    fresh, fresh_engines = _fixture()
    scans, _, _ = scan_games(fresh, fresh_engines, stub_config, cache)
    assert len(scans) == 3
    assert fresh_engines.quick_eval_calls > 0  # the corrupted game was rescanned


def test_scan_skips_games_without_the_student(stub_config, tmp_path):
    pgn = ('[Event "T"]\n[Site "?"]\n[White "somebody"]\n[Black "else"]\n'
           '[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n')
    scans, skipped, _ = scan_games(pgn, FakeEngines(), stub_config, str(tmp_path / "b"))
    assert scans == [] and skipped == 1


# --- deterministic clustering ---

def _rec(**kw):
    base = dict(ply=2, number=2, color="white", san="g4", uci="g2g4",
                fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                eval_before_cp=0, eval_after_cp=-900, drop_pct=46.0, quality="blunder")
    base.update(kw)
    return MoveRecord(**base)


def test_light_objective_forcing_check_punisher():
    board = chess.Board()
    for san in ("f3", "e5", "g4"):
        board.push_san(san)
    rec = _rec(fen_after=board.fen(), punisher_uci="d8h4")
    assert _light_objective(rec) == "defend_against_direct_threat"


def test_light_objective_en_prise_when_punisher_takes_the_moved_piece():
    fen_after = "rnbqkbnr/pp1ppppp/2p5/3Q4/8/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
    rec = _rec(uci="d1d5", san="Qd5", fen_after=fen_after, punisher_uci="c6d5")
    assert _light_objective(rec) == "left_piece_en_prise"


def test_light_objective_capture_and_time_and_default():
    assert _light_objective(_rec(is_capture=True)) == "material_temptation_vs_position_priority"
    assert _light_objective(_rec(time_flag="hope_chess")) == "threat_scan_under_time_pressure"
    assert _light_objective(_rec()) == "priority_selection"


def _scan_dict(game_key, records):
    from dataclasses import asdict
    return {"game_key": game_key, "headers": {}, "student_color": "white",
            "records": [asdict(r) for r in records]}


def test_aggregate_requires_three_moments_across_two_games():
    a, b = _rec(time_flag="hope_chess", phase="middlegame"), _rec(time_flag="hope_chess", phase="middlegame")
    # 2 moments / 2 games: below the moment threshold
    assert aggregate_patterns([_scan_dict("g1", [a]), _scan_dict("g2", [b])]) == []
    # 3 moments but all in one game: below the game threshold
    assert aggregate_patterns([_scan_dict("g1", [a, a, a])]) == []
    # 3 moments / 2 games: a pattern
    patterns = aggregate_patterns([_scan_dict("g1", [a, a]), _scan_dict("g2", [b])])
    assert len(patterns) == 1
    assert patterns[0].pattern_id == "threat_scan_under_time_pressure:middlegame"
    assert patterns[0].count == 3 and patterns[0].games_seen == 2


def test_aggregate_ignores_opponent_moves_and_small_drops():
    student = _rec(time_flag="hope_chess", phase="opening")
    opponent = _rec(color="black", time_flag="hope_chess", phase="opening")
    small = _rec(drop_pct=5.0, quality="ok", time_flag="hope_chess", phase="opening")
    patterns = aggregate_patterns([
        _scan_dict("g1", [student, opponent, small]),
        _scan_dict("g2", [student, opponent, small]),
        _scan_dict("g3", [student]),
    ])
    assert len(patterns) == 1 and patterns[0].count == 3


def test_representatives_at_most_one_per_game():
    moments = [{"game_key": "g1", "drop_pct": 40.0, "move_label": "2. g4", "san": "g4", "fen_before": "x"},
               {"game_key": "g1", "drop_pct": 35.0, "move_label": "9. h4", "san": "h4", "fen_before": "y"},
               {"game_key": "g2", "drop_pct": 20.0, "move_label": "5. a4", "san": "a4", "fen_before": "z"}]
    pattern = BatchPattern(pattern_id="p", objective="o", phase="?", title="t", habit="h",
                           count=3, games_seen=2, total_drop=95.0, moments=moments)
    chosen = select_representatives(pattern, 3)
    assert [m["game_key"] for m in chosen] == ["g1", "g2"]
    assert chosen[0]["drop_pct"] == 40.0


# --- the full pipeline & BatchReview ---

def test_run_batch_review_end_to_end(stub_config):
    pgn, engines = _fixture()
    batch = run_batch_review(pgn, engines, stub_config)
    assert batch.games_analyzed == 3 and batch.games_skipped == 0
    assert batch.moments_found == 3
    assert len(batch.patterns) == 1
    p = batch.patterns[0]
    assert p.objective == "defend_against_direct_threat"
    assert p.count == 3 and p.games_seen == 3
    # one deep-verified representative per game
    assert len(batch.records) == 3 and len(batch.agenda) == 3
    for r in batch.records:
        assert r.is_agenda
        assert r.punishment_line == ["Qh4#"]
        assert r.recommended_diagnosis
        assert r.question_ladder
    digest = batch.digest()
    assert "BATCH TRAINING REPORT" in digest
    assert "RECURRING PATTERNS" in digest
    assert "[Pattern 1]" in digest and p.title.split(" (")[0] in digest
    assert "TRAINING PLAN PROTOCOL" in digest
    assert "Do NOT begin teaching a position until the student picks a pattern." in digest
    # report persisted next to the lesson file
    reports = [f for f in os.listdir(batches_dir(stub_config)) if f.endswith(".json")]
    assert any(batch.batch_id in f for f in reports)


def test_saved_report_reloads_without_rescanning(stub_config):
    pgn, engines = _fixture()
    batch = run_batch_review(pgn, engines, stub_config)
    loaded = load_latest_batch(stub_config)
    assert loaded.batch_id == batch.batch_id
    assert len(loaded.records) == 3
    assert loaded.patterns[0].count == 3
    assert loaded.records[0].question_ladder  # int keys restored after JSON round-trip
    assert 1 in loaded.records[0].question_ladder
    assert loaded.records[0].initial_question.startswith("White to move")
    assert "BATCH TRAINING REPORT" in loaded.digest()


def test_batch_lessons_use_their_own_resume_slot(stub_config):
    from app.tools import CoachTools
    pgn, engines = _fixture()
    batch = run_batch_review(pgn, engines, stub_config)
    state = SessionState()
    tools = CoachTools(state, TimelineManager(state), engines, stub_config)
    # a single-game lesson is mid-flight
    state.lesson_metadata = {"last_game_id": "someGame", "last_agenda_index": 1,
                             "lesson_status": "awaiting_answer"}
    state.review = batch
    rep = batch.records[0]
    result = tools.submit_diagnosis(rep.fen_before, rep.recommended_diagnosis, "x", "y")
    assert result["success"] is True
    meta = state.lesson_metadata
    # the batch slot moved...
    assert meta["batch_id"] == batch.batch_id
    assert meta["batch_agenda_index"] == 1
    assert meta["batch_lesson_status"] == "awaiting_answer" or meta["batch_lesson_status"] == "diagnosis_logged"
    # ...and the game slot survived untouched
    assert meta["last_game_id"] == "someGame"
    assert meta["last_agenda_index"] == 1
    assert meta["lesson_status"] == "awaiting_answer"


# --- dossier evolution ---

def _summary(batch_id, rate, include=True):
    patterns = []
    if include:
        patterns = [{"id": "defend_against_direct_threat:opening", "title": "T",
                     "count": int(rate * 10), "games_seen": 5, "habit": "H",
                     "rate": rate, "representative_positions": []}]
    return {"batch_id": batch_id, "games": 10, "patterns": patterns}


def test_dossier_batch_update_tracks_trend_and_status(stub_config):
    f = stub_config.dossier_file
    update_dossier_with_batch(f, _summary("b1", 1.4))
    d = load_dossier(f)
    p = d["recurring_patterns"][0]
    assert p["status"] == "active" and p["previous_rate"] is None and p["rate"] == 1.4
    assert len(d["batch_trends"]) == 1

    update_dossier_with_batch(f, _summary("b2", 0.5))  # rate fell >40%
    p = load_dossier(f)["recurring_patterns"][0]
    assert p["status"] == "improving" and p["previous_rate"] == 1.4

    update_dossier_with_batch(f, _summary("b3", 0, include=False))
    update_dossier_with_batch(f, _summary("b4", 0, include=False))
    d = load_dossier(f)
    assert d["recurring_patterns"][0]["status"] == "stale"
    assert len(d["batch_trends"]) == 4


def test_dossier_context_surfaces_batch_patterns(stub_config):
    from app.prompts import get_dossier_context
    update_dossier_with_batch(stub_config.dossier_file, _summary("b1", 1.4))
    ctx = get_dossier_context(stub_config.dossier_file)
    assert "Confirmed recurring patterns" in ctx and "T (14 moments across 5 games" in ctx
    update_dossier_with_batch(stub_config.dossier_file, _summary("b2", 0.5))
    assert "IMPROVING" in get_dossier_context(stub_config.dossier_file)


# --- /batch command ---

def _processor(stub_config, engines):
    from app.commands import CommandProcessor
    state = SessionState()
    return CommandProcessor(state, TimelineManager(state), engines, stub_config), state


def test_batch_command_missing_file(stub_config):
    proc, _ = _processor(stub_config, FakeEngines())
    result = proc.process("/batch C:/nope/missing.pgn")
    assert "Could not find" in result.message


def test_batch_command_no_arg_without_report(stub_config):
    proc, _ = _processor(stub_config, FakeEngines())
    result = proc.process("/batch")
    assert "No saved batch report" in result.message


def test_batch_command_end_to_end_and_reload_resume(stub_config, tmp_path):
    pgn, engines = _fixture()
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(pgn, encoding="utf-8")

    proc, state = _processor(stub_config, engines)
    result = proc.process(f"/batch {pgn_file}")
    assert "BATCH TRAINING REPORT" in result.prompt
    assert "Present the training plan now" in result.prompt
    assert getattr(state.review, "batch_id", None)
    assert state.lesson_metadata["batch_id"] == state.review.batch_id
    assert state.lesson_metadata["batch_lesson_status"] == "fresh_review"
    # board tokens minted for every representative
    assert all(r.board_token for r in state.review.records)
    # dossier got the distilled pattern
    assert load_dossier(stub_config.dossier_file)["recurring_patterns"]

    # mid-lesson resume on reload
    from app.memory import save_lesson_metadata
    meta = dict(state.lesson_metadata)
    meta.update({"batch_agenda_index": 1, "batch_lesson_status": "awaiting_answer"})
    save_lesson_metadata(stub_config.lesson_file, meta)
    proc2, state2 = _processor(stub_config, FakeEngines())
    from app.memory import load_lesson_metadata
    state2.lesson_metadata = load_lesson_metadata(stub_config.lesson_file)
    result2 = proc2.process("/batch")
    assert "RESUMING" in result2.prompt
    assert "position 2" in result2.prompt
    assert state2.lesson_metadata["batch_lesson_status"] == "awaiting_answer"


def test_batch_command_clears_previous_game_timeline(stub_config, tmp_path):
    from tests.test_commands import PGN as SINGLE_GAME_PGN
    pgn, engines = _fixture()
    pgn_file = tmp_path / "games.pgn"
    pgn_file.write_text(pgn, encoding="utf-8")

    proc, state = _processor(stub_config, engines)
    proc.process(SINGLE_GAME_PGN)
    assert state.loaded_moves

    proc.process(f"/batch {pgn_file}")
    assert getattr(state.review, "batch_id", None)
    assert state.loaded_moves == []
    assert state.move_index == 0
    assert state.board.fen() == chess.STARTING_FEN

    result = proc.process("/next")
    assert "no game is loaded" in result.message
