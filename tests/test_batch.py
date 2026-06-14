"""Batch pattern review: fast-scan knobs, cache, deterministic clustering,
the duck-typed BatchReview, dossier evolution, and the /batch command."""
import io
import json
import os

import chess
import chess.pgn

from dataclasses import asdict

from app.batch import (BatchPattern, BatchReview, aggregate_patterns, aggregate_slides,
                       batches_dir, game_key_for, load_latest_batch, run_batch_review,
                       scan_games, select_representatives, split_games, _light_objective)
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


def test_batch_reps_get_rating_laddered_aspirational(stub_config):
    """Regression (Phase 5, §8.6): batch representatives flow through
    _enrich_agenda, so the aspirational 'one band up' instinct is computed at
    user_elo + 350 and surfaces in the batch digest. Locks the reuse so a future
    refactor of the batch deep pass can't silently drop it."""
    pgn, engines = _fixture()
    target = stub_config.user_elo + 350  # _aspirational_instinct's mover_elo + 350
    # Give each rep's fen_before a maia distribution at the aspirational band so
    # the instinct resolves to a concrete move.
    for moves, opp in _GAMES:
        g = chess.pgn.read_game(io.StringIO(
            f'[White "mcmd"]\n[Black "{opp}"]\n\n{moves} 0-1'))
        board = g.board()
        for i, node in enumerate(g.mainline()):
            if i == 2:  # before pushing White's 2nd move: the blunder's fen_before
                legal = next(iter(board.legal_moves)).uci()
                engines.maia_by_elo[(board.fen(), target)] = {legal: 0.6}
                break
            board.push(node.move)
    batch = run_batch_review(pgn, engines, stub_config)
    assert any(r.aspirational and r.aspirational["elo"] == target for r in batch.records)
    digest = batch.digest()
    assert "aspirational instinct" in digest
    assert f"~{target} Elo" in digest


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


def _slide_summary(batch_id, rate, present=True):
    sig = None
    if present:
        sig = {"slide_games": round(rate * 10), "games_total": 10, "rate": rate,
               "baseline": 0.17, "above_baseline": rate >= 0.255, "markers": []}
    return {"batch_id": batch_id, "games": 10, "patterns": [], "slide_signature": sig}


def test_dossier_slide_trend_lifecycle(stub_config):
    f = stub_config.dossier_file
    update_dossier_with_batch(f, _slide_summary("b1", 0.4))
    t = load_dossier(f)["slide_trend"]
    assert t["status"] == "active" and t["rate"] == 0.4 and t["previous_rate"] is None

    update_dossier_with_batch(f, _slide_summary("b2", 0.2))  # rate fell >40%
    t = load_dossier(f)["slide_trend"]
    assert t["status"] == "improving" and t["previous_rate"] == 0.4

    update_dossier_with_batch(f, _slide_summary("b3", 0, present=False))
    update_dossier_with_batch(f, _slide_summary("b4", 0, present=False))
    assert load_dossier(f)["slide_trend"]["status"] == "stale"


def test_dossier_context_surfaces_slide_habit_alone(stub_config):
    # a slide habit with no clustered patterns and no single-game history must still
    # surface (it used to be suppressed by the "no historical data" guard).
    from app.prompts import get_dossier_context
    update_dossier_with_batch(stub_config.dossier_file, _slide_summary("b1", 0.4))
    ctx = get_dossier_context(stub_config.dossier_file)
    assert "Cross-game slide habit" in ctx and "4/10" in ctx


# --- cross-game slide habit (eval-only) ---

def _seg(net=30.0, onset="ahead", queens_on=True, clock_low=None,
         start=20, end=26, plies=4, max_single=10.0):
    return {"start_move": start, "end_move": end, "plies": plies,
            "start_winpct": 80.0, "end_winpct": 50.0, "net_loss": net,
            "max_single": max_single, "onset": onset, "queens_on": queens_on,
            "clock_low": clock_low}


def _slide_scan(key, slides, records=None):
    return {"game_key": key, "student_color": "white",
            "records": records or [], "slides": slides}


def test_aggregate_slides_below_gate_returns_none():
    # only one game has a slide -> below MIN_SLIDE_GAMES.
    scans = [_slide_scan("g1", [_seg()]), _slide_scan("g2", [])]
    assert aggregate_slides(scans) is None


def test_aggregate_slides_above_gate_reports_rate_and_no_markers_yet():
    scans = [_slide_scan("g1", [_seg()]), _slide_scan("g2", [_seg()]), _slide_scan("g3", [])]
    sig = aggregate_slides(scans)
    assert sig["slide_games"] == 2 and sig["games_total"] == 3
    assert sig["rate"] == 0.67 and sig["above_baseline"] is True
    # 2 slide games < SLIDE_MARKER_MIN_GAMES -> situational markers withheld.
    assert sig["markers"] == []


def test_aggregate_slides_surfaces_majority_co_markers():
    scans = [_slide_scan(f"g{i}", [_seg(onset="ahead", queens_on=True)]) for i in range(3)]
    sig = aggregate_slides(scans)
    assert any("AHEAD" in m for m in sig["markers"])
    assert any("QUEENS" in m for m in sig["markers"])


def test_aggregate_slides_withholds_split_co_marker():
    # 3 slide games but onset split -> no 2/3 majority -> no 'ahead' marker.
    scans = [_slide_scan("g1", [_seg(onset="ahead")]),
             _slide_scan("g2", [_seg(onset="worse")]),
             _slide_scan("g3", [_seg(onset="level")])]
    sig = aggregate_slides(scans)
    assert not any("AHEAD" in m for m in sig["markers"])


def test_aggregate_slides_clock_marker_ignores_unknown_clocks():
    # known clocks = 2 (both low), the None is excluded from the denominator.
    scans = [_slide_scan("g1", [_seg(clock_low=True, onset="level")]),
             _slide_scan("g2", [_seg(clock_low=True, onset="level")]),
             _slide_scan("g3", [_seg(clock_low=None, onset="level")])]
    sig = aggregate_slides(scans)
    assert any("LOW CLOCK" in m for m in sig["markers"])


def _mistake_scan(key, move_number, slides):
    r = MoveRecord(ply=2 * move_number, number=move_number, color="white", san="Bxh7",
                   uci="c2h7", fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                   eval_before_cp=0, eval_after_cp=-300, drop_pct=20.0,
                   quality="blunder", phase="middlegame", is_capture=True)
    return {"game_key": key, "student_color": "white",
            "records": [asdict(r)], "slides": slides}


def test_pattern_axis_execution_when_mistakes_inside_slides():
    # 3 games, each mistake at move 25 sits inside a 20-30 slide -> execution leak.
    scans = [_mistake_scan(f"g{i}", 25, [_seg(start=20, end=30)]) for i in range(3)]
    patterns = aggregate_patterns(scans)
    assert patterns and patterns[0].axis == "execution"
    assert patterns[0].in_slide_count == 3


def test_pattern_axis_knowledge_when_mistakes_outside_slides():
    scans = [_mistake_scan(f"g{i}", 25, []) for i in range(3)]
    patterns = aggregate_patterns(scans)
    assert patterns and patterns[0].axis == "knowledge"
    assert patterns[0].in_slide_count == 0


def test_batch_digest_renders_cross_game_slide_block():
    batch = BatchReview(
        batch_id="x", created="2026-06-14", games_analyzed=10, games_skipped=0,
        truncated=False, moments_found=0, patterns=[], strengths=[],
        slide_signature={"slide_games": 3, "games_total": 10, "rate": 0.3,
                         "baseline": 0.17, "above_baseline": True,
                         "markers": ["when AHEAD (consolidating a winning position)"],
                         "examples": [{"game_key": "g1", "start_move": 20, "end_move": 26,
                                       "net_loss": 40.0, "plies": 4}]})
    out = batch.digest()
    assert "CROSS-GAME SLIDE PATTERN" in out
    assert "3/10" in out and "AHEAD" in out


def test_enrich_reconciles_best_move_to_deep_pv(stub_config):
    """Bug #3: when the shallow scan best_move disagrees with the deep PV's first
    move, the explanation follows the deep line — so the taught move must too."""
    from app.review import _enrich_agenda, GameReview
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"   # after 1.e4 e5
    after = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2"  # 2.Ke2 (a blunder)
    r = MoveRecord(ply=2, number=2, color="white", san="Ke2", uci="e1e2",
                   fen_before=fen, fen_after=after,
                   eval_before_cp=80, eval_after_cp=-400, drop_pct=35.0,
                   quality="blunder", best_move_san="Nf3")   # shallow scan said Nf3
    review = GameReview(headers={}, records=[r], student_color="white",
                        agenda_plies=[0], white_elo=1150, black_elo=1150)
    engines = FakeEngines(multipv={fen: [("d2d4", 90)]})     # deep PV prefers d4
    _enrich_agenda(review, engines, online=False)
    assert r.best_line[0] == "d4"
    assert r.best_move_san == "d4"   # reconciled away from the shallow "Nf3"


def test_slide_above_baseline_hysteresis(stub_config):
    """Bug #4: the persisted 'ABOVE typical' flag turns ON at 1.5x baseline, OFF
    only below 1.2x, and holds in the dead-band (no flip-flop on eval noise)."""
    f = stub_config.dossier_file
    update_dossier_with_batch(f, _slide_summary("b1", 0.30))   # >= 0.255 -> ON
    assert load_dossier(f)["slide_trend"]["above_baseline"] is True
    update_dossier_with_batch(f, _slide_summary("b2", 0.22))   # dead-band -> HOLD ON
    assert load_dossier(f)["slide_trend"]["above_baseline"] is True
    update_dossier_with_batch(f, _slide_summary("b3", 0.18))   # < 0.204 -> OFF
    assert load_dossier(f)["slide_trend"]["above_baseline"] is False


def test_batch_progress_spans_whole_pgn(stub_config, tmp_path):
    """FR #7: one progress total for the whole PGN; done is monotonic (no per-game
    reset) and ends at the grand total."""
    pgn, engines = _fixture()   # 3 fresh games, each scanned
    calls = []
    scan_games(pgn, engines, stub_config, str(tmp_path / "b"),
               on_progress=lambda d, t: calls.append((d, t)))
    totals = {t for _, t in calls}
    assert len(totals) == 1                 # a single batch-wide total
    dones = [d for d, _ in calls]
    assert dones == sorted(dones)           # monotonic across game boundaries
    assert dones[-1] == totals.pop()        # finishes at 100%


def test_severity_label_thresholds():
    from app.batch import _severity_label
    assert _severity_label(33) == "a blunder"
    assert _severity_label(25) == "a mistake"
    assert _severity_label(15) == "an inaccuracy"
    assert _severity_label(5) == "a minor slip"


def _pat(pid, obj, phase, title, count, games, drop, moments):
    return BatchPattern(pattern_id=pid, objective=obj, phase=phase, title=title,
                        habit="build the habit", count=count, games_seen=games,
                        total_drop=drop, moments=moments, in_slide_count=0, axis="knowledge")


def test_digest_dual_ranking_severity_and_same_habit():
    """FR #5: lead with frequency AND impact, severity in plain words, demote the
    cumulative win% sum, and name a habit spanning two phases as one weakness."""
    p1 = _pat("priority_selection:middlegame", "priority_selection", "middlegame",
              "Urgent priority (middlegame)", 14, 8, 210.0,
              [{"game_key": "g1", "move_label": "22. Re1"}])
    p2 = _pat("left_piece_en_prise:middlegame", "left_piece_en_prise", "middlegame",
              "Piece en prise (middlegame)", 11, 7, 367.0,
              [{"game_key": "g2", "move_label": "18... Rad8"}])
    p3 = _pat("priority_selection:opening", "priority_selection", "opening",
              "Urgent priority (opening)", 6, 4, 113.0,
              [{"game_key": "g3", "move_label": "7... Bg4"}])
    batch = BatchReview(batch_id="t", created="2026-06-15", games_analyzed=14,
                        games_skipped=0, truncated=False, moments_found=31,
                        patterns=[p1, p2, p3], strengths=[])
    digest = batch.digest()
    # frequency leader is P1 (14 moments); damage leader is P2 (367, blunder-sized)
    assert "Most FREQUENT: Pattern 1" in digest
    assert "Most DAMAGING: Pattern 2" in digest
    assert "averages a blunder" in digest          # 367/11 = 33.4
    # games-affected lead + demoted cumulative
    assert "appears in 8 of 14 games" in digest
    assert "raw cumulative 210 win%" in digest
    # P1 and P3 share an objective -> framed as one weakness across phases
    assert "SAME HABIT, TWO PHASES" in digest
    assert "Pattern 1 and Pattern 3" in digest


def test_digest_renders_lichess_link_and_orientation():
    """FR #6: teaching-position header shows the Lichess link; the per-moment
    orientation states the move number and the opponent's last move."""
    r = MoveRecord(ply=42, number=22, color="white", san="Re1", uci="f1e1",
                   fen_before="2r4r/p1Bk3p/4bp2/8/4P1p1/P7/P1RNnPPP/5R1K w - - 3 22",
                   fen_after="2r4r/p1Bk3p/4bp2/8/4P1p1/P7/P1RN1PPP/4R2K b - - 4 22",
                   eval_before_cp=120, eval_after_cp=-200, drop_pct=30.0,
                   quality="blunder", prev_move_san="Kd7")
    p = _pat("x:middlegame", "x", "middlegame", "T", 3, 2, 60.0,
             [{"game_key": "qqJUqmuk", "move_label": "22. Re1"}])
    batch = BatchReview(batch_id="t", created="2026-06-15", games_analyzed=2,
                        games_skipped=0, truncated=False, moments_found=3,
                        patterns=[p], strengths=[], records=[r], rep_pattern_idx=[0],
                        rep_sources=[{"game_key": "qqJUqmuk", "white": "mcmd",
                                      "black": "antakyadurum",
                                      "site": "https://lichess.org/qqJUqmuk"}])
    digest = batch.digest()
    assert "https://lichess.org/qqJUqmuk" in digest
    assert "this is move 22 of the game" in digest
    assert "the opponent has just played 21... Kd7" in digest


def test_scan_captures_opponent_previous_move(stub_config):
    """FR #6: review_game records prev_move_san (the move into fen_before)."""
    pgn, engines = _fixture()
    review = review_game(pgn, engines, stub_config, with_maia=False, enrich=False)
    assert review.records[0].prev_move_san is None      # first move has no predecessor
    assert review.records[2].prev_move_san == review.records[1].san  # g4 follows e5


def test_slide_signature_survives_save_and_reload(stub_config):
    pgn, engines = _fixture()
    batch = run_batch_review(pgn, engines, stub_config)
    # inject a signature and a pattern axis, persist, reload.
    from app.batch import _save_report
    batch.slide_signature = {"slide_games": 2, "games_total": 3, "rate": 0.67,
                             "baseline": 0.17, "above_baseline": True,
                             "markers": [], "examples": []}
    batch.patterns[0].axis = "execution"
    batch.patterns[0].in_slide_count = 2
    _save_report(batches_dir(stub_config), batch)
    loaded = load_latest_batch(stub_config)
    assert loaded.slide_signature["slide_games"] == 2
    assert loaded.patterns[0].axis == "execution"
    assert loaded.patterns[0].in_slide_count == 2


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
