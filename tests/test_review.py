import io

import chess
import chess.pgn

from app.review import (
    cp_to_win_pct, classify_move, time_flag_for, review_game,
    _initial_question,
)
from tests.conftest import FakeEngines


# --- UNIT: win% model and classification thresholds ---

def test_cp_to_win_pct_anchors():
    assert abs(cp_to_win_pct(0) - 50.0) < 0.01
    assert cp_to_win_pct(300) > 70
    assert cp_to_win_pct(-300) < 30
    # Saturation: -2 -> -5 in a lost position is NOT a 30-point swing.
    assert cp_to_win_pct(-200) - cp_to_win_pct(-500) < 25


def test_classify_thresholds():
    assert classify_move(35, False, False, False) == "blunder"
    assert classify_move(22, False, False, False) == "mistake"
    assert classify_move(12, False, False, False) == "inaccuracy"
    assert classify_move(1, False, False, False) == "good"
    assert classify_move(0, True, False, False) == "best"
    assert classify_move(0, True, True, False) == "only_move"
    assert classify_move(50, False, False, True) == "forced"


def test_classify_severity_overrides_best_move_flags():
    # Local best-move flags can disagree with the calibrated eval stream
    # (especially when a shallow local best meets Lichess PGN evals). The coach
    # must not produce contradictions like "31.Kh1 (best, -33.7 win%)".
    assert classify_move(33.7, True, False, False) == "blunder"
    assert classify_move(21.0, True, True, False) == "mistake"
    assert classify_move(12.0, False, True, False) == "inaccuracy"


def test_time_flags():
    assert time_flag_for("blunder", 4, 300, False) == "hope_chess"
    assert time_flag_for("blunder", 4, 15, False) == "time_scramble"
    assert time_flag_for("forced", 90, 300, True) == "freezing"
    assert time_flag_for("good", 4, 300, False) is None
    assert time_flag_for("blunder", None, None, False) is None


# --- INTEGRATION: full pipeline with scripted evals ---

PGN = """[Event "Test"]
[White "mcmd"]
[Black "Opponent"]
[WhiteElo "1150"]
[BlackElo "1200"]
[Result "0-1"]

1. e4 e5 2. Qh5 Nc6 3. Qxf7+ Kxf7 0-1
"""


def _scripted_engines():
    """Evals scripted so 3. Qxf7+ is a clean White blunder (0 -> -700)."""
    game = chess.pgn.read_game(io.StringIO(PGN))
    board = game.board()
    fens = [board.fen()]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700   # after Qxf7+
    evals[fens[6]] = -700   # after Kxf7
    maia = {fens[4]: {"h5f7": 0.45, "g1f3": 0.25, "f1c4": 0.10}}  # position before Qxf7+
    return FakeEngines(evals=evals, maia=maia)


class _Cfg:
    user_elo = 1150
    player_name = "mcmd"


def test_review_finds_the_blunder_and_builds_agenda():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert review is not None
    assert review.student_color == "white"
    assert len(review.records) == 6

    blunder = review.records[4]  # 3. Qxf7+
    assert blunder.san == "Qxf7+"
    assert blunder.quality == "blunder"
    assert blunder.drop_pct > 30

    assert review.agenda_plies == [4]
    assert blunder.is_agenda
    assert blunder.maia_played_prob == 0.45  # the played move was Maia's top pick


def test_review_pgn_eval_severity_overrides_shallow_best_flag():
    # Regression for live game 2dtDqQgRy2OD: a shallow local best flag disagreed
    # with the calibrated PGN eval stream, producing "best, -33.7 win%".
    pgn = """[Event "Test"]
[White "mcmd"]
[Black "Opponent"]
[WhiteElo "1150"]
[BlackElo "1200"]
[Result "0-1"]

1. e4 { [%eval 0.00] } e5 { [%eval 0.00] } 2. Qh5 { [%eval 0.00] } Nc6 { [%eval 0.00] } 3. Qxf7+ { [%eval -7.00] } Kxf7 { [%eval -7.00] } 0-1
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    fens = []
    for node in game.mainline():
        fens.append(board.fen())
        board.push(node.move)

    engines = FakeEngines(bests={fens[4]: "h5f7"})
    review = review_game(pgn, engines, _Cfg(), with_maia=False)
    record = review.records[4]

    assert record.san == "Qxf7+"
    assert record.drop_pct >= 30
    assert record.quality == "blunder"


def test_agenda_bad_move_uses_non_played_deep_best_line():
    # If calibrated PGN evals make the played move a blunder but local MultiPV
    # still lists the played move first, the agenda must teach the best non-played
    # alternative instead of saying the blunder was also "better".
    pgn = """[Event "Test"]
[White "mcmd"]
[Black "Opponent"]
[WhiteElo "1150"]
[BlackElo "1200"]
[Result "0-1"]

1. e4 { [%eval 0.00] } e5 { [%eval 0.00] } 2. Qh5 { [%eval 0.00] } Nc6 { [%eval 0.00] } 3. Qxf7+ { [%eval -7.00] } Kxf7 { [%eval -7.00] } 0-1
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    fens = []
    for node in game.mainline():
        fens.append(board.fen())
        board.push(node.move)

    engines = FakeEngines(
        bests={fens[4]: "h5f7"},
        multipv={fens[4]: [("h5f7", 100), ("g1f3", 80)]},
    )

    review = review_game(pgn, engines, _Cfg(), with_maia=False)
    record = review.records[4]

    assert record.quality == "blunder"
    assert record.san == "Qxf7+"
    assert record.best_move_san == "Nf3"
    assert record.best_line[0] == "Nf3"


def test_digest_contains_agenda_and_withholds_answer():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    digest = review.digest()
    assert "STUDENT played WHITE" in digest
    assert "[Moment 1]" in digest
    assert "Qxf7+" in digest
    assert "WITHHELD ANSWER" in digest
    assert "evaluate_student_answer" in digest


def test_in_check_question_states_the_goal_the_risk_profile_supports():
    # Live eval 19enjp8m, move 31: "which keeps your winning chances?" in a position whose Maia
    # WDL (10/53/37, must-hold) and best play (a draw) make holding the goal.
    board = chess.Board("1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31")
    must_hold = _initial_question(board, risk="must-hold")
    assert "hold" in must_hold and "winning" not in must_hold and "threatening" not in must_hold
    assert "winning chances" in _initial_question(board, risk="winning")
    unknown = _initial_question(board)
    assert "in check" in unknown and "winning" not in unknown and "hold" not in unknown


def test_forced_recapture_classified_forced():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert review.records[5].san == "Kxf7"
    assert review.records[5].quality == "forced"


def test_unparseable_input_returns_none():
    assert review_game("this is not a pgn at all", FakeEngines(), _Cfg()) is None


def test_suggested_categories_from_motif_evidence():
    from app.review import suggest_error_categories, MoveRecord

    def rec(**kw):
        base = dict(ply=0, number=1, color="white", san="x", uci="0000",
                    fen_before="", fen_after="", eval_before_cp=0, eval_after_cp=0,
                    drop_pct=30, quality="blunder")
        base.update(kw)
        return MoveRecord(**base)

    r = rec(best_line_motifs=[{"motif": "knight_fork", "move": "Nc7+", "detail": ""}])
    assert "missed_fork" in suggest_error_categories(r)

    r = rec(motifs=[{"motif": "back_rank_mate", "move": "Ra8#", "detail": ""}])
    assert "king_safety" in suggest_error_categories(r)

    r = rec(motifs=[{"motif": "pin", "move": "Bb5", "detail": ""}], time_flag="hope_chess")
    cats = suggest_error_categories(r)
    assert "ignored_threat" in cats and "time_management" in cats

    assert suggest_error_categories(rec()) == []


def test_position_facts_grounds_file_claims():
    from app.review import position_facts
    import chess as _chess

    # Starting position: nothing is open, and the facts say so explicitly.
    assert "open files: NONE" in position_facts(_chess.Board())

    # The exact Moment-2 position from the live session where the coach
    # hallucinated "the g-file is completely open" (White pawn on g4, Black
    # pawns on g2/g7): c and e are the open files, g must not be listed.
    board = _chess.Board("r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5Pp1/3R1RK1 w - - 0 24")
    facts = position_facts(board)
    open_segment = [p for p in facts.split(" | ") if p.startswith("open files:")][0]
    assert open_segment == "open files: c, e"


def test_digest_includes_verified_position_facts():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert "Verified position facts" in review.digest()


def test_new_motif_suggestion_hooks():
    from app.review import suggest_error_categories, MoveRecord

    def rec(**kw):
        base = dict(ply=0, number=1, color="white", san="x", uci="0000",
                    fen_before="", fen_after="", eval_before_cp=0, eval_after_cp=0,
                    drop_pct=30, quality="blunder")
        base.update(kw)
        return MoveRecord(**base)

    # A discovered attack the student MISSED -> missed_tactic.
    r = rec(best_line_motifs=[{"motif": "discovered_attack", "move": "Nc3", "detail": ""}])
    assert "missed_tactic" in suggest_error_categories(r)

    # A zwischenzug in the PUNISHMENT -> the student miscalculated a "forced" exchange.
    r = rec(motifs=[{"motif": "zwischenzug", "move": "Re8+", "detail": ""}])
    assert "calculation_error" in suggest_error_categories(r)

    # Smothered mate in the punishment counts as a king-safety failure.
    r = rec(motifs=[{"motif": "smothered_mate", "move": "Nf7#", "detail": ""}])
    assert "king_safety" in suggest_error_categories(r)


def test_maia_runs_on_every_move_and_flags_alignment():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    blunder = review.records[4]  # Qxf7+ — scripted as Maia's top human move
    assert blunder.maia_played_prob == 0.45
    assert blunder.was_maia_top is True
    assert "Human-alignment" in review.digest()


def test_avoided_natural_mistake_is_praised():
    import chess as _chess
    import chess.pgn as _pgn
    import io as _io

    pgn = """[Event "Test"]
[White "mcmd"]
[Black "Opponent"]
[Result "*"]

1. e4 e5 2. Nf3 *
"""
    game = _pgn.read_game(_io.StringIO(pgn))
    board = game.board()
    fens = [board.fen()]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())

    # Maia says half of the student's peers play 2. f3?? here; the student
    # played 2. Nf3 instead. Scripted eval makes f3 lose ~25 win%.
    natural_board = _chess.Board(fens[2])
    natural_board.push_uci("f2f3")
    engines = FakeEngines(
        evals={natural_board.fen(): -300},
        maia={fens[2]: {"f2f3": 0.50, "g1f3": 0.20}},
    )
    review = review_game(pgn, engines, _Cfg())
    nf3 = review.records[2]
    assert nf3.was_maia_top is False
    assert nf3.avoided_natural_mistake is not None
    assert "f3" in nf3.avoided_natural_mistake
    digest = review.digest()
    assert "PRAISE-WORTHY" in digest


def test_review_eval_time_is_passed_to_engine():
    calls = []

    class TimingFakeEngines(FakeEngines):
        def quick_eval(self, fen, time_limit=0.3):
            calls.append(time_limit)
            return super().quick_eval(fen, time_limit)

    class Cfg(_Cfg):
        review_eval_time = 0.8

    review_game(PGN, TimingFakeEngines(), Cfg())
    assert calls and all(t == 0.8 for t in calls)


def _fens_of(pgn):
    import io as _io, chess.pgn as _pgn
    game = _pgn.read_game(_io.StringIO(pgn))
    board = game.board()
    fens = [board.fen()]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())
    return fens


def test_brilliancy_only_move_that_peers_miss():
    fens = _fens_of(PGN)
    # Make 2. Qh5 (ply 2) the engine-best ONLY move (gap >= 200cp) that
    # only 5% of peers find.
    engines = FakeEngines(
        evals={fen: 0 for fen in fens},
        bests={fens[2]: "d1h5"},
        multipv={fens[2]: [("d1h5", 300), ("g1f3", 0)]},
        maia={fens[2]: {"d1h5": 0.05, "g1f3": 0.60}},
    )
    review = review_game(PGN, engines, _Cfg())
    record = review.records[2]
    assert record.quality == "only_move"
    assert record.brilliancy_note is not None
    assert "ONLY good move" in record.brilliancy_note
    digest = review.digest()
    assert "Qh5!!" in digest          # brilliancy glyph in the movetext
    assert "PRAISE-WORTHY" in digest


def test_typical_blunder_flagged_for_normalization():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    blunder = review.records[4]  # Qxf7+ played by 45% of peers
    assert blunder.error_typicality == "typical_for_level"
    digest = review.digest()
    assert "COMMON at this level" in digest
    # Live eval 19enjp8m: "at your level" was said of numbers Maia gave for the game-time rating.
    assert blunder.maia_elo == 1150
    assert "~45% of players rated ~1150 choose Qxf7+" in digest


def test_atypical_blunder_flagged_as_process_slip():
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        bests={fens[4]: "g1f3"},
        maia={fens[4]: {"h5f7": 0.02, "g1f3": 0.55}},  # peers do NOT play Qxf7+
    )
    review = review_game(PGN, engines, _Cfg())
    blunder = review.records[4]
    assert blunder.error_typicality == "atypical_for_level"
    assert blunder.best_move_human_prob == 0.55  # findability of the right move
    digest = review.digest()
    assert "RARELY play this" in digest
    assert "Findability" in digest


def test_practical_difficulty_divergence_noted():
    fens = _fens_of(PGN)
    evals = {fen: 300 for fen in fens}  # engine: clearly better for White (~75 win%)
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        maia={fens[4]: {"h5f7": 0.45}},
        maia_win={fens[4]: 0.30},  # ...but peers only score 30% from here
    )
    review = review_game(PGN, engines, _Cfg())
    assert review.records[4].human_win_prob == 0.30
    assert "Practical difficulty" in review.digest()
    assert "HARDER to play" in review.digest()


def test_wdl_risk_profile_categories():
    from app.review import _wdl_risk_profile
    assert _wdl_risk_profile(0.60, 0.10, 0.30)[0] == "winning"
    assert _wdl_risk_profile(0.20, 0.10, 0.70)[0] == "must-hold"
    assert _wdl_risk_profile(0.25, 0.55, 0.20)[0] == "drawish"
    # A balanced low-draw spread (e.g. the start position, ~51/3/46) must NOT be
    # flagged as sharp — sharpness isn't in maia3's value head.
    assert _wdl_risk_profile(0.51, 0.03, 0.46)[0] == "balanced"
    # Winning chances but draw-dominant (e.g. K+P vs K, ~39/61/1) reads as drawish,
    # not "winning".
    assert _wdl_risk_profile(0.39, 0.60, 0.01)[0] == "drawish"


def test_wdl_risk_profile_surfaced_in_digest():
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        maia={fens[4]: {"h5f7": 0.45}},
        maia_wdl={fens[4]: (0.20, 0.60, 0.20)},  # drawish, mover POV
    )
    review = review_game(PGN, engines, _Cfg())
    rec = review.records[4]
    assert rec.human_draw_prob == 0.60 and rec.human_loss_prob == 0.20
    digest = review.digest()
    assert "Risk profile" in digest
    assert "drawish" in digest


def test_wdl_absent_when_prediction_has_no_wdl():
    # a prediction with no WDL → no draw/loss fields → no Risk profile line.
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(evals=evals, maia={fens[4]: {"h5f7": 0.45}})  # no maia_wdl
    review = review_game(PGN, engines, _Cfg())
    assert review.records[4].human_draw_prob is None
    assert "Risk profile" not in review.digest()


# --- within-game slide detection (eval-only) ---

def _slide_review(cps, color="white"):
    """Build a GameReview whose student plays consecutive middlegame moves with the
    given WHITE-relative (cp_before, cp_after) evals. drop_pct is computed in the
    student's perspective, exactly as review_game does."""
    from app.review import GameReview, MoveRecord, cp_to_win_pct
    student_white = color == "white"
    recs = []
    for i, (cb, ca) in enumerate(cps):
        mb = cb if student_white else -cb
        ma = ca if student_white else -ca
        drop = round(max(0.0, cp_to_win_pct(mb) - cp_to_win_pct(ma)), 1)
        recs.append(MoveRecord(ply=2 * i, number=i + 10, color=color, san="x", uci="e2e4",
                               fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                               eval_before_cp=cb, eval_after_cp=ca, drop_pct=drop,
                               quality="inaccuracy", phase="middlegame"))
    return GameReview(headers={}, records=recs, student_color=color,
                      agenda_plies=[], white_elo=1500, black_elo=1500)


def test_slide_detects_gradual_collapse():
    from app.review import _detect_slides
    # winning -> losing across 5 moves, each shedding ~8 win%, no single cliff.
    review = _slide_review([(300, 200), (200, 100), (100, 0), (0, -100), (-100, -200)])
    _detect_slides(review)
    assert review.slides is not None
    s = review.slides[0]
    assert s["plies"] == 5 and s["net_loss"] >= 25
    assert "WITHIN-GAME SLIDE" in review.digest()


def test_slide_silent_on_single_cliff():
    # one move sheds ~35 win%, the rest only a few -> a blunder the agenda already
    # covers, NOT a gradual slide (max single >= 70% of the net loss).
    from app.review import _detect_slides
    review = _slide_review([(200, -200), (-200, -250), (-250, -300)])
    _detect_slides(review)
    assert review.slides is None


def test_slide_silent_when_net_small():
    # three losing moves but tiny NET loss -> below the net threshold.
    from app.review import _detect_slides
    review = _slide_review([(50, 10), (10, -30), (-30, -70)])
    _detect_slides(review)
    assert review.slides is None


def test_slide_silent_from_floored_start():
    # already lost when the run starts (win% < 30) -> mop-up, not a collapse.
    from app.review import _detect_slides
    review = _slide_review([(-300, -400), (-400, -500), (-500, -600)])
    _detect_slides(review)
    assert review.slides is None


def test_slide_needs_min_run():
    # only two consecutive losing moves -> below MIN_RUN even with a big net loss.
    from app.review import _detect_slides
    review = _slide_review([(300, 100), (100, -100)])
    _detect_slides(review)
    assert review.slides is None


def test_slide_works_for_black_student():
    # same gradual collapse, student is Black: eval is white-relative, so a Black
    # slide is white-eval RISING. Confirms the perspective handling.
    from app.review import _detect_slides
    review = _slide_review([(-300, -200), (-200, -100), (-100, 0), (0, 100), (100, 200)],
                           color="black")
    _detect_slides(review)
    assert review.slides is not None and review.slides[0]["plies"] == 5


def test_slide_co_markers_from_starting_position():
    # the helper uses the starting FEN (queens on) and no clock -> queens_on True,
    # clock_low unknown (None); a winning onset reads "ahead".
    from app.review import _detect_slides
    review = _slide_review([(300, 200), (200, 100), (100, 0), (0, -100), (-100, -200)])
    _detect_slides(review)
    s = review.slides[0]
    assert s["onset"] == "ahead"
    assert s["queens_on"] is True
    assert s["clock_low"] is None


def test_slide_co_markers_queenless_and_low_clock():
    # queenless endgame with a low clock -> queens_on False, clock_low True.
    from app.review import _detect_slides, GameReview, MoveRecord, cp_to_win_pct
    qless = "8/5k2/8/8/8/8/5K2/R7 w - - 0 1"
    recs = []
    for i, (cb, ca) in enumerate([(300, 200), (200, 100), (100, 0), (0, -100), (-100, -200)]):
        drop = round(max(0.0, cp_to_win_pct(cb) - cp_to_win_pct(ca)), 1)
        recs.append(MoveRecord(ply=2 * i, number=i + 10, color="white", san="Ra2", uci="a1a2",
                               fen_before=qless, fen_after=qless,
                               eval_before_cp=cb, eval_after_cp=ca, drop_pct=drop,
                               quality="inaccuracy", phase="middlegame", clock_remaining=30.0))
    review = GameReview(headers={}, records=recs, student_color="white",
                        agenda_plies=[], white_elo=1500, black_elo=1500)
    _detect_slides(review)
    s = review.slides[0]
    assert s["queens_on"] is False
    assert s["clock_low"] is True


def test_likely_wrong_answer_anticipated_with_refutation():
    import chess as _chess
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    # The guess peers would offer at the blunder position: Nf3 — scripted to lose ~25 win%.
    guess_board = _chess.Board(fens[4])
    guess_board.push_san("Nf3")
    evals[guess_board.fen()] = -300
    engines = FakeEngines(
        evals=evals,
        maia={fens[4]: {"h5f7": 0.45, "g1f3": 0.25}},
    )
    review = review_game(PGN, engines, _Cfg())
    record = review.records[4]
    assert record.likely_wrong_answer is not None
    assert "Nf3" in record.likely_wrong_answer
    assert "refute with" in record.likely_wrong_answer
    assert "most likely student guess" in review.digest()


def test_aspirational_instinct_from_stronger_maia():
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        maia={fens[4]: {"h5f7": 0.45}},
        maia_by_elo={(fens[4], 1500): {"g1f3": 0.70}},  # WhiteElo 1150 + 350
    )
    review = review_game(PGN, engines, _Cfg())
    record = review.records[4]
    assert record.aspirational == {"elo": 1500, "move": "Nf3", "probability": 0.70}
    assert "aspirational instinct" in review.digest()
    assert "growing toward" in review.digest()


def test_detect_student_color_exact_not_substring():
    # Audit: substring matching let "Ann" match "JoanneSmith". Exact normalized only.
    from app.review import configured_player_identities, detect_student_color

    class C:
        player_name = "Ann"
        usernames = {}

    assert detect_student_color({"White": "JoanneSmith", "Black": "Ann"}, C()) == "black"
    assert detect_student_color({"White": "ann", "Black": "Bob"}, C()) == "white"   # case-insensitive
    assert detect_student_color({"White": "Joanne", "Black": "Bob"}, C()) is None   # no substring hit

    class Multi:
        player_name = "legacy_handle"
        usernames = {"lichess": ["mcmd", "AltName"], "chesscom": ["mcmd_cc"]}

    assert detect_student_color({"White": "opponent", "Black": "ALTNAME"}, Multi()) == "black"
    assert detect_student_color({"White": "mcmd_cc", "Black": "opponent"}, Multi()) == "white"
    assert configured_player_identities(Multi()) == {"legacy_handle", "mcmd", "altname", "mcmd_cc"}


def test_agenda_empty_for_clean_student_even_if_opponent_blundered():
    # Audit: a clean student game must NOT coach the opponent's blunders.
    import chess
    from app.review import _select_agenda, MoveRecord

    def rec(ply, color, drop):
        return MoveRecord(ply=ply, number=ply // 2 + 1, color=color, san="x", uci="a2a3",
                          fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                          eval_before_cp=0, eval_after_cp=0, drop_pct=drop, quality="x")

    records = [rec(0, "white", 2), rec(1, "black", 40), rec(2, "white", 1)]
    assert _select_agenda(records, "white", 4) == []     # student clean → no agenda
    assert _select_agenda(records, None, 4) == [1]        # unknown side → widen to both


def test_agenda_criticality_uses_capped_human_danger_without_reclassifying():
    # P0-1b: the agenda is no longer pure raw drop. A nearby mistake with strong
    # human-danger/typicality evidence can be more instructive, but provisional
    # Maia/local-eval signals are capped and do not rewrite severity.
    import chess
    from app.review import _select_agenda, MoveRecord

    def rec(ply, drop, quality, *, danger=None, played_prob=None, best_prob=None):
        return MoveRecord(
            ply=ply, number=ply // 2 + 1, color="white", san="x", uci="a2a3",
            fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
            eval_before_cp=0, eval_after_cp=0, drop_pct=drop, quality=quality,
            maia_played_prob=played_prob, best_move_human_prob=best_prob,
            human_danger=danger or {},
        )

    raw_higher = rec(0, 22.0, "mistake")
    human_dangerous = rec(
        2, 21.0, "mistake",
        played_prob=0.35,
        best_prob=0.05,
        danger={
            "available": True,
            "provisional": True,
            "source": "maia_top_moves_plus_local_quick_eval",
            "bad_or_worse_mass": 0.70,
            "mistake_or_blunder_mass": 0.70,
        },
    )
    large_blunder = rec(4, 35.0, "blunder")

    assert _select_agenda([raw_higher, human_dangerous], "white", 1) == [2]
    assert human_dangerous.quality == "mistake"
    assert human_dangerous.drop_pct == 21.0
    assert human_dangerous.agenda_score["human_boost"] == 10.0
    assert human_dangerous.agenda_score["human_danger_provisional"] is True

    assert _select_agenda([raw_higher, human_dangerous, large_blunder], "white", 1) == [4]


def test_review_aborts_on_engine_failure_instead_of_fabricating_zero():
    # Audit: a Stockfish failure must surface, never become a 0.00 ("equal") eval.
    import pytest
    from app.review import EngineUnavailable

    class Boom(FakeEngines):
        def quick_eval(self, fen, time_limit=0.3):
            raise RuntimeError("stockfish died")

    with pytest.raises(EngineUnavailable):
        review_game(PGN, Boom(), _Cfg())


def test_parse_increment():
    from app.review import _parse_increment
    assert _parse_increment({"TimeControl": "300+3"}) == 3
    assert _parse_increment({"TimeControl": "600"}) == 0
    assert _parse_increment({}) == 0


def test_reason_set_marks_forcing_tactic_as_key():
    # P0-2: a forcing tactic the better move prevents is the load-bearing reason —
    # keyed, anchored on the target square, with the motif as a sayable keyword.
    import chess
    from app.review import _reason_set, MoveRecord
    r = MoveRecord(ply=0, number=1, color="white", san="Re1", uci="f1e1",
                   fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
                   eval_before_cp=0, eval_after_cp=-300, drop_pct=20.0, quality="mistake")
    r.motifs = [{"motif": "smothered_mate", "detail": "Qxh2# is smothered mate"}]
    data = {"target_square": "h2", "forcing_move": "Qxh2#", "forcing_move_prevented": True,
            "why_wrong_move_fails": "Re1 does not address Qxh2#; the opponent gets a forcing move on h2"}
    reasons = _reason_set(r, chess.Board(chess.STARTING_FEN), data)
    key = [x for x in reasons if x["key"]]
    assert len(key) == 1 and key[0]["squares"] == ["h2"] and "mate" in key[0]["keywords"]


def test_aspirational_anchors_on_conversation_elo_not_game_elo():
    # P0-4a: aspiration is for the student's CURRENT level. A game rated 1150
    # reviewed by a 1500 student aspires from 1500 (+350 = 1850), not the game-time
    # 1150 — and the digest flags the rating divergence.
    class C:
        user_elo = 1500
        player_name = "mcmd"

    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(evals=evals, maia={fens[4]: {"h5f7": 0.45}},
                          maia_by_elo={(fens[4], 1850): {"g1f3": 0.70}})
    review = review_game(PGN, engines, C())
    assert review.conversation_elo == 1500                      # anchor = config, not header
    assert review.white_elo == 1150                             # typicality = PGN game-time
    assert review.records[4].aspirational == {"elo": 1850, "move": "Nf3", "probability": 0.70}
    digest = review.digest()
    assert "Rating note" in digest and "~1150" in digest and "~1500" in digest


def test_no_rating_note_when_game_elo_matches_conversation():
    review = review_game(PGN, _scripted_engines(), _Cfg())      # both 1150
    assert review.conversation_elo == 1150
    assert "Rating note" not in review.digest()


def test_resolve_student_elo_two_purposes_and_sources():
    # P0-4a: conversation anchors to config; typicality reads the PGN game-time Elo
    # of the mover (the session's 1142-vs-config case), falling back to the anchor.
    from app.review import resolve_student_elo

    class C:
        user_elo = 1150

    headers = {"WhiteElo": "1142", "BlackElo": "1500"}
    assert resolve_student_elo("white", headers, C(), "conversation") == (1150, "config")
    assert resolve_student_elo("white", headers, C(), "typicality") == (1142, "pgn")
    assert resolve_student_elo("black", headers, C(), "typicality") == (1500, "pgn")
    assert resolve_student_elo("white", {}, C(), "typicality") == (1150, "config")


def test_aspirational_suppressed_when_band_plays_the_same_move():
    # P0-6 divergence guard: if the +350 band's top move equals the move the
    # student actually played, no aspirational instinct is surfaced (it must never
    # dress the student's own move up as "the instinct you're growing toward").
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    base = FakeEngines(evals=evals, maia={fens[4]: {"h5f7": 0.45}})
    played_uci = review_game(PGN, base, _Cfg()).records[4].uci
    engines = FakeEngines(
        evals=evals,
        maia={fens[4]: {"h5f7": 0.45}},
        maia_by_elo={(fens[4], 1500): {played_uci: 0.70}},  # band plays the SAME move
    )
    review = review_game(PGN, engines, _Cfg())
    assert review.records[4].aspirational is None
    assert "growing toward" not in review.digest()


def test_phase_alignment_breakdown_in_digest():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    assert review.records[0].phase == "opening"
    assert "opening" in review.digest().split("Human-alignment")[1].split("\n")[0]


def test_opening_alignment_facts_separate_book_from_maia():
    from app.review import MoveRecord, _opening_alignment_facts
    records = [
        MoveRecord(
            ply=0, number=1, color="white", san="e4", uci="e2e4",
            fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
            eval_before_cp=0, eval_after_cp=0, drop_pct=0, quality="good",
            phase="opening", maia_top=[{"move": "e4", "probability": 0.5}],
            was_maia_top=True, book_share=0.32,
        ),
        MoveRecord(
            ply=2, number=2, color="white", san="Nf3", uci="g1f3",
            fen_before=chess.STARTING_FEN, fen_after=chess.STARTING_FEN,
            eval_before_cp=0, eval_after_cp=0, drop_pct=0, quality="good",
            phase="opening", maia_top=[{"move": "Nc3", "probability": 0.4}],
            was_maia_top=False,
        ),
    ]
    facts = _opening_alignment_facts(records)
    assert "book/theory" in facts
    assert "Maia human-commonness" in facts
    assert "DIFFERENT" in facts
    assert "1/2 student opening moves were in the masters database" in facts
    assert "matched Maia's top move on 1/2 student opening moves" in facts


def test_agenda_has_question_ladder_and_reveal_primitives():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    record = review.agenda[0]
    assert record.lesson_objective
    assert record.recommended_diagnosis
    assert record.initial_question.startswith("White to move")
    assert 0 not in record.question_ladder
    assert "Initial question (ask this EXACTLY first" in review.digest()
    assert "Level 0 or Level 1" not in review.digest()
    assert "Hint ladder (use only after the student attempts" in review.digest()
    assert "Recommended diagnosis" in review.digest()
    assert record.explanation_primitives["played_move"] == "Qxf7+"
    assert record.explanation_primitives["played_move_label"] == "3. Qxf7+"
    assert "best_move_label" in record.explanation_primitives
    assert "diagnosis_sentence" in record.explanation_primitives
    assert "reveal-phase explanation primitives" in review.digest()


def test_forcing_move_consequences_name_actual_targets_not_wrong_rook():
    from app.review import MoveRecord, _explanation_primitives
    before = chess.Board("1r2r1k1/p4ppp/6q1/2QP4/1P4P1/P7/5PK1/3R1R2 w - - 3 26")
    after_wrong = before.copy(stack=False)
    after_wrong.push_san("Qxa7")
    record = MoveRecord(
        ply=50, number=26, color="white", san="Qxa7", uci="c5a7",
        fen_before=before.fen(), fen_after=after_wrong.fen(),
        eval_before_cp=0, eval_after_cp=0, drop_pct=40, quality="blunder",
        best_move_san="Rd4", is_capture=True, punishment_line=["Qxg4+"],
        lesson_objective="defend_against_direct_threat",
    )
    primitives = _explanation_primitives(record, before, after_wrong)
    consequences = primitives["forcing_move_consequences"]
    assert consequences["move"] == "Qxg4+"
    assert consequences["checks"] == "king on g2"
    assert "rook on d1" in consequences["also_attacks"]
    assert "rook on f1" not in consequences["also_attacks"]


def test_drop_rounding_consistent_between_classify_and_agenda():
    # Unrounded drop ~9.99 rounds to 10.0: it must BOTH classify as an
    # inaccuracy AND qualify for the agenda (no more "critical moment (ok)").
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[1]] = -110  # after 1. e4: drop just under 10, rounds to 10.0
    review = review_game(PGN, FakeEngines(evals=evals), _Cfg())
    record = review.records[0]
    assert record.drop_pct == 10.0
    assert record.quality == "inaccuracy"
    assert record.ply in review.agenda_plies


# --- PRACTICAL vs CONCRETE mistakes (the "why is ...c5 dangerous?" session) ---

def test_practical_mistake_flagged_with_narrow_saving_resource():
    """Mirrors the live d5 case: shallow scan says the move lost win%, but deep
    analysis shows it holds — via one narrow, rarely-found resource."""
    import chess as _chess
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700   # shallow scan: Qxf7+ looks losing -> agenda
    evals[fens[6]] = -700

    # Deep punishment analysis at the position AFTER the played move: actually fine
    # for White (-20 from Black's POV), with forced reply Kxf7.
    reply_board = _chess.Board(fens[6])
    saving_uci = next(iter(reply_board.legal_moves)).uci()
    engines = FakeEngines(
        evals=evals,
        multipv={
            fens[5]: [("e8f7", -20)],                      # deep: White is NOT lost
            fens[6]: [(saving_uci, 300), ("d2d4", 0)],     # one saving move, gap 300
        },
        maia={fens[4]: {"h5f7": 0.45},
              fens[6]: {saving_uci: 0.07}},                # peers rarely find it
    )
    review = review_game(PGN, engines, _Cfg())
    record = review.records[4]
    assert record.practical_note is not None
    assert "PRACTICAL, not forced" in record.practical_note
    assert "NARROW" in record.practical_note
    assert record.saving_resource is not None
    assert "only good move" in record.saving_resource
    assert "7%" in record.saving_resource
    digest = review.digest()
    assert "Nature of the mistake: PRACTICAL" in digest
    assert "saving resource" in digest


def test_agenda_enrichment_uses_configured_engine_limits():
    class Cfg(_Cfg):
        agenda_best_depth = 21
        agenda_best_time = 7.0
        agenda_punishment_depth = 19
        agenda_punishment_time = 5.0
        practicality_depth = 17
        practicality_time = 3.5

    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        multipv={fens[5]: [("e8f7", 700)]},
        maia={fens[4]: {"h5f7": 0.45}},
    )
    engines.config = Cfg()

    review_game(PGN, engines, Cfg())

    assert any(c["depth"] == 21 and c["time_cap"] == 7.0 and c["multipv"] == 3
               for c in engines.top_line_calls)
    assert any(c["depth"] == 19 and c["time_cap"] == 5.0 and c["multipv"] == 1
               for c in engines.top_line_calls)


def test_practicality_probe_uses_configured_engine_limits():
    class Cfg(_Cfg):
        practicality_depth = 17
        practicality_time = 3.5

    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    reply_board = chess.Board(fens[6])
    saving_uci = next(iter(reply_board.legal_moves)).uci()
    engines = FakeEngines(
        evals=evals,
        multipv={
            fens[5]: [("e8f7", -20)],
            fens[6]: [(saving_uci, 300), ("d2d4", 0)],
        },
        maia={fens[4]: {"h5f7": 0.45}, fens[6]: {saving_uci: 0.07}},
    )
    engines.config = Cfg()

    review_game(PGN, engines, Cfg())

    assert any(c["depth"] == 17 and c["time_cap"] == 3.5 and c["multipv"] == 2
               for c in engines.top_line_calls)


def test_concrete_mistake_flagged_when_deep_analysis_confirms_loss():
    fens = _fens_of(PGN)
    evals = {fen: 0 for fen in fens}
    evals[fens[5]] = -700
    evals[fens[6]] = -700
    engines = FakeEngines(
        evals=evals,
        multipv={fens[5]: [("e8f7", 700)]},  # deep analysis agrees: White is lost
        maia={fens[4]: {"h5f7": 0.45}},
    )
    review = review_game(PGN, engines, _Cfg())
    record = review.records[4]
    assert record.practical_note is not None
    assert "CONCRETE" in record.practical_note
    assert record.saving_resource is None
    assert "Nature of the mistake: CONCRETE" in review.digest()


def test_forcing_check_punishment_recommends_ignored_threat():
    """A refutation that starts with a forcing check means the student ignored
    a direct threat — even when the mistake was a capture (live-session bug:
    26. Qxa7 punished by Qxg4+ was recommended as bad_trade)."""
    from app.review import (MoveRecord, suggest_error_categories,
                            _infer_lesson_objective, _recommended_diagnosis)
    r = MoveRecord(
        ply=50, number=26, color="white", san="Qxa7", uci="c5a7",
        fen_before="1r2r1k1/p4ppp/6q1/2QP4/1P4P1/P7/5PK1/3R1R2 w - - 3 26",
        fen_after="1r2r1k1/Q4ppp/6q1/3P4/1P4P1/P7/5PK1/3R1R2 b - - 0 26",
        eval_before_cp=200, eval_after_cp=-300, drop_pct=39.7,
        quality="blunder", best_move_san="Rd4", is_capture=True,
        punishment_line=["Qxg4+", "Kh1", "Re2"])
    cats = suggest_error_categories(r)
    assert cats[0] == "ignored_threat"
    r.suggested_categories = cats
    assert _infer_lesson_objective(r) == "defend_against_direct_threat"
    r.lesson_objective = "defend_against_direct_threat"
    assert _recommended_diagnosis(r) == "ignored_threat"


def test_material_grab_recaptured_is_not_taught_as_standing_square_threat():
    from app.review import (MoveRecord, suggest_error_categories, _infer_lesson_objective,
                            _recommended_diagnosis, _explanation_primitives)
    before = chess.Board("r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 0 20")
    after = before.copy(stack=False)
    after.push_san("Nxb5")
    r = MoveRecord(
        ply=38, number=20, color="white", san="Nxb5", uci="c3b5",
        fen_before=before.fen(), fen_after=after.fen(),
        eval_before_cp=350, eval_after_cp=90, drop_pct=19.1,
        quality="inaccuracy", best_move_san="Ne4", is_capture=True,
        punishment_line=["Nxb5"],
        motifs=[{"motif": "knight_fork", "detail": "synthetic residue"}],
    )
    cats = suggest_error_categories(r)
    assert cats[0] == "bad_trade"
    assert "ignored_threat" not in cats
    r.suggested_categories = cats
    assert _infer_lesson_objective(r) == "material_vs_activity"
    r.lesson_objective = "material_vs_activity"
    assert _recommended_diagnosis(r) == "bad_trade"
    r.recommended_diagnosis = "bad_trade"

    primitives = _explanation_primitives(r, before, after)

    key = [reason for reason in primitives["reasons"] if reason.get("key")][0]
    assert key["squares"] == ["e4", "d6"]
    assert "tempo" in key["keywords"]
    assert "forcing threat on b5" not in key["text"]
    assert "attacks the queen on d6" in key["text"]


# Real agenda records for game 2dtDqQgR as produced by the current review pipeline
# (recorded in the 2026-10-12 manual session). Engine lines are recorded evidence.
_GAME_2DTD_RECORDS = {
    24: dict(ply=46, number=24, color="white", san="Kxg2", uci="g1g2",
             fen_before="r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5Pp1/3R1RK1 w - - 0 24",
             fen_after="r3r1k1/p4ppp/3q4/1Q1P4/1P4P1/P7/5PK1/3R1R2 b - - 0 24",
             quality="inaccuracy", drop_pct=16.3, is_capture=True, best_move_san="Rfe1",
             best_line=["Rfe1", "Rec8", "Re4", "Rab8", "Qd3", "h6", "Rde1", "Rf8"],
             punishment_line=["Qf4", "Rfe1", "Qxg4+", "Kf1", "Qh3+", "Kg1", "Qg4+"],
             best_line_motifs=[{"motif": "skewer", "move": "Rab8"}]),
    26: dict(ply=50, number=26, color="white", san="Qxa7", uci="c5a7",
             fen_before="1r2r1k1/p4ppp/6q1/2QP4/1P4P1/P7/5PK1/3R1R2 w - - 3 26",
             fen_after="1r2r1k1/Q4ppp/6q1/3P4/1P4P1/P7/5PK1/3R1R2 b - - 0 26",
             quality="blunder", drop_pct=41.9, is_capture=True, best_move_san="Rd4",
             best_line=["Rd4", "Re4", "f3", "Re5", "Qc6", "Qg5", "Kg3", "h5"],
             punishment_line=["Qxg4+", "Kh2", "Qh5+", "Kg3", "Qg6+", "Kh2", "Rb6", "Qxb6"],
             best_line_motifs=[{"motif": "skewer", "move": "Re4"}, {"motif": "pin", "move": "Re5"}]),
    31: dict(ply=60, number=31, color="white", san="Kh1", uci="g1h1",
             fen_before="1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1RK1 w - - 8 31",
             fen_after="1r2r1k1/Q4ppp/8/3P4/1P4q1/P7/5P2/3R1R1K b - - 9 31",
             quality="blunder", drop_pct=33.7, is_capture=False, best_move_san="Kh2",
             best_line=["Kh2", "Qh5+", "Kg2", "Qg5+", "Kh1", "Qh6+", "Kg2", "Qg6+"],
             punishment_line=["Qh5+", "Kg2", "Qg6+", "Kf3", "Rb6", "Qxb6", "Qxb6", "Rg1"]),
}


def _real_primitives(move_number):
    from app.review import MoveRecord, _explanation_primitives
    r = MoveRecord(eval_before_cp=0, eval_after_cp=0, **_GAME_2DTD_RECORDS[move_number])
    return _explanation_primitives(r, chess.Board(r.fen_before), chess.Board(r.fen_after))


def _key_reasons(primitives):
    return [reason for reason in primitives["reasons"] if reason.get("key")]


def test_key_reason_names_what_the_best_move_does_to_the_threatened_square():
    # Move 26: 26. Rd4 defends g4 along the fourth rank, so Qxg4+ loses the queen
    # (Stockfish: Rxg4 +10). The key reason must be that defence, not only why Qxa7 failed.
    key = _key_reasons(_real_primitives(26))
    assert len(key) == 1
    assert "g4" in key[0]["squares"]
    assert "Rd4" in key[0]["text"] and "controls g4" in key[0]["text"]


def test_no_key_reason_when_the_best_move_does_not_stop_the_forcing_move():
    # Move 31: the verified best line itself is 31. Kh2 Qh5+, so "Kh2 stops Qh5+" is false.
    primitives = _real_primitives(31)
    assert _key_reasons(primitives) == []
    assert "does not address" not in primitives["why_wrong_move_fails"]
    assert "Kh2" in primitives["why_wrong_move_fails"] and "Qh5+" in primitives["why_wrong_move_fails"]


# Baseline live eval 1l_zs5z1 (2026-10-13), move 31 as reviewed in that run.
_MOVE31_RUN13 = dict(_GAME_2DTD_RECORDS[31],
                     best_line=["Kh2", "Qh5+", "Kg3", "Qg6+", "Kh4", "Rb6", "Qxb6", "Qxb6"],
                     punishment_line=["Qh4+", "Kg1", "Qg5+", "Kh2", "Qh6+", "Kg2", "Qg6+", "Kf3"])


def test_diagnosis_sentence_never_claims_the_better_move_stops_the_forcing_check():
    # The digest said "Played 31. Kh1 without stopping Qh4+; 31. Kh2 was needed to meet that
    # forcing threat", and the coach taught that Kh2 prevents Qh4+. Qh4+ is legal after Kh2.
    from app.review import MoveRecord, _diagnosis_sentence
    sentence = _diagnosis_sentence(MoveRecord(eval_before_cp=0, eval_after_cp=-445, **_MOVE31_RUN13))
    assert "without stopping" not in sentence and "needed to meet" not in sentence
    assert "Kh1" in sentence and "Qh4+" in sentence and "Kh2" in sentence


def test_moment_header_shows_the_deep_eval_when_the_scan_eval_disagrees():
    # The scan said +0.00 before 31. Kh1; the depth-20 best line (Kh2 ... Rb6 Qxb6 Qxb6) wins
    # White's queen, and the coach called the position "completely balanced".
    from app.review import MoveRecord, moment_digest_lines
    contradicted = MoveRecord(eval_before_cp=0, eval_after_cp=-445, deep_eval_before_cp=-894, **_MOVE31_RUN13)
    agreeing = MoveRecord(eval_before_cp=0, eval_after_cp=-445, deep_eval_before_cp=-40, **_MOVE31_RUN13)
    assert "-8.94" in moment_digest_lines(4, contradicted)[0]
    assert "-8.94" not in moment_digest_lines(4, agreeing)[0] and "deep" not in moment_digest_lines(4, agreeing)[0]


class _DepthEngines:
    """Practical-depth scores by position, as the agenda analysis asks for them."""

    def __init__(self, scores):
        self.scores = scores   # {fen: white-POV cp at practical depth}

    def practical_eval(self, board, depth=10, time_cap=1.0):
        return chess.engine.PovScore(chess.engine.Cp(self.scores[board.fen()]), chess.WHITE)


def test_engine_only_refutation_of_the_better_move_is_reported_as_practical():
    # Move 31, measured 2026-10-13: after 31. Kh2 Stockfish says -1.95 at depth 8 and -7.16 at
    # depth 20 (the win needs ...Qh5+, ...Qg6+, ...Rb6!); after 31. Kh1 it is -7.63 at every depth.
    from app.review import MoveRecord, _practical_refutation
    r = MoveRecord(eval_before_cp=0, eval_after_cp=-445, **_MOVE31_RUN13)
    before = chess.Board(r.fen_before)
    after_best = before.copy(stack=False)
    after_best.push_san("Kh2")
    engines = _DepthEngines({after_best.fen(): -195, r.fen_after: -763})

    note = _practical_refutation(r, engines, before, chess.Board(r.fen_after), deep_best_white_cp=-716)
    assert note and "Kh2" in note and "Kh1" in note
    # Router run r_2e1yzi S19/S20: "at depth 10 Kh2 holds" became "Kh2 held the position" and "salvaba
    # la partida"; at depth 30 Kh2 still loses. S18 left "Kh2 avoids the perpetual" uncorrected.
    assert " holds " not in note and "also loses with best play" in note
    assert "repetition is a drawing chance" in note
    assert _practical_refutation(r, engines, before, chess.Board(r.fen_after), deep_best_white_cp=-150) is None


def test_lines_shown_to_the_coach_say_whose_move_it_is():
    # Runs par4cj6m and t0zauzig: the punishment line "Nxb5 Qxb5 e4 ..." starts with BLACK's
    # recapture after 20. Nxb5; read from the position before the move it became the illegal
    # "20... Qxb5", which the coach taught and passed to analyze_position.
    from app.review import MoveRecord, _diagnosis_sentence, moment_digest_lines
    before = chess.Board("r3r1k1/p1n2ppp/3q4/1p1Pp3/1P4n1/P1N2N2/4QPPP/3R1RK1 w - - 3 20")
    after = before.copy(stack=False)
    after.push_san("Nxb5")
    r = MoveRecord(ply=38, number=20, color="white", san="Nxb5", uci="c3b5", fen_before=before.fen(),
                   fen_after=after.fen(), eval_before_cp=280, eval_after_cp=62, drop_pct=18.0,
                   quality="inaccuracy", is_capture=True, best_move_san="Ne4",
                   best_line=["Ne4", "Qd7", "h3", "Nf6"], punishment_line=["Nxb5", "Qxb5", "e4", "Rfe1"])

    assert "20... Nxb5" in _diagnosis_sentence(r)
    digest = "\n".join(moment_digest_lines(1, r))
    assert "20... Nxb5 21. Qxb5 e4 22. Rfe1" in digest
    assert "20. Ne4 Qd7 21. h3 Nf6" in digest


def _record_after(fen_before, san, best, punishment):
    """A record as review builds it: the engine's punishment line and the motifs found in it."""
    from app.review import MoveRecord, detect_motifs_in_line
    before = chess.Board(fen_before)
    move = before.parse_san(san)
    after = before.copy(stack=False)
    after.push(move)
    line = punishment.split()
    return MoveRecord(ply=0, number=before.fullmove_number, color="white" if before.turn else "black",
                      san=san, uci=move.uci(), fen_before=fen_before, fen_after=after.fen(),
                      eval_before_cp=0, eval_after_cp=-300, drop_pct=20.0, quality="blunder",
                      best_move_san=best, is_capture=before.is_capture(move), punishment_line=line,
                      motifs=detect_motifs_in_line(after, line))


def _diagnose(record):
    from app.review import _infer_lesson_objective, _recommended_diagnosis, suggest_error_categories
    record.suggested_categories = suggest_error_categories(record)
    record.lesson_objective = _infer_lesson_objective(record)
    return _recommended_diagnosis(record)


# Positions and lines below are from the coaching corpus, checked with scripts/verify_chess.py
# (Stockfish depth 30, cleared hash, every search complete).

def test_a_new_check_in_the_punishment_line_is_an_allowed_forcing_sequence():
    # ZU8uPxvn 10...f5? (-0.40 -> +3.20) opens a2-g8; the review's line reaches 12. Bc4+ (+3.10).
    record = _record_after("r1b2rk1/pp2bppp/2n5/2pqp3/8/1P1P1N1P/P1P1BPP1/R1BQ1RK1 b - - 2 10", "f5",
                           "Be6", "d4 Qd6 Bc4+ Kh8 Nxe5 Nxe5")
    assert _diagnose(record) == "allows_forcing_sequence"
    assert record.lesson_objective == "check_opponent_forcing_replies"


def test_a_recapture_refuted_by_a_new_check_is_an_allowed_forcing_sequence():
    # #9's case, YbjNb8wD 16. Re3? (-7.19 -> -9.49): after ...Bxd2, 17. Qxd2 runs into ...Qxa1+, a check
    # the rook used to block; the engine line plays 17. Re2 (both -9.55 at depth 30).
    record = _record_after("r4rk1/ppp2ppp/5n2/8/1bPq4/1P1P1BPb/P2N1P1P/R2QR1K1 w - - 2 16", "Re3",
                           "Rb1", "Bxd2 Re2 Bc3 Rc1 Rae8 c5 Rxe2 Qxe2")
    assert _diagnose(record) == "allows_forcing_sequence"


def test_a_threat_that_already_existed_stays_ignored_threat():
    # Move 26 of 2dtDqQgR: ...Qxg4+ was already available before 26. Qxa7.
    from app.review import MoveRecord
    record = MoveRecord(eval_before_cp=0, eval_after_cp=0, **_GAME_2DTD_RECORDS[26])
    assert _diagnose(record) == "ignored_threat"


def test_an_allowed_mate_is_king_safety_even_after_a_recapture():
    # YbjNb8wD 20. Rxe8? (-9.58 -> mate in 5): ...Rxe8 recaptures, then ...Qxf1# at ply 9.
    record = _record_after("4rrk1/pBp2ppp/5n2/8/2Pq4/1P1P2Pb/P2Q1P1P/4R1K1 w - - 1 20", "Rxe8",
                           "Bg2", "Rxe8 Bf3 Qxd3 Be2 Qxd2 Bf1 Qe1 b4 Qxf1#")
    assert _diagnose(record) == "king_safety"


def test_pawn_push_to_avoid_tension_is_not_missed_fork():
    from app.review import MoveRecord, suggest_error_categories, _infer_lesson_objective, _recommended_diagnosis
    r = MoveRecord(
        ply=32, number=17, color="white", san="d5", uci="d4d5",
        fen_before="r4rk1/p4ppp/n1pq4/1p2p3/1P1PP1n1/P1N2N2/4QPPP/R4RK1 w - - 0 17",
        fen_after="r4rk1/p4ppp/n1pq4/1p1Pp3/1P2P1n1/P1N2N2/4QPPP/R4RK1 b - - 0 17",
        eval_before_cp=50, eval_after_cp=-30, drop_pct=10.2,
        quality="inaccuracy", best_move_san="Rad1",
        best_line_motifs=[{"motif": "knight_fork", "detail": "synthetic tactical residue"}],
    )
    cats = suggest_error_categories(r)
    assert cats[0] == "premature_attack"
    assert "missed_fork" not in cats
    r.suggested_categories = cats
    assert _infer_lesson_objective(r) == "improve_pieces_before_committing_pawns"
    r.lesson_objective = "improve_pieces_before_committing_pawns"
    assert _recommended_diagnosis(r) == "premature_attack"


def _short_game(moves: str, result: str, termination: str = "Normal", evals: float = None):
    """A PGN for mcmd (White) vs Opponent; `evals` adds the same [%eval] after every move."""
    tokens = []
    for tok in moves.split():
        tokens.append(tok)
        if evals is not None and not tok[0].isdigit():
            tokens.append("{ [%%eval %.2f] }" % evals)
    return (f'[Event "Test"]\n[White "mcmd"]\n[Black "Opponent"]\n[WhiteElo "1150"]\n[BlackElo "1200"]\n'
            f'[Result "{result}"]\n[Termination "{termination}"]\n\n{" ".join(tokens)} {result}\n')


def test_review_says_how_the_game_ended_and_what_followed_each_move():
    # Live eval 5kf7rqwj: 2dtDqQgR ended in a draw by repetition right after 31. Kh1 (the move
    # played), and the coach told the student that the draw proved the value of 31. Kh2.
    pgn = _short_game("1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3", "1/2-1/2", evals=-5.0)
    review = review_game(pgn, FakeEngines(), _Cfg(), with_maia=False)
    ending = review.ending()
    assert "repetition" in ending and "5. Nf3" in ending and "third" in ending
    assert "good result for the student" in ending          # White was 5 pawns worse at the end
    assert ending in review.digest()
    assert "final move of the game" in review.continuation_after(review.records[-1])
    assert "1... Nf6" in review.continuation_after(review.records[0])


def test_review_ending_names_checkmate_and_resignation():
    mate = review_game(_short_game("1. f3 e5 2. g4 Qh4#", "0-1"), FakeEngines(), _Cfg(), with_maia=False)
    assert "checkmate" in mate.ending().lower() and "2... Qh4#" in mate.ending()
    resigned = review_game(_short_game("1. e4 e5 2. Qh5 Nc6", "0-1"), FakeEngines(), _Cfg(), with_maia=False)
    assert "White resigned" in resigned.ending() and "good result" not in resigned.ending()
