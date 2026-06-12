import io

import chess
import chess.pgn

from app.review import (
    cp_to_win_pct, classify_move, time_flag_for, review_game,
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


def test_digest_contains_agenda_and_withholds_answer():
    review = review_game(PGN, _scripted_engines(), _Cfg())
    digest = review.digest()
    assert "STUDENT played WHITE" in digest
    assert "[Moment 1]" in digest
    assert "Qxf7+" in digest
    assert "WITHHELD ANSWER" in digest
    assert "evaluate_student_answer" in digest


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
    assert "COMMON at this level" in review.digest()


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
    assert record.question_ladder[0].startswith("White to move")
    assert "Level 0 or Level 1" in review.digest()
    assert "Recommended diagnosis" in review.digest()
    assert record.explanation_primitives["played_move"] == "Qxf7+"
    assert record.explanation_primitives["played_move_label"] == "3. Qxf7+"
    assert "best_move_label" in record.explanation_primitives
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
