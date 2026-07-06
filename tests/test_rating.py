from app import external
from app.rating import (current_rating_enrichment, perf_from_time_control,
                        platform_from_headers, rating_offer_note)
from app.review import review_game
from tests.test_review import PGN, _scripted_engines, _Cfg


def _rated_pgn(site="https://lichess.org/abc12345", time_control="600+0"):
    return PGN.replace('[Event "Test"]', f'[Event "Test"]\n[Site "{site}"]\n[TimeControl "{time_control}"]')


def test_platform_and_perf_detection():
    assert platform_from_headers({"Site": "https://lichess.org/abc12345"}) == "lichess"
    assert platform_from_headers({"Site": "https://www.chess.com/game/live/1"}) == "chesscom"
    assert platform_from_headers({"Site": "?"}) is None
    assert perf_from_time_control({"TimeControl": "60+0"}) == "bullet"
    assert perf_from_time_control({"TimeControl": "300+3"}) == "blitz"
    assert perf_from_time_control({"TimeControl": "600+0"}) == "rapid"
    assert perf_from_time_control({"TimeControl": "1800+0"}) == "classical"


def test_lichess_current_rating_enrichment_offers_update(monkeypatch):
    monkeypatch.setattr(external, "lichess_user_profile",
                        lambda username: {"perfs": {"rapid": {"rating": 1420}}})
    review = review_game(_rated_pgn(), _scripted_engines(), _Cfg())

    enrichment = current_rating_enrichment(review, _Cfg())

    assert enrichment["platform"] == "lichess"
    assert enrichment["username"] == "mcmd"
    assert enrichment["perf"] == "rapid"
    assert enrichment["current_rating"] == 1420
    assert enrichment["configured_elo"] == 1150
    assert enrichment["offer_update"] is True
    assert "Do NOT change conversation_elo silently" in rating_offer_note(enrichment)


def test_current_rating_enrichment_degrades_silently_offline(monkeypatch):
    monkeypatch.setattr(external, "lichess_user_profile", lambda username: None)
    review = review_game(_rated_pgn(), _scripted_engines(), _Cfg())

    assert current_rating_enrichment(review, _Cfg()) is None


def test_current_rating_enrichment_respects_online_database_switch(monkeypatch):
    calls = []
    monkeypatch.setattr(external, "lichess_user_profile", lambda username: calls.append(username))

    class C(_Cfg):
        use_online_databases = False

    review = review_game(_rated_pgn(), _scripted_engines(), C())

    assert current_rating_enrichment(review, C()) is None
    assert calls == []


def test_chesscom_current_rating_enrichment(monkeypatch):
    monkeypatch.setattr(external, "chesscom_player_stats",
                        lambda username: {"chess_blitz": {"last": {"rating": 1275}}})
    review = review_game(
        _rated_pgn(site="https://www.chess.com/game/live/123", time_control="300+0"),
        _scripted_engines(),
        _Cfg(),
    )

    enrichment = current_rating_enrichment(review, _Cfg())

    assert enrichment["platform"] == "chesscom"
    assert enrichment["perf"] == "blitz"
    assert enrichment["current_rating"] == 1275


def test_platform_specific_handle_mismatch_suppresses_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr(external, "lichess_user_profile", lambda username: calls.append(username))

    class C(_Cfg):
        usernames = {"lichess": ["other_handle"]}

    review = review_game(_rated_pgn(), _scripted_engines(), C())

    assert current_rating_enrichment(review, C()) is None
    assert calls == []
