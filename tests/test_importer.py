from app import external
from app.importer import (import_configured_recent_games, import_pgn_text,
                          import_remote_games, should_offer_onboarding_import)
from app.memory import load_dossier
from tests.test_batch import _fixture


def test_import_local_pgn_updates_dossier_and_seeds_drills(stub_config):
    pgn, engines = _fixture()

    result = import_pgn_text(pgn, engines, stub_config, source="fixture.pgn")

    assert result.error is None
    assert result.games_seen == 3
    assert result.games_matched == 3
    assert result.games_imported == 3
    assert result.patterns == 1
    dossier = load_dossier(stub_config.dossier_file)
    assert len(dossier["imported_games"]) == 3
    assert dossier["recurring_patterns"]
    assert dossier["drill_sets"]
    assert any(p.get("source") == "imported_batch_representative"
               for p in dossier["recent_positions"])


def test_import_local_pgn_deduplicates_previously_imported_games(stub_config):
    pgn, engines = _fixture()
    first = import_pgn_text(pgn, engines, stub_config, source="fixture.pgn")
    second = import_pgn_text(pgn, engines, stub_config, source="fixture.pgn")

    assert first.error is None
    assert second.error == "No new matching games to import."
    assert second.games_duplicate == 3
    assert len(load_dossier(stub_config.dossier_file)["imported_games"]) == 3


def test_import_requires_configured_identity(stub_config):
    stub_config.player_name = None
    stub_config.usernames = {}
    pgn, engines = _fixture()

    result = import_pgn_text(pgn, engines, stub_config, source="fixture.pgn")

    assert "No configured player identity" in result.error


def test_remote_lichess_import_uses_same_dossier_pipeline(stub_config, monkeypatch):
    pgn, engines = _fixture()
    monkeypatch.setattr(external, "lichess_user_games_pgn",
                        lambda username, max_games=50: pgn)

    result = import_remote_games("lichess", "mcmd", engines, stub_config)

    assert result.error is None
    assert result.games_imported == 3
    dossier = load_dossier(stub_config.dossier_file)
    assert dossier["imported_games"][0]["source"] == "lichess:mcmd"
    assert dossier["drill_sets"]


def test_configured_remote_import_tries_supported_usernames(stub_config, monkeypatch):
    pgn, engines = _fixture()
    stub_config.usernames = {"lichess": ["mcmd"]}
    monkeypatch.setattr(external, "lichess_user_games_pgn",
                        lambda username, max_games=50: pgn)

    result = import_configured_recent_games(stub_config, engines)

    assert result.error is None
    assert result.games_imported == 3


def test_remote_import_degrades_offline(stub_config, monkeypatch):
    monkeypatch.setattr(external, "lichess_user_games_pgn",
                        lambda username, max_games=50: None)

    result = import_remote_games("lichess", "mcmd", None, stub_config)

    assert "Could not fetch" in result.error


def test_onboarding_import_offer_requires_handles_and_empty_dossier(stub_config):
    stub_config.usernames = {"lichess": ["mcmd"]}
    assert should_offer_onboarding_import(stub_config) is True

    pgn, engines = _fixture()
    import_pgn_text(pgn, engines, stub_config, source="fixture.pgn")

    assert should_offer_onboarding_import(stub_config) is False
