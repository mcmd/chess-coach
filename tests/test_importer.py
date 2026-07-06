from app.importer import import_pgn_text
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
