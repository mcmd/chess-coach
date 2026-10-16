import json

from app.config import load_config


def test_load_config_reads_nested_engine_section(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "user_elo": 1150,
        "language": "English",
        "stockfish_path": "stockfish",
        "gemini_model": "test-model",
        "engine": {
            "review_eval_time": 1.25,
            "agenda_best_depth": 21,
            "agenda_best_time": 7.0,
            "agenda_punishment_depth": 19,
            "agenda_punishment_time": 5.0,
            "threat_depth": 18,
            "threat_time": 4.0,
            "threat_multipv": 3,
            "practicality_depth": 17,
            "practicality_time": 3.5,
            "batch_eval_time": 0.45,
        },
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["coach.py"])

    config = load_config()

    assert config.review_eval_time == 1.25
    assert config.agenda_best_depth == 21
    assert config.agenda_best_time == 7.0
    assert config.agenda_punishment_depth == 19
    assert config.agenda_punishment_time == 5.0
    assert config.threat_depth == 18
    assert config.threat_time == 4.0
    assert config.threat_multipv == 3
    assert config.practicality_depth == 17
    assert config.practicality_time == 3.5
    assert config.batch_eval_time == 0.45


def test_valid_profile_name_rejects_path_escapes():
    from app.config import _valid_profile_name
    assert _valid_profile_name("mcmd")
    assert not _valid_profile_name("../outside")
    assert not _valid_profile_name("a/b")
    assert not _valid_profile_name("..")
    assert not _valid_profile_name("../outside")
    assert not _valid_profile_name("")


def test_router_is_on_by_default_and_can_be_turned_off(tmp_path, monkeypatch):
    # Default since 2026-10-15 (decisive pair ec0yrv9l/sacvi7je: same quality, 17% fewer Pro calls).
    # An explicit null, "" or "off" turns it off, which the live eval's baseline arm needs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["coach.py"])
    for value, expected in (("absent", "gemini-3.8-flash"), (None, None), ("", None), ("off", None),
                            ("gemini-x", "gemini-x")):
        settings = {"stockfish_path": "stockfish"} if value == "absent" else \
            {"stockfish_path": "stockfish", "router_model": value}
        (tmp_path / "config.json").write_text(json.dumps(settings), encoding="utf-8")
        assert load_config().router_model == expected, value
