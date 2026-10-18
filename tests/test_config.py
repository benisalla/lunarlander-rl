import pytest

from lunarlander_rl.config import METHODS, Config, make_config, parse_override


def test_every_preset_builds_a_config():
    for name, preset in METHODS.items():
        cfg = make_config(name)
        assert cfg.method == name and cfg.algo == preset.algo and cfg.continuous == preset.continuous


def test_overrides_and_unknown_fields():
    cfg = make_config("dqn", total_steps=123, hidden=[32, 32])
    assert cfg.total_steps == 123 and cfg.hidden == (32, 32)
    with pytest.raises(KeyError):
        make_config("dqn", not_a_field=1)
    with pytest.raises(KeyError):
        make_config("not-a-method")


def test_parse_override_casts_to_field_type():
    assert parse_override("lr=1e-3") == ("lr", 1e-3)
    assert parse_override("n_step=3") == ("n_step", 3)
    assert parse_override("per=true") == ("per", True)
    assert parse_override("hidden=64,64") == ("hidden", (64, 64))


def test_config_json_roundtrip(tmp_path):
    cfg = make_config("dueling-dqn", seed=7)
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == cfg
