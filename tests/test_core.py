"""core 单测：config / seed / errors。作者: 晨星"""

import numpy as np
import pytest

from semiforge.core.config import Config, load_config
from semiforge.core.errors import ConfigError
from semiforge.core.seed import set_all


def test_config_default_valid():
    Config().validate()


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_SEMIFORGE_N_SAMPLES", "300")
    monkeypatch.setenv("ENV_SEMIFORGE_GACS_ALPHA", "0.4")
    monkeypatch.setenv("ENV_SEMIFORGE_DATASETS", "moons, circles")
    monkeypatch.setenv("ENV_SEMIFORGE_GACS_BALANCE", "false")
    cfg = load_config()
    assert cfg.n_samples == 300
    assert cfg.gacs_alpha == 0.4
    assert cfg.datasets == ["moons", "circles"]
    assert cfg.gacs_balance is False


def test_config_invalid_raises():
    cfg = Config()
    cfg.n_samples = 10
    with pytest.raises(ConfigError):
        cfg.validate()


def test_seed_deterministic():
    set_all(7)
    a = np.random.randint(0, 100, 5)
    set_all(7)
    b = np.random.randint(0, 100, 5)
    assert (a == b).all()
