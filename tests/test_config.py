from __future__ import annotations

import importlib

from _pytest.monkeypatch import MonkeyPatch

import config as config_module


def test_config_env_override(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INDEXIUM_VIDEO_DIR", "videos")
    cfg_mod = importlib.reload(config_module)
    cfg = cfg_mod.Config()
    assert cfg.VIDEO_DIR == "videos"


def test_nfo_config_defaults() -> None:
    """Test NFO-related config defaults."""
    cfg = config_module.Config()
    assert cfg.NFO_REMOVE_STALE_ACTORS is True
    assert cfg.NFO_BACKUP_MAX_AGE_DAYS == 30
