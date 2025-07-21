import importlib
import os

import config as config_module


def test_config_env_override(monkeypatch):
    monkeypatch.setenv('INDEXIUM_VIDEO_DIR', 'videos')
    cfg_mod = importlib.reload(config_module)
    cfg = cfg_mod.Config()
    assert cfg.VIDEO_DIR == 'videos'
