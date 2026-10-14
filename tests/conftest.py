import pytest

from doomscent.config import load_config


@pytest.fixture(scope="session")
def v0_cfg():
    return load_config("v0_defend_center.yaml")
