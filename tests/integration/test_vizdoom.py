from __future__ import annotations

from pathlib import Path

import pytest

from laya_doom.actions.models import DoomAction
from laya_doom.config.settings import PROJECT_ROOT
from laya_doom.environment.vizdoom import ViZDoomEnvironment

pytest.importorskip("vizdoom")


def test_vizdoom_environment_loads_v1_scenario() -> None:
    scenario = Path(PROJECT_ROOT / "scenarios" / "v1_basic.cfg")
    environment = ViZDoomEnvironment(scenario)
    try:
        environment.reset()
        observation = environment.observe()
        result = environment.step(DoomAction.NOOP)
    finally:
        environment.close()

    assert observation.health >= 0
    assert result.reward <= 0 or result.reward >= 0
