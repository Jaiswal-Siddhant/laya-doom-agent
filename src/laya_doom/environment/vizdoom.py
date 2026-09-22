from __future__ import annotations

from pathlib import Path

from laya_doom.actions.mapper import ViZDoomActionMapper
from laya_doom.actions.models import DoomAction
from laya_doom.environment.base import StepResult
from laya_doom.observation.builder import ObservationBuilder
from laya_doom.observation.models import Observation


class ViZDoomEnvironment:
    def __init__(self, scenario_path: Path) -> None:
        try:
            import vizdoom as vzd
        except ImportError as exc:
            msg = "ViZDoom is required to run the environment. Install it with `uv sync`."
            raise RuntimeError(msg) from exc

        self._vzd = vzd
        self._scenario_path = scenario_path
        self._game = vzd.DoomGame()
        self._game.load_config(str(scenario_path))
        self._game.set_labels_buffer_enabled(True)
        self._game.init()
        self._mapper = ViZDoomActionMapper()
        self._builder = ObservationBuilder(vzd)

    def reset(self) -> None:
        self._game.new_episode()

    def observe(self) -> Observation:
        state = self._game.get_state()
        return self._builder.from_vizdoom(state, self._game)

    def step(self, action: DoomAction) -> StepResult:
        reward = self._game.make_action(self._mapper.to_vizdoom(action))
        return StepResult(reward=float(reward), finished=self.is_finished())

    def is_finished(self) -> bool:
        return bool(self._game.is_episode_finished())

    def close(self) -> None:
        self._game.close()
