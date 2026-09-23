from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from laya_doom.actions.mapper import ViZDoomActionMapper
from laya_doom.actions.models import DoomAction
from laya_doom.environment.base import StepResult
from laya_doom.observation.builder import ObservationBuilder
from laya_doom.observation.models import Observation


class ViZDoomEnvironment:
    def __init__(self, scenario_path: Path, *, window_visible: bool = True) -> None:
        try:
            import vizdoom as vzd
        except ImportError as exc:
            msg = "ViZDoom is required to run the environment. Install it with `uv sync`."
            raise RuntimeError(msg) from exc

        self._vzd = vzd
        self._scenario_path = scenario_path
        self._game = vzd.DoomGame()
        self._game.load_config(str(scenario_path))
        self._game.set_window_visible(window_visible)
        bundled_wad = self._bundled_asset_for_config(
            scenario_path, "doom_scenario_path", "scenarios"
        )
        if bundled_wad is not None:
            self._game.set_doom_scenario_path(str(bundled_wad))
        bundled_iwad = self._bundled_asset_for_config(scenario_path, "doom_game_path", "")
        if bundled_iwad is not None:
            self._game.set_doom_game_path(str(bundled_iwad))
        self._game.set_labels_buffer_enabled(True)
        self._game.init()
        button_order = tuple(button.name for button in self._game.get_available_buttons())
        self._mapper = ViZDoomActionMapper(button_order)
        self._builder = ObservationBuilder(vzd)

    def _bundled_asset_for_config(
        self, scenario_path: Path, setting: str, package_subdirectory: str
    ) -> Path | None:
        """Resolve a missing WAD name against ViZDoom's packaged assets."""
        config = scenario_path.read_text(encoding="utf-8")
        match = re.search(rf"^\s*{setting}\s*=\s*([^\s#]+)", config, re.MULTILINE)
        if match is None:
            return None

        wad = Path(match.group(1))
        if wad.is_absolute() or (scenario_path.parent / wad).exists():
            return None

        bundled_wad = Path(self._vzd.__file__).resolve().parent / package_subdirectory / wad.name
        return bundled_wad if bundled_wad.is_file() else None

    @property
    def available_actions(self) -> tuple[DoomAction, ...]:
        return self._mapper.available_actions

    def reset(self) -> None:
        self._game.new_episode()

    def observe(self) -> Observation:
        state = self._game.get_state()
        return self._builder.from_vizdoom(state, self._game)

    def frame(self) -> Any | None:
        """Return the current RGB game frame for an optional native monitor."""
        state = self._game.get_state()
        return None if state is None else state.screen_buffer

    def step(self, action: DoomAction) -> StepResult:
        reward = self._game.make_action(self._mapper.to_vizdoom(action))
        return StepResult(reward=float(reward), finished=self.is_finished())

    def is_finished(self) -> bool:
        return bool(self._game.is_episode_finished())

    def close(self) -> None:
        self._game.close()
