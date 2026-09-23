from __future__ import annotations

import argparse
import logging
from pathlib import Path

from laya_doom.actions.validator import ActionValidator
from laya_doom.agent.agent import DoomAgent
from laya_doom.config.settings import PROJECT_ROOT, AppSettings
from laya_doom.decision.laya import LayaDecisionEngine, LayaRuntimeUnavailable
from laya_doom.environment.vizdoom import ViZDoomEnvironment
from laya_doom.telemetry.events import (
    JsonlTelemetryRecorder,
    NullTelemetryRecorder,
    TelemetryRecorder,
)
from laya_doom.ui.window import NativeMonitor, NativeMonitorTelemetryRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-doom",
        description="Run the V1 Laya-MLX + ViZDoom autonomous DOOM combat agent.",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=AppSettings().scenario,
        help="Path to a ViZDoom .cfg scenario file.",
    )
    parser.add_argument(
        "--freedoom",
        action="store_true",
        help="Run the bundled FreeDoom E1M1 level instead of the basic combat map.",
    )
    parser.add_argument(
        "--model",
        default=AppSettings().model,
        help="Laya-MLX model id or local checkpoint path.",
    )
    parser.add_argument(
        "--dtype",
        default=AppSettings().dtype,
        choices=["float16", "float32", "bfloat16"],
        help="MLX inference dtype passed to laya.load().",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=AppSettings().episodes,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable JSONL telemetry under runs/.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Show the native game monitor with a live frame and decision telemetry.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = AppSettings(
        scenario=PROJECT_ROOT / "scenarios" / "freedoom1.cfg" if args.freedoom else args.scenario,
        model=args.model,
        dtype=args.dtype,
        episodes=args.episodes,
        logging_level="DEBUG" if args.debug else "INFO",
        telemetry_enabled=not args.no_telemetry,
    )
    logging.basicConfig(
        level=getattr(logging, settings.logging_level),
        format="%(levelname)s %(message)s",
    )

    telemetry: TelemetryRecorder = (
        JsonlTelemetryRecorder.create(settings.telemetry_root)
        if settings.telemetry_enabled
        else NullTelemetryRecorder()
    )
    environment = ViZDoomEnvironment(settings.scenario, window_visible=not args.ui)
    monitor = NativeMonitor(settings.model) if args.ui else None
    if monitor is not None:
        telemetry = NativeMonitorTelemetryRecorder(monitor, telemetry, environment.frame)
        logging.info("native_monitor_started")
    try:
        decision_engine = LayaDecisionEngine(
            model=settings.model, dtype=settings.dtype, actions=environment.available_actions
        )
    except LayaRuntimeUnavailable as exc:
        environment.close()
        telemetry.close()
        if monitor is not None:
            monitor.close()
        raise SystemExit(str(exc)) from exc
    agent = DoomAgent(
        environment=environment,
        decision_engine=decision_engine,
        action_validator=ActionValidator(),
        telemetry=telemetry,
    )

    try:
        for episode_index in range(settings.episodes):
            logging.info("episode_started index=%s", episode_index)
            result = agent.run_episode()
            logging.info(
                "episode_finished index=%s steps=%s total_reward=%.3f reason=%s",
                episode_index,
                result.steps,
                result.total_reward,
                result.termination_reason,
            )
            if result.termination_reason == "out_of_ammo":
                break
    finally:
        environment.close()
        telemetry.close()
        if monitor is not None:
            monitor.close()


if __name__ == "__main__":
    main()
