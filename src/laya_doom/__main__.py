from __future__ import annotations

import argparse
import logging
from pathlib import Path

from laya_doom.actions.validator import ActionValidator
from laya_doom.agent.agent import DoomAgent
from laya_doom.config.settings import AppSettings
from laya_doom.decision.laya import LayaDecisionEngine, LayaRuntimeUnavailable
from laya_doom.environment.vizdoom import ViZDoomEnvironment
from laya_doom.telemetry.events import JsonlTelemetryRecorder, NullTelemetryRecorder


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = AppSettings(
        scenario=args.scenario,
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

    try:
        decision_engine = LayaDecisionEngine(model=settings.model, dtype=settings.dtype)
    except LayaRuntimeUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    telemetry = (
        JsonlTelemetryRecorder.create(settings.telemetry_root)
        if settings.telemetry_enabled
        else NullTelemetryRecorder()
    )
    environment = ViZDoomEnvironment(settings.scenario)
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
                "episode_finished index=%s steps=%s total_reward=%.3f",
                episode_index,
                result.steps,
                result.total_reward,
            )
    finally:
        environment.close()
        telemetry.close()


if __name__ == "__main__":
    main()
