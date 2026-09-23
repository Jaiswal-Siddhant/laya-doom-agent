# Laya DOOM

Laya DOOM is a local autonomous DOOM agent built with ViZDoom and Laya-MLX. It turns structured game state into a typed observation, selects one safe action at a time, executes it, and records lightweight telemetry. In **Freedom Mode**, it can explore the bundled FreeDoom E1M1 level with local map-building and A\* path finding.

<video src="static/v1_recording.mp4" autoplay muted loop playsinline controls aria-label="Laya DOOM Freedom Mode recording"></video>

[Watch the V1 recording](static/v1_recording.mp4)

The agent deliberately uses structured ViZDoom state rather than screenshots or computer vision. It runs Laya-MLX locally on Apple Silicon—no cloud LLMs, RAG, or reinforcement-learning training is involved.

## Why this exists

Laya DOOM is a controlled testbed for Laya's decision model. ViZDoom provides a fast, repeatable environment in which the model receives a compact, typed game-state snapshot and must choose its next action from a constrained set. This makes it practical to inspect whether Laya can make sensible short-horizon combat decisions—such as turning toward a detected enemy, firing when aligned, or moving safely—while measuring the choice, confidence when available, latency, reward, and episode outcome.

The project intentionally separates deterministic responsibilities from model judgment: observation building, action validation, environment control, telemetry, and exploration routing are conventional code; Laya decides the immediate combat action. That boundary makes failures easier to reproduce and diagnose than in an end-to-end vision or full-game agent.

## Architecture

```text
ViZDoom -> ObservationBuilder -> Observation -> LayaDecisionEngine
       -> AgentDecision -> ActionValidator -> DoomAction -> ViZDoomActionMapper
       -> ViZDoom
```

The core boundaries live under `src/laya_doom/`:

- `environment/`: small environment interface and ViZDoom adapter
- `observation/`: typed observations and deterministic serialization
- `decision/`: Laya-MLX integration
- `actions/`: V1 action enum, validator, and ViZDoom button mapping
- `agent/`: synchronous episode loop and prompt text
- `telemetry/`: optional JSON Lines event writer
- `config/`: typed defaults

## Requirements

- Python 3.11+
- `uv`
- ViZDoom runtime support
- Apple Silicon/macOS for Laya-MLX local MLX inference

Laya-MLX currently documents the API as:

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx", dtype="float16")
result = agent.predict(state, questions)
```

This project keeps that usage isolated in `LayaDecisionEngine`.

## Installation

```bash
uv sync --extra dev
```

On first Laya use, model weights may be downloaded by the Laya/Hugging Face stack. Later inference is local.

## ViZDoom Setup

The default scenario is `scenarios/continuous_combat.cfg`, which references ViZDoom's bundled
`defend_the_center.wad`. It continuously spawns enemies until the episode ends and exposes three
combat buttons:

- `TURN_LEFT`
- `TURN_RIGHT`
- `ATTACK`

Those map to the normal-mode actions: `turn_left`, `turn_right`, `shoot`, and `noop`. The previous
single-enemy basic map remains available through `--scenario scenarios/v1_basic.cfg`.

## Run the combat scenario

```bash
uv run python -m laya_doom
```

### Native game monitor

Install the optional desktop UI once, then run the agent with a single native window. It renders
the live Doom frame on the left, decision/model statistics on the right, and a recent-decision
log along the bottom.

```bash
uv sync --extra ui
uv run laya-doom --model aac6fef/laya-mlx --ui
```

## Freedom Mode: explore a full level

Freedom Mode runs the bundled open-source FreeDoom campaign's first level (E1M1):

```bash
uv run laya-doom --freedom
```

It enables movement, strafing, turning, firing, and `use`, and maps only the controls exposed by the selected ViZDoom scenario. `--freedoom` remains accepted as an alias. To run an owned DOOM or DOOM II IWAD, provide a matching ViZDoom config with `--scenario`.

## Navigation and A\* path finding

When no enemy is visible, the agent explores with a local navigator instead of spending model calls on aimless movement. It discretizes the player's position into a sparse 32-unit grid, learns which cells are traversable, and identifies **frontiers**: known reachable cells next to unknown space.

For each frontier, A\* searches the known traversable graph using Manhattan distance as its heuristic. The navigator selects the shortest resulting path, steers toward its next waypoint, and asks Laya to resume combat decisions as soon as there is an enemy to engage.

```text
known cells -> reachable frontier -> A* shortest path -> next waypoint -> turn or advance
```

If three forward moves fail to produce meaningful position change, the cell ahead is marked blocked. The agent then performs a short alternating wall-follow recovery (turn, then advance) and replans around the newly discovered obstacle. This keeps exploration responsive without pretending the map is known in advance.

Useful options:

```bash
uv run python -m laya_doom --help
uv run python -m laya_doom --episodes 3 --debug
uv run python -m laya_doom --freedom --ui
uv run python -m laya_doom --model aac6fef/laya-multilingual-mlx --dtype float16
uv run python -m laya_doom --no-telemetry
```

## Agent Loop

For each episode, the agent:

1. resets the environment
2. observes structured ViZDoom state
3. builds an `Observation`
4. serializes that observation deterministically
5. asks Laya-MLX one typed `choice` question
6. validates the returned action
7. executes the mapped ViZDoom action
8. records telemetry
9. repeats until the episode ends

Invalid or malformed model output falls back to `noop` and never reaches ViZDoom as raw commands.

## Telemetry

Telemetry is enabled by default and writes JSON Lines:

```text
runs/<timestamp>/events.jsonl
```

Each event includes health, armor, ammo, enemy visibility, selected action, probability when available, measured inference latency, reward, and finished state.

## Testing

Unit tests do not require Laya-MLX or ViZDoom:

```bash
uv run pytest tests/unit
```

Integration tests are separated and skipped automatically when ViZDoom is unavailable:

```bash
uv run pytest tests/integration
```

Code quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## Current limitations

- V1 uses structured state only; no screenshots or visual observations.
- Enemy visibility comes from ViZDoom labels when available.
- Enemy distance and direction are reported only when position variables and label positions are available.
- The V1 combat scenario is intentionally tiny and meant to test `observe -> decide -> act`.
- Navigation is a local, discovered map—not global level knowledge—and is only available in scenarios that expose movement and position data.
