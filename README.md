# Laya DOOM

Laya DOOM is a V1 autonomous DOOM combat agent. It runs ViZDoom, converts structured game state into a typed domain observation, asks Laya-MLX for one immediate action, validates that action, executes it, and records lightweight telemetry.

V1 deliberately uses structured ViZDoom state only. It does not send screenshots to Laya, use computer vision, train reinforcement learning policies, call cloud LLMs, use RAG, or attempt a full DOOM campaign.

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

The default scenario is `scenarios/v1_basic.cfg`, which references ViZDoom's bundled `basic.wad`. It exposes only four buttons:

- `MOVE_FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `ATTACK`

Those map to the V1 domain actions: `move_forward`, `turn_left`, `turn_right`, `shoot`, and `noop`.

## Running V1

```bash
uv run python -m laya_doom
```

Useful options:

```bash
uv run python -m laya_doom --help
uv run python -m laya_doom --episodes 3 --debug
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

## Current Limitations

- V1 uses structured state only; no screenshots or visual observations.
- Enemy visibility comes from ViZDoom labels when available.
- Enemy distance and direction are reported only when position variables and label positions are available.
- The scenario is intentionally tiny and meant to test `observe -> decide -> act`, not level completion.
