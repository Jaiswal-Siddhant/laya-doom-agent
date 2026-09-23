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

The default scenario is `scenarios/continuous_combat.cfg`, which references ViZDoom's bundled
`defend_the_center.wad`. It continuously spawns enemies until the episode ends and exposes three
combat buttons:

- `TURN_LEFT`
- `TURN_RIGHT`
- `ATTACK`

Those map to the normal-mode actions: `turn_left`, `turn_right`, `shoot`, and `noop`. The previous
single-enemy basic map remains available through `--scenario scenarios/v1_basic.cfg`.

## Running V1

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

## Running a full level

ViZDoom bundles the open-source FreeDoom campaign. Start its first level with:

```bash
uv run python -m laya_doom --freedoom
```

This mode enables movement, strafing, turning, firing, and `use` actions, with controls
mapped from the selected scenario rather than relying on the V1 four-button layout. To
run an owned DOOM or DOOM II IWAD, provide a matching ViZDoom config through `--scenario`.

## Navigation

When no enemy is visible, the agent uses a local navigation map rather than repeatedly
asking the action model to explore. It marks cells reached by successful movement as
traversable, marks a forward direction blocked after three failed movement attempts, and
uses A* to route to the nearest unexplored frontier. Combat decisions remain with Laya.
When forward movement repeatedly fails at a corner, it first follows the wall with a committed
turn-and-advance maneuver, then replans from its new position.

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
