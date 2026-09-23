from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from laya_doom.actions.models import DoomAction
from laya_doom.decision.base import AgentDecision
from laya_doom.environment.base import StepResult
from laya_doom.observation.models import Observation
from laya_doom.telemetry.events import TelemetryRecorder


@dataclass(frozen=True)
class MonitorEvent:
    timestamp: str
    laya_action: str | None
    action: str
    latency_ms: float
    probability: float | None
    target_name: str | None
    enemy_count: int
    enemy_in_crosshair: bool
    ammo: int
    reward: float
    source: str | None


class MonitorState:
    """GUI-independent state used by the native monitor and its tests."""

    def __init__(self, model: str, max_events: int = 100) -> None:
        self.model = model
        self.events: deque[MonitorEvent] = deque(maxlen=max_events)

    def publish(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
    ) -> MonitorEvent:
        raw_response = decision.raw_response or {}
        event = MonitorEvent(
            timestamp=datetime.now(UTC).isoformat(),
            laya_action=_string_value(raw_response.get("laya_action")),
            action=action.value,
            latency_ms=decision.latency_ms,
            probability=decision.probability,
            target_name=observation.target_name,
            enemy_count=observation.visible_enemy_count,
            enemy_in_crosshair=observation.enemy_in_crosshair,
            ammo=observation.ammo,
            reward=result.reward,
            source=_string_value(raw_response.get("source")),
        )
        self.events.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        events = [asdict(event) for event in self.events]
        return {"model": self.model, "events": events, "latest": events[-1] if events else None}


class NativeMonitor:
    """A PySide6 monitor that presents the ViZDoom frame and agent decisions."""

    def __init__(self, model: str) -> None:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError as exc:
            msg = (
                "The native monitor requires PySide6-Essentials. "
                "Install it with `uv sync --extra ui`."
            )
            raise RuntimeError(msg) from exc

        self._QtCore = QtCore
        self._QtGui = QtGui
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self._state = MonitorState(model)
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("Laya DOOM Monitor")
        self._window.resize(1280, 820)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, stretch=1)

        self._frame = QtWidgets.QLabel("Waiting for the first game frame…")
        self._frame.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._frame.setMinimumSize(720, 480)
        self._frame.setStyleSheet("background:#080b10; border:1px solid #344054; color:#98a2b3;")
        top.addWidget(self._frame, stretch=4)

        sidebar = QtWidgets.QFrame()
        sidebar.setMinimumWidth(290)
        sidebar.setStyleSheet("QFrame { background:#121826; border:1px solid #344054; }")
        fields = QtWidgets.QFormLayout(sidebar)
        fields.setContentsMargins(16, 16, 16, 16)
        fields.setSpacing(12)
        self._labels: dict[str, Any] = {}
        for key, title, value in (
            ("model", "LAYA DOOM MODEL", model),
            ("laya_action", "LAYA ACTION", "—"),
            ("action", "EXECUTED", "—"),
            ("latency", "LATENCY", "—"),
            ("target", "TARGET", "—"),
            ("enemies", "ENEMIES", "—"),
            ("ammo", "AMMO", "—"),
            ("reward", "REWARD", "—"),
        ):
            label = QtWidgets.QLabel(value)
            label.setWordWrap(True)
            label.setStyleSheet("color:#f9fafb; font-family:Menlo, monospace; font-weight:600;")
            fields.addRow(f"{title}:", label)
            self._labels[key] = label
        top.addWidget(sidebar, stretch=1)

        self._history = QtWidgets.QPlainTextEdit()
        self._history.setReadOnly(True)
        self._history.setMaximumBlockCount(100)
        self._history.setMinimumHeight(175)
        self._history.setStyleSheet(
            "background:#080b10; border:1px solid #344054; color:#d0d5dd; "
            "font-family:Menlo, monospace;"
        )
        self._history.setPlainText("Waiting for Laya's first decision…")
        root.addWidget(self._history)
        self._window.setCentralWidget(central)
        self._window.show()
        self._app.processEvents()

    def publish(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
        frame: Any | None,
    ) -> None:
        event = self._state.publish(observation, decision, action, result)
        self._labels["laya_action"].setText(event.laya_action or "—")
        self._labels["action"].setText(event.action)
        self._labels["latency"].setText(f"{event.latency_ms:.1f} ms")
        self._labels["target"].setText(event.target_name or "—")
        self._labels["enemies"].setText(str(event.enemy_count))
        self._labels["ammo"].setText(str(event.ammo))
        self._labels["reward"].setText(f"{event.reward:+.2f}")
        self._history.appendPlainText(_history_line(event))
        self._show_frame(frame)
        self._app.processEvents()

    def close(self) -> None:
        self._window.close()
        self._app.processEvents()

    def _show_frame(self, frame: Any | None) -> None:
        if frame is None or getattr(frame, "ndim", 0) != 3:
            return
        height, width, channels = frame.shape
        if channels != 3:
            return
        image = self._QtGui.QImage(
            frame.data,
            width,
            height,
            frame.strides[0],
            self._QtGui.QImage.Format.Format_RGB888,
        ).copy()
        pixmap = self._QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self._frame.size(),
            self._QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            self._QtCore.Qt.TransformationMode.FastTransformation,
        )
        self._frame.setPixmap(scaled)


class NativeMonitorTelemetryRecorder:
    """Mirrors existing JSONL telemetry into the native monitor after each action."""

    def __init__(
        self,
        monitor: NativeMonitor,
        delegate: TelemetryRecorder,
        frame_provider: Callable[[], Any | None],
    ) -> None:
        self._monitor = monitor
        self._delegate = delegate
        self._frame_provider = frame_provider

    def record(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
    ) -> None:
        self._delegate.record(observation, decision, action, result)
        self._monitor.publish(observation, decision, action, result, self._frame_provider())

    def close(self) -> None:
        self._delegate.close()


def _history_line(event: MonitorEvent) -> str:
    target = event.target_name or "none"
    laya_action = event.laya_action or "—"
    return (
        f"{event.timestamp[11:19]}  laya={laya_action:<14} executed={event.action:<14} "
        f"latency={event.latency_ms:6.1f}ms target={target:<18} "
        f"enemies={event.enemy_count} ammo={event.ammo} reward={event.reward:+.2f}"
    )


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None
