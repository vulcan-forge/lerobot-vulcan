from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from lerobot.robots.sourccey.sourccey.sourccey.sourccey_host import _RelayAgentProcessManager


@dataclass
class _Config:
    relay_agent_autostart: bool = True
    relay_agent_module: str = "example.module"
    relay_agent_python_executable: str | None = "python"
    relay_agent_restart_on_exit: bool = True
    relay_agent_restart_backoff_s: float = 0.0
    relay_agent_max_restarts: int = 2


def test_start_launches_process(monkeypatch) -> None:
    popen_mock = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None
    popen_mock.return_value = proc
    monkeypatch.setattr("lerobot.robots.sourccey.sourccey.sourccey.sourccey_host.subprocess.Popen", popen_mock)

    manager = _RelayAgentProcessManager(_Config())
    manager.start()

    assert popen_mock.called


def test_poll_restarts_exited_process(monkeypatch) -> None:
    popen_mock = MagicMock()
    first_proc = MagicMock()
    first_proc.poll.return_value = 1
    second_proc = MagicMock()
    second_proc.poll.return_value = None
    popen_mock.side_effect = [first_proc, second_proc]
    monkeypatch.setattr("lerobot.robots.sourccey.sourccey.sourccey.sourccey_host.subprocess.Popen", popen_mock)

    manager = _RelayAgentProcessManager(_Config())
    manager.start()
    manager.poll()

    assert popen_mock.call_count >= 2


def test_stop_terminates_process(monkeypatch) -> None:
    popen_mock = MagicMock()
    proc = MagicMock()
    proc.poll.return_value = None
    popen_mock.return_value = proc
    monkeypatch.setattr("lerobot.robots.sourccey.sourccey.sourccey.sourccey_host.subprocess.Popen", popen_mock)

    manager = _RelayAgentProcessManager(_Config())
    manager.start()
    manager.stop()

    assert proc.terminate.called
