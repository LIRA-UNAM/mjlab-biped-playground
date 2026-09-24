"""Tests for the sequential training queue script."""

import sys
from pathlib import Path

import pytest
from playground.scripts.train_queue import (
  ProgressParser,
  QueueTask,
  load_queue,
  parse_timedelta,
  validate_tasks,
)


def _write(tmp_path: Path, text: str) -> Path:
  path = tmp_path / "queue.yaml"
  path.write_text(text)
  return path


def test_load_queue_applies_defaults_and_overrides(tmp_path: Path):
  path = _write(
    tmp_path,
    """
defaults:
  num_envs: 4096
  extra_args: ["--agent.seed", "1"]
stop_on_failure: true
tasks:
  - task: A
  - task: B
    num_envs: 128
    extra_args: []
  - C
""",
  )
  cfg = load_queue(path)
  assert cfg.stop_on_failure
  assert [(t.task, t.num_envs, t.extra_args) for t in cfg.tasks] == [
    ("A", 4096, ["--agent.seed", "1"]),
    ("B", 128, []),
    ("C", 4096, ["--agent.seed", "1"]),
  ]


def test_load_queue_rejects_bad_input(tmp_path: Path):
  with pytest.raises(ValueError):
    load_queue(_write(tmp_path, "tasks: []"))
  with pytest.raises(ValueError, match="unknown keys"):
    load_queue(_write(tmp_path, "tasks:\n  - task: A\n    num_env: 4\n"))


def test_command():
  qt = QueueTask("A", 64, ["--agent.max-iterations", "20"])
  assert qt.command() == [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    "A",
    "--env.scene.num-envs",
    "64",
    "--agent.max-iterations",
    "20",
  ]
  assert "--env.scene.num-envs" not in QueueTask("A").command()


def test_validate_tasks():
  validate_tasks([QueueTask("A")], ["A", "B"])
  with pytest.raises(ValueError, match="Nope"):
    validate_tasks([QueueTask("A"), QueueTask("Nope")], ["A"])


@pytest.mark.parametrize(
  "text,expected",
  [
    ("0:00:00", 0),
    ("1:02:03", 3723),
    ("1 day, 2:03:04", 93784),
    ("3 days, 0:00:01", 259201),
    ("garbage", None),
  ],
)
def test_parse_timedelta(text: str, expected: float | None):
  assert parse_timedelta(text) == expected


def test_progress_parser_from_log_block():
  pad = 35
  block = [
    f"{'Iteration time:':>{pad}} 0.50s\n",
    f"{'Time elapsed:':>{pad}} 0:15:00\n",
    f"{'ETA:':>{pad}} 0:45:00\n",
  ]
  parser = ProgressParser()
  results = [parser.feed(line) for line in block]
  assert results[:2] == [None, None]
  assert results[2] == pytest.approx(0.25)
  assert parser.elapsed == "0:15:00"
  assert parser.eta == "0:45:00"
