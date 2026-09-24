"""Run several training tasks back to back from a YAML queue file.

Tasks run strictly one at a time; the next starts as soon as the previous
exits. Training stdout is hidden from the terminal (it goes to per-task log
files) and each task gets a progress bar derived from the ``Time elapsed`` /
``ETA`` lines printed by the rsl_rl logger.

Usage:
  uv run train_queue queues/train_queue.example.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

_ELAPSED_RE = re.compile(r"Time elapsed:\s+(.+?)\s*$")
_ETA_RE = re.compile(r"ETA:\s+(.+?)\s*$")
_TIMEDELTA_RE = re.compile(
  r"^(?:(?P<days>-?\d+) days?, )?(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})$"
)


@dataclass
class QueueTask:
  task: str
  num_envs: int | None = None
  extra_args: list[str] = field(default_factory=list)

  def command(self) -> list[str]:
    cmd = [sys.executable, "-m", "mjlab.scripts.train", self.task]
    if self.num_envs is not None:
      cmd += ["--env.scene.num-envs", str(self.num_envs)]
    return cmd + self.extra_args


@dataclass
class QueueConfig:
  tasks: list[QueueTask]
  stop_on_failure: bool = False


def load_queue(path: Path) -> QueueConfig:
  """Parse a queue YAML file, applying ``defaults`` to every task."""
  raw = yaml.safe_load(path.read_text()) or {}
  if not isinstance(raw, dict) or not raw.get("tasks"):
    raise ValueError(f"{path}: expected a mapping with a non-empty 'tasks' list")

  defaults = raw.get("defaults") or {}
  tasks: list[QueueTask] = []
  for i, entry in enumerate(raw["tasks"]):
    if isinstance(entry, str):
      entry = {"task": entry}
    if not isinstance(entry, dict) or "task" not in entry:
      raise ValueError(f"{path}: tasks[{i}] must be a task id or have a 'task' key")
    unknown = set(entry) - {"task", "num_envs", "extra_args"}
    if unknown:
      raise ValueError(f"{path}: tasks[{i}] has unknown keys {sorted(unknown)}")
    num_envs = entry.get("num_envs", defaults.get("num_envs"))
    extra_args = entry.get("extra_args", defaults.get("extra_args")) or []
    tasks.append(
      QueueTask(
        task=str(entry["task"]),
        num_envs=int(num_envs) if num_envs is not None else None,
        extra_args=[str(a) for a in extra_args],
      )
    )
  return QueueConfig(tasks=tasks, stop_on_failure=bool(raw.get("stop_on_failure")))


def validate_tasks(tasks: list[QueueTask], known: list[str]) -> None:
  missing = [t.task for t in tasks if t.task not in known]
  if missing:
    raise ValueError(
      f"Unknown task(s): {', '.join(missing)}. Run `uv run list_envs` to see options."
    )


def parse_timedelta(text: str) -> float | None:
  """Parse ``str(datetime.timedelta)`` output (e.g. '1 day, 2:03:04') to seconds."""
  m = _TIMEDELTA_RE.match(text.strip())
  if m is None:
    return None
  days = int(m["days"]) if m["days"] else 0
  return days * 86400 + int(m["h"]) * 3600 + int(m["m"]) * 60 + int(m["s"])


class ProgressParser:
  """Tracks elapsed/ETA from training log lines."""

  def __init__(self) -> None:
    self.elapsed: str | None = None
    self.eta: str | None = None
    self._elapsed_s: float | None = None

  def feed(self, line: str) -> float | None:
    """Consume a log line; return completion fraction when an ETA line arrives."""
    if m := _ELAPSED_RE.search(line):
      self.elapsed = m[1]
      self._elapsed_s = parse_timedelta(m[1])
      return None
    if m := _ETA_RE.search(line):
      self.eta = m[1]
      eta_s = parse_timedelta(m[1])
      if self._elapsed_s is None or eta_s is None:
        return None
      total = self._elapsed_s + eta_s
      return 1.0 if total <= 0 else self._elapsed_s / total
    return None


def _slug(task: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", task)


def _stop_process(proc: subprocess.Popen, timeout: float = 30.0) -> None:
  """Interrupt the child's whole process group, escalating if it hangs."""
  for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
    if proc.poll() is not None:
      return
    try:
      os.killpg(proc.pid, sig)
    except ProcessLookupError:
      return
    try:
      proc.wait(timeout=timeout)
      return
    except subprocess.TimeoutExpired:
      continue


def _fmt_duration(seconds: float) -> str:
  s = int(seconds)
  return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run training tasks sequentially from a YAML queue file."
  )
  parser.add_argument("queue", type=Path, help="Path to the queue YAML file.")
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path("logs/queue"),
    help="Directory for per-task stdout logs (default: logs/queue).",
  )
  args = parser.parse_args()

  from rich.console import Console
  from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
  from rich.table import Table

  console = Console()
  cfg = load_queue(args.queue)

  with console.status("Loading task registry..."):
    import mjlab.tasks  # noqa: F401  (registers tasks, incl. entry points)
    from mjlab.tasks.registry import list_tasks

    validate_tasks(cfg.tasks, list_tasks())

  run_dir = args.log_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir.mkdir(parents=True, exist_ok=True)
  console.print(f"Queue: {len(cfg.tasks)} task(s). Logs: [bold]{run_dir}[/]")

  progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(bar_width=None),
    TaskProgressColumn(),
    TextColumn("{task.fields[info]}"),
    console=console,
    expand=True,
  )
  width = max(len(t.task) for t in cfg.tasks)
  overall = progress.add_task(
    f"[bold]{'Overall':<{width}}[/]",
    total=len(cfg.tasks),
    info=f"0/{len(cfg.tasks)} tasks",
  )
  rows = [
    progress.add_task(f"{t.task:<{width}}", total=1.0, info="[dim]queued", start=False)
    for t in cfg.tasks
  ]

  results: list[tuple[QueueTask, str, float, Path | None]] = []
  interrupted = False
  with progress:
    for i, (qt, row) in enumerate(zip(cfg.tasks, rows, strict=True)):
      if interrupted:
        progress.update(row, info="[yellow]skipped")
        results.append((qt, "skipped", 0.0, None))
        continue

      log_path = run_dir / f"{i:02d}_{_slug(qt.task)}.log"
      progress.start_task(row)
      progress.update(row, info="[cyan]starting…")
      tracker = ProgressParser()
      start = time.monotonic()
      with open(log_path, "w") as log:
        log.write("$ " + " ".join(qt.command()) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
          qt.command(),
          stdin=subprocess.DEVNULL,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          bufsize=1,
          env={**os.environ, "PYTHONUNBUFFERED": "1"},
          start_new_session=True,
        )
        try:
          assert proc.stdout is not None
          for line in proc.stdout:
            log.write(line)
            frac = tracker.feed(line)
            if frac is not None:
              progress.update(
                row,
                completed=frac,
                info=f"elapsed {tracker.elapsed} · ETA {tracker.eta}",
              )
          returncode = proc.wait()
        except KeyboardInterrupt:
          interrupted = True
          progress.update(row, info="[yellow]stopping…")
          _stop_process(proc)
          returncode = proc.returncode

      wall = time.monotonic() - start
      if interrupted:
        status = "interrupted"
        progress.update(row, info="[yellow]interrupted")
      elif returncode == 0:
        status = "done"
        progress.update(
          row, completed=1.0, info=f"[green]✓ done[/] in {_fmt_duration(wall)}"
        )
      else:
        status = f"failed ({returncode})"
        progress.update(row, info=f"[red]✗ failed ({returncode})[/] see {log_path}")
      results.append((qt, status, wall, log_path))

      progress.update(overall, advance=1, info=f"{i + 1}/{len(cfg.tasks)} tasks")
      if status.startswith("failed") and cfg.stop_on_failure:
        interrupted = True

  table = Table(title="Queue summary")
  table.add_column("Task")
  table.add_column("num_envs", justify="right")
  table.add_column("Status")
  table.add_column("Wall time", justify="right")
  table.add_column("Log")
  colors = {"done": "green", "skipped": "yellow", "interrupted": "yellow"}
  for qt, status, wall, log_path in results:
    color = colors.get(status, "red")
    table.add_row(
      qt.task,
      str(qt.num_envs) if qt.num_envs is not None else "default",
      f"[{color}]{status}[/]",
      _fmt_duration(wall) if log_path else "-",
      str(log_path) if log_path else "-",
    )
  console.print(table)

  if any(status != "done" for _, status, _, _ in results):
    sys.exit(1)


if __name__ == "__main__":
  main()
