"""Export a trained policy checkpoint (PPO or FlashSAC) to ONNX.

Works for any mjlab task registered via `register_mjlab_task`, with a
checkpoint loaded either from a local file or downloaded from W&B. Mirrors
the two-stage CLI parsing pattern used by mjlab's own train/play scripts:
the task id is a positional argument, everything else is a tyro-parsed
`ExportConfig`.

Examples:
  uv run python export_policy.py Pumas-Velocity-Flat-Booster-T1-PPO \\
      --checkpoint-file logs/rsl_rl/t1_velocity/wandb_checkpoints/mg5wodgg/model_2999.pt

  uv run python export_policy.py Pumas-Velocity-Flat-Booster-T1-FlashSAC \\
      --wandb-run-path my-entity/t1_velocity/abcd1234
"""

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path


@dataclass(frozen=True)
class ExportConfig:
  wandb_run_path: str | None = None
  """W&B run path (e.g. 'entity/project/run_id') to pull the checkpoint
  from. Exactly one of `checkpoint_file` or `wandb_run_path` must be given."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g.
  'model_4000.pt'). Defaults to the highest-iteration checkpoint."""
  checkpoint_file: str | None = None
  """Local path to a checkpoint .pt file. Exactly one of `checkpoint_file`
  or `wandb_run_path` must be given."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs (and cached W&B checkpoint
  downloads) live."""
  export_dir: str = "export"
  """Directory the ONNX file is written into."""
  filename: str | None = None
  """ONNX output filename. Defaults to a slugified version of the task id
  (e.g. 'pumas_velocity_flat_booster_t1_ppo.onnx') so exporting different
  tasks into the same `export_dir` doesn't silently overwrite each other."""
  device: str = "cpu"
  """Device used to build the env/runner and load the checkpoint. The
  exported ONNX graph itself is always traced on cpu regardless."""


def _default_filename(task_id: str) -> str:
  """Slugify a task id into a default ONNX filename.

  E.g. 'Pumas-Velocity-Flat-Booster-T1-PPO' ->
  'pumas_velocity_flat_booster_t1_ppo.onnx'.
  """
  slug = re.sub(r"[^0-9a-zA-Z]+", "_", task_id).strip("_").lower()
  return f"{slug}.onnx"


def run_export(task_id: str, cfg: ExportConfig) -> None:
  if cfg.checkpoint_file is not None and cfg.wandb_run_path is not None:
    raise ValueError(
      "Provide exactly one of `checkpoint_file` or `wandb_run_path`, not both."
    )
  if cfg.checkpoint_file is None and cfg.wandb_run_path is None:
    raise ValueError(
      "`wandb_run_path` is required when `checkpoint_file` is not provided."
    )

  agent_cfg = load_rl_cfg(task_id)

  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    provenance = str(resume_path)
  else:
    assert cfg.wandb_run_path is not None
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
    run_id = resume_path.parent.name
    checkpoint_name = resume_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
    )
    provenance = cfg.wandb_run_path

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )

  filename = cfg.filename or _default_filename(task_id)
  runner.export_policy_to_onnx(cfg.export_dir, filename)
  onnx_path = str(Path(cfg.export_dir) / filename)

  metadata = get_base_metadata(env.unwrapped, run_path=provenance)
  attach_metadata_to_onnx(onnx_path, metadata)
  print(f"[INFO]: Exported policy to {onnx_path}")

  env.close()


def main():
  maybe_print_top_level_help("export_policy.py")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments. Load agent_cfg here too so config
  # errors for the chosen task surface at parse time, same as play.py.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    default=ExportConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
