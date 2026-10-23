"""Export a trained checkpoint to ONNX with the mjlab deployment metadata.

The training runner writes the ONNX next to each checkpoint it saves, but
checkpoints downloaded from wandb are plain rsl_rl ``.pt`` files. This rebuilds
the task (one environment, CPU), loads the checkpoint into the task's runner and
exports the policy exactly as ``VelocityOnPolicyRunner.save`` does: the actor as
ONNX plus the metadata (joint names, default pose, action scale, gains,
observation terms) that the ROS policy nodes check at load time.

Usage:
  uv run export_onnx Pumas-Velocity-Flat-Roomie-V3-PPO path/to/model_14999.pt
  uv run export_onnx <task> <checkpoint> --output ~/roomie_ppowalk_models/flat.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import playground  # noqa: F401  (registers the playground tasks)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("task", help="registered task id (see `uv run list_envs`)")
  parser.add_argument("checkpoint", type=Path, help="rsl_rl checkpoint (.pt)")
  parser.add_argument(
    "--output", type=Path, default=None,
    help="ONNX file to write (default: <checkpoint folder>/<task>_<checkpoint>.onnx)",
  )
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  checkpoint = args.checkpoint.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  output = (args.output or checkpoint.with_name(f"{args.task}_{checkpoint.stem}.onnx"))
  output = output.expanduser().resolve()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device="cpu"), clip_actions=agent_cfg.clip_actions
  )
  try:
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
    runner.export_policy_to_onnx(str(output.parent), output.name)
    attach_metadata_to_onnx(
      str(output), get_base_metadata(env.unwrapped, f"local:{checkpoint.name}")
    )
  finally:
    env.close()
  print(f"Exported {args.task} ({checkpoint.name}) to {output}")


if __name__ == "__main__":
  main()
