"""Off-policy runner wiring FlashSAC into mjlab's checkpoint/export conventions.

Adapted from ``rsl_rl_flashsac.runners.off_policy_runner.OffPolicyRunner``
(itself a subclass of ``rsl_rl.runners.OnPolicyRunner`` that overrides
construction and the learning loop for off-policy training), replacing its
save/load and adding ONNX export to match ``mjlab.rl.runner.MjlabOnPolicyRunner``
and ``mjlab.tasks.velocity.rl.runner.VelocityOnPolicyRunner`` instead of the
Isaac-Lab-oriented behavior of the original.
"""

import os
import time
from pathlib import Path

import torch
import wandb
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger

from playground.rl.flashsac.algorithms import FlashSAC
from playground.rl.flashsac.utils import resolve_callable


class MjlabOffPolicyRunner(OnPolicyRunner):
  """Off-policy runner for training with FlashSAC.

  Inherits checkpointing scaffolding, ONNX export helpers, and multi-GPU setup
  from the upstream ``rsl_rl.runners.OnPolicyRunner``; overrides construction
  (to resolve FlashSAC classes) and the learning loop (off-policy update
  cadence and windowed logging), and persists the environment's
  ``common_step_counter`` across checkpoints like ``MjlabOnPolicyRunner`` does
  for PPO.
  """

  env: RslRlVecEnvWrapper
  alg: FlashSAC

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.device = device

    self._configure_multi_gpu()

    obs = self.env.get_observations()

    self.cfg["algorithm"] = resolve_symmetry_config(self.cfg["algorithm"], self.env)

    alg_class: type[FlashSAC] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore[assignment]
    self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

    self.logger = Logger(
      log_dir=log_dir,
      cfg=self.cfg,
      env_cfg=self.env.cfg,
      num_envs=self.env.num_envs,
      is_distributed=self.is_distributed,
      gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank,
      device=self.device,
    )

    self.current_learning_iteration = 0
    self.start_training = self.cfg.get("start_training", 0)
    self.log_interval = self.cfg.get("log_interval", 20)
    self._wall_time_offset = 0.0
    self._learn_start_time: float | None = None
    self.loaded_wall_time: float | None = None

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run the off-policy learning loop for the specified number of iterations."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    self._learn_start_time = time.time()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations

    save_interval = self.cfg["save_interval"]
    if save_interval <= 0:
      save_interval = max(1, num_learning_iterations // 10)

    window_collect_time = 0.0
    window_learn_time = 0.0
    window_iters = 0

    for it in range(start_it, total_it):
      start = time.time()
      with torch.no_grad():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = (
            self.alg.act(obs)
            if self.alg.can_start_training()
            else self.alg.act_random(obs)
          )
          next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(next_obs, rewards, dones)
          next_obs, rewards, dones = (
            next_obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(next_obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards=None)
          obs = next_obs

        stop = time.time()
        collect_time = stop - start
        start = stop

      loss_dict = self.alg.update() if it >= self.start_training else {}

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      window_collect_time += collect_time
      window_learn_time += learn_time
      window_iters += 1

      if (it % self.log_interval == 0) or (it == total_it - 1):
        avg_collect = window_collect_time / window_iters
        avg_learn = window_learn_time / window_iters
        self.logger.log(
          it=it,
          start_it=start_it,
          total_it=total_it,
          collect_time=avg_collect,
          learn_time=avg_learn,
          loss_dict=loss_dict,
          learning_rate=self.alg.actor_learning_rate,
          action_std=self.alg.get_policy().output_std,
          rnd_weight=None,
        )
        if self.logger.writer is not None:
          collection_size = (
            self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
          )
          self.logger.tot_timesteps += collection_size * (window_iters - 1)
          self.logger.tot_time += (window_collect_time + window_learn_time) - (
            avg_collect + avg_learn
          )
          self.logger.writer.add_scalar("Policy/temperature", self.alg.alpha, it)
          self.logger.writer.add_scalar(
            "Train/env_steps", self.logger.tot_timesteps, it
          )
          self.logger.writer.add_scalar(
            "Train/wall_time", self._elapsed_wall_time(), it
          )
        window_collect_time = 0.0
        window_learn_time = 0.0
        window_iters = 0

      if self.logger.writer is not None and it % save_interval == 0 and it != 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
      )
      self.logger.stop_logging_writer()

  def export_history_encoder_to_jit(
    self, path: str, filename: str = "history_encoder.pt"
  ) -> None:
    """Export the policy's history encoder (Estimator variant) as a TorchScript file."""
    policy = self.alg.get_policy()
    if not hasattr(policy, "history_encoder_as_jit"):
      raise AttributeError(
        f"{type(policy).__name__} has no history encoder; nothing to export."
      )
    os.makedirs(path, exist_ok=True)
    jit_module = policy.history_encoder_as_jit().to("cpu")  # type: ignore[operator]
    scripted = torch.jit.script(jit_module)
    scripted.save(os.path.join(path, filename))

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Mirrors ``MjlabOnPolicyRunner.export_policy_to_onnx``: sets dynamo=False to
    avoid warnings about dynamic_axes being deprecated with the new
    TorchDynamo export path (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def _elapsed_wall_time(self) -> float:
    """Cumulative training wall-clock seconds (across resumes)."""
    running = (
      time.time() - self._learn_start_time
      if self._learn_start_time is not None
      else 0.0
    )
    return self._wall_time_offset + running

  def save(self, path: str, infos: dict | None = None) -> None:
    """Save checkpoint.

    Extends the base FlashSAC checkpoint with the environment's
    common_step_counter (to preserve curricula state) and elapsed wall time,
    and respects the ``upload_model`` config flag like
    ``MjlabOnPolicyRunner.save``.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    saved_dict["wall_time"] = self._elapsed_wall_time()
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint, restoring common_step_counter and wall-time offset."""
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    self.loaded_wall_time = loaded_dict.get("wall_time")
    if self.loaded_wall_time is not None:
      self._wall_time_offset = self.loaded_wall_time

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


class VelocityOffPolicyRunner(MjlabOffPolicyRunner):
  """FlashSAC runner for velocity tasks: uploads the exported ONNX policy to W&B on save."""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
