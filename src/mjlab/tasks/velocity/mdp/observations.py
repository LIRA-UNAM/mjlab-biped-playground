from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


class gait_clock:
  """Gait clock observation - sin/cos of gait phase like Booster.

  Returns [cos(2*pi*phase), sin(2*pi*phase)] as a 2D observation.
  The phase advances based on gait_frequency when robot is commanded to move.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.gait_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    self.gait_frequency = cfg.params.get("gait_frequency", 1.5)
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    gait_frequency: float = 1.5,
  ) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert command is not None

    # Only advance phase when moving
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    # Update gait phase
    self.gait_phase = torch.fmod(
      self.gait_phase + self.step_dt * gait_frequency * active, 1.0
    )

    # Reset phase on episode reset (only if termination_manager exists)
    if hasattr(env, 'termination_manager') and env.termination_manager is not None:
      reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).flatten()
      if len(reset_ids) > 0:
        self.gait_phase[reset_ids] = torch.rand(len(reset_ids), device=env.device)

    # Return cos/sin of phase
    phase_2pi = 2.0 * 3.14159265359 * self.gait_phase
    return torch.stack([torch.cos(phase_2pi), torch.sin(phase_2pi)], dim=1)
