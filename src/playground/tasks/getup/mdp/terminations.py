"""Termination conditions for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def energy_termination(
  env: ManagerBasedRlEnv,
  threshold: float = float("inf"),
  settle_steps: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  match_by_name: bool = False,
) -> torch.Tensor:
  """Terminate when mechanical power exceeds threshold.

  Power = sum(|actuator_force * joint_vel|). Skips the first settle_steps steps so
  drop/settle dynamics don't trigger early termination.

  By default actuators and joints are paired positionally, which assumes one
  actuator per joint in the same order. With ``match_by_name`` each actuator is
  paired with the joint it drives (actuators are named after their joint), for
  models with unactuated joints such as a closed kinematic loop.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if match_by_name:
    joint_ids = [asset.joint_names.index(n) for n in asset.actuator_names]
    power = torch.sum(
      torch.abs(asset.data.actuator_force * asset.data.joint_vel[:, joint_ids]),
      dim=-1,
    )
  else:
    power = torch.sum(
      torch.abs(
        asset.data.actuator_force[:, asset_cfg.actuator_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
      ),
      dim=-1,
    )
  past_settle = env.episode_length_buf > settle_steps
  return past_settle & (power > threshold)
