"""Velocity task event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from playground.asset_zoo.robots.roomie.actuators import SteadywinCascadeActuator

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def hold_default_joint_targets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Set the position targets of the selected joints to their default pose.

  Entity resets zero every joint position target, and only joints in an action
  term get new targets each step. Actuated joints left out of the action space
  would otherwise servo to 0 instead of holding the keyframe.
  """
  entity: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  joint_ids = asset_cfg.joint_ids
  if not isinstance(joint_ids, slice):
    # Tensor ids make the setter index the env x joint outer product.
    joint_ids = torch.as_tensor(joint_ids, device=env.device)
  target = entity.data.default_joint_pos[env_ids][:, joint_ids]
  entity.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


def randomize_steadywin_cascade(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  integral_gain_range: tuple[float, float],
  backlash_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Resample the SteadyWin cascade terms that dr.pd_gains does not cover.

  For every SteadywinCascadeActuator of the asset, one sample per joint and env:
  the position-loop integral gain is the configured value scaled by
  integral_gain_range (scaled from the config, so draws do not compound), and the
  gearbox backlash (rad) is drawn from backlash_range.
  """
  entity: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  for actuator in entity.actuators:
    if not isinstance(actuator, SteadywinCascadeActuator):
      continue
    assert actuator.integral_gain is not None and actuator.backlash is not None
    shape = (len(env_ids), actuator.integral_gain.shape[1])
    lo, hi = integral_gain_range
    scale = lo + (hi - lo) * torch.rand(shape, device=env.device)
    actuator.integral_gain[env_ids] = actuator.cfg.integral_gain * scale
    lo, hi = backlash_range
    actuator.backlash[env_ids] = lo + (hi - lo) * torch.rand(shape, device=env.device)
