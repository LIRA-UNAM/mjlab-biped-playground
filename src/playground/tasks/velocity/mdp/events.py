"""Velocity task event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
