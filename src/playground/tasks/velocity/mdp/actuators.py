"""Actuator randomization selected by target (joint) names."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from mjlab.entity import Entity
from mjlab.envs.mdp.dr import pd_gains
from mjlab.envs.mdp.dr._types import Operation
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["pd_gains_by_target"]


def _matching_actuator_groups(
  asset: Entity, target_names_expr: Sequence[str]
) -> list[int]:
  """Indices into ``asset.actuators`` whose targets all match ``target_names_expr``.

  ``asset.actuators`` holds one entry per actuator *config* (a group of joints),
  while ``SceneEntityCfg.actuator_names`` resolves against the per-joint spec
  actuators, so ``dr.pd_gains`` can only select a subset correctly via group
  indices. Groups must match fully: a partial match raises instead of silently
  randomizing joints outside ``target_names_expr``.
  """
  patterns = [re.compile(p) for p in target_names_expr]
  ids: list[int] = []
  for i, act in enumerate(asset.actuators):
    hits = [any(p.fullmatch(n) for p in patterns) for n in act.target_names]
    if all(hits):
      ids.append(i)
    elif any(hits):
      raise ValueError(
        f"Actuator group {act.target_names} only partially matches "
        f"{tuple(target_names_expr)}; split the group or adjust the expression."
      )
  if not ids:
    raise ValueError(f"No actuator group matches {tuple(target_names_expr)}.")
  return ids


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains_by_target(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  target_names_expr: Sequence[str],
  asset_name: str = "robot",
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Operation | str = "scale",
) -> None:
  """``dr.pd_gains`` restricted to the actuator groups driving ``target_names_expr``.

  Gains are sampled independently per joint, as in ``dr.pd_gains``.
  """
  asset: Entity = env.scene[asset_name]
  asset_cfg = SceneEntityCfg(asset_name)
  asset_cfg.actuator_ids = _matching_actuator_groups(asset, target_names_expr)
  pd_gains(
    env,
    env_ids,
    kp_range=kp_range,
    kd_range=kd_range,
    asset_cfg=asset_cfg,
    distribution=distribution,
    operation=operation,
  )
