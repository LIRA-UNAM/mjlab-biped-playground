"""Velocity task observation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

# Actor terms with no onboard sensor on the real robots: the true base linear
# velocity (a MuJoCo velocimeter; a physical IMU only measures angular velocity,
# acceleration and orientation) and the ray-cast terrain height map. They stay in
# the critic as privileged information (asymmetric actor-critic).
NON_DEPLOYABLE_ACTOR_TERMS = ("base_lin_vel", "height_scan")


def remove_non_deployable_actor_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  """Restrict the actor to IMU (gyro, gravity) and joint-encoder observations."""
  for name in NON_DEPLOYABLE_ACTOR_TERMS:
    cfg.observations["actor"].terms.pop(name, None)
