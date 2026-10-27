"""Tests for the Roomie V3 left/right mirror used for symmetry data augmentation."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.sensor.raycast_sensor import GridPatternCfg
from playground.asset_zoo.robots.roomie.roomie_constants import (
  HOME_KEYFRAME,
  get_roomie_robot_cfg,
)
from playground.tasks.velocity.config.roomie.env_cfgs import OBS_JOINT_NAMES
from playground.tasks.velocity.config.roomie.symmetry import (
  MirrorContext,
  build_group_mirror,
  grid_mirror_perm,
  joint_mirror,
  mirror_joint_name,
)

# The "joint_pos" action term targets the same 9 waist + hip + knee joints, in MJCF order.
ACTION_JOINT_NAMES = OBS_JOINT_NAMES

ACTOR_TERMS = [
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 9),
  ("joint_vel", 9),
  ("actions", 9),
  ("command", 3),
]
CRITIC_EXTRA_TERMS = [
  ("base_lin_vel", 3),
  ("foot_height", 2),
  ("foot_air_time", 2),
  ("foot_contact", 2),
  ("foot_contact_forces", 6),
]

_REFLECT_Y = np.array([1.0, -1.0, 1.0])


@pytest.fixture(scope="module")
def height_scan_perm() -> list[int]:
  offsets, _ = GridPatternCfg(size=(1.6, 1.0), resolution=0.1).generate_rays(
    None, "cpu"
  )
  return grid_mirror_perm(offsets[:, :2])


@pytest.fixture(scope="module")
def ctx(height_scan_perm) -> MirrorContext:
  return MirrorContext(
    obs_joint_names=OBS_JOINT_NAMES,
    action_joint_names=ACTION_JOINT_NAMES,
    height_scan_perm=height_scan_perm,
  )


def _apply(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
  return x[:, perm] * sign


def test_obs_joints_are_waist_hips_and_knees():
  assert len(OBS_JOINT_NAMES) == 9
  assert OBS_JOINT_NAMES[0] == "waist_yaw"
  assert not any(part in n for n in OBS_JOINT_NAMES for part in ("shoulder", "elbow", "ankle"))


def test_mirror_joint_name():
  assert mirror_joint_name("left_hip_roll") == "right_hip_roll"
  assert mirror_joint_name("right_ankle_near_motor") == "left_ankle_near_motor"
  assert mirror_joint_name("waist_yaw") == "waist_yaw"


def test_joint_mirror_is_involution():
  perm, sign = joint_mirror(OBS_JOINT_NAMES)
  for i in range(len(OBS_JOINT_NAMES)):
    assert perm[perm[i]] == i
    assert sign[perm[i]] == sign[i]


def test_joint_mirror_signs():
  _, sign = joint_mirror(OBS_JOINT_NAMES)
  sign_of = dict(zip(OBS_JOINT_NAMES, sign, strict=True))
  for name in OBS_JOINT_NAMES:
    flipped = name.endswith(("_roll", "_yaw"))
    assert sign_of[name] == (-1.0 if flipped else 1.0), name
  assert sign_of["waist_yaw"] == -1.0
  # The mirror still keeps the sign of both ankle cranks (axes along +y), should
  # they return to the joint list.
  _, crank_sign = joint_mirror(("left_ankle_near_motor", "right_ankle_near_motor"))
  assert crank_sign == [1.0, 1.0]


def test_joint_mirror_requires_partner():
  with pytest.raises(ValueError, match="no mirror"):
    joint_mirror(("left_hip_pitch", "right_hip_roll"))


@pytest.mark.parametrize("with_scan", [False, True])
def test_group_mirror_is_involution(ctx, with_scan):
  terms = ACTOR_TERMS + ([("height_scan", 187)] if with_scan else [])
  for group_terms in (terms, terms + CRITIC_EXTRA_TERMS):
    perm, sign = build_group_mirror(group_terms, ctx)
    x = torch.randn(7, perm.numel())
    torch.testing.assert_close(_apply(_apply(x, perm, sign), perm, sign), x)


def test_home_pose_is_mirror_invariant():
  perm, sign = joint_mirror(OBS_JOINT_NAMES)
  home = np.array([HOME_KEYFRAME.joint_pos.get(n, 0.0) for n in OBS_JOINT_NAMES])
  mirrored = home[perm] * np.array(sign)
  np.testing.assert_allclose(mirrored, home, atol=1e-9)


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def test_mirror_matches_forward_kinematics():
  """Mirrored joint angles must produce the mirror image of every body position.

  This validates the sign of each joint, including the waist, against the actual kinematic tree rather than against assumptions.
  """
  model = get_roomie_robot_cfg().build().spec.compile()
  perm, sign = joint_mirror(OBS_JOINT_NAMES)
  rng = np.random.default_rng(0)
  q = rng.uniform(-0.5, 0.5, size=len(OBS_JOINT_NAMES))
  q_mirrored = q[perm] * np.array(sign)

  def body_positions(angles: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    for name, angle in zip(OBS_JOINT_NAMES, angles, strict=True):
      data.qpos[_joint_qpos_address(model, name)] = angle
    mujoco.mj_kinematics(model, data)
    return {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): data.xpos[i].copy()
      for i in range(1, model.nbody)
    }

  original = body_positions(q)
  mirrored = body_positions(q_mirrored)

  checked = 0
  for name, pos in original.items():
    if name.startswith("left_"):
      partner = "right_" + name[5:]
    elif name.startswith("right_"):
      partner = "left_" + name[6:]
    else:
      partner = name  # Torso, pelvis: their own mirror image.
    np.testing.assert_allclose(
      pos, mirrored[partner] * _REFLECT_Y, atol=1e-6, err_msg=f"{name} <-> {partner}"
    )
    checked += 1
  assert checked > 20


def test_env_spaces_exclude_the_ankles():
  from mjlab.envs.mdp.actions import JointPositionActionCfg
  from playground.tasks.velocity.config.roomie.env_cfgs import roomie_flat_env_cfg

  cfg = roomie_flat_env_cfg()
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert not any("ankle" in p for p in action.actuator_names)
  assert set(action.scale) == {
    "waist_yaw", ".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw", ".*_knee_pitch"}
  held = cfg.events["hold_fixed_targets"].params["asset_cfg"].joint_names
  assert ".*_ankle_near_motor" in held and ".*_ankle_far_motor" in held
  # The cranks hold the linkage zero.
  assert not any("ankle" in n or "rod" in n for n in HOME_KEYFRAME.joint_pos)
