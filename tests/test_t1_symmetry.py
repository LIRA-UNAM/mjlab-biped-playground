"""Tests for the Booster T1 (23-DOF) left/right mirror used for symmetry data augmentation."""

import re

import mujoco
import numpy as np
import pytest
import torch
from mjlab.entity import Entity
from playground.asset_zoo.robots.booster_t1.t1_constants import (
  HOME_KEYFRAME,
  get_t1_robot_cfg,
)
from playground.tasks.velocity.config.t1.symmetry import (
  MirrorContext,
  build_group_mirror,
  joint_mirror,
  mirror_joint_name,
)

_REFLECT_Y = np.array([1.0, -1.0, 1.0])


def _mirror_body_name(name: str) -> str:
  """T1 bodies use three naming styles: AL*/AR*, *_Left/*_Right, left_*/right_*."""
  for a, b in (("AL", "AR"), ("left_", "right_")):
    if name.startswith(a):
      return b + name[len(a) :]
    if name.startswith(b):
      return a + name[len(b) :]
  for a, b in (("_Left", "_Right"),):
    if name.endswith(a):
      return name[: -len(a)] + b
    if name.endswith(b):
      return name[: -len(b)] + a
  return name


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(get_t1_robot_cfg()).spec.compile()


@pytest.fixture(scope="module")
def joint_names(model) -> tuple[str, ...]:
  # Hinge joints in MJCF order (skip the floating base).
  return tuple(
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )


def test_joint_count(joint_names):
  assert len(joint_names) == 23


def test_mirror_joint_name():
  assert mirror_joint_name("Left_Hip_Roll") == "Right_Hip_Roll"
  assert mirror_joint_name("Right_Elbow_Yaw") == "Left_Elbow_Yaw"
  assert mirror_joint_name("Waist") == "Waist"
  assert mirror_joint_name("AAHead_yaw") == "AAHead_yaw"


def test_joint_mirror_is_involution(joint_names):
  perm, sign = joint_mirror(joint_names)
  for i in range(len(joint_names)):
    assert perm[perm[i]] == i
    assert sign[perm[i]] == sign[i]


def test_joint_mirror_signs(joint_names):
  _, sign = joint_mirror(joint_names)
  sign_of = dict(zip(joint_names, sign, strict=True))
  for name in joint_names:
    flipped = name == "Waist" or name.endswith(("_Roll", "_Yaw", "_yaw"))
    assert sign_of[name] == (-1.0 if flipped else 1.0), name
  assert sign_of["Waist"] == -1.0
  assert sign_of["AAHead_yaw"] == -1.0
  assert sign_of["Head_pitch"] == 1.0
  assert sign_of["Left_Elbow_Pitch"] == 1.0
  assert sign_of["Left_Elbow_Yaw"] == -1.0


def test_group_mirror_is_involution(joint_names):
  # Actions exclude the neck, as in the velocity task.
  action_names = tuple(n for n in joint_names if n not in ("AAHead_yaw", "Head_pitch"))
  ctx = MirrorContext(obs_joint_names=joint_names, action_joint_names=action_names)
  terms = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 23),
    ("joint_vel", 23),
    ("actions", 21),
    ("command", 3),
    ("foot_height", 2),
    ("foot_air_time", 2),
    ("foot_contact", 2),
    ("foot_contact_forces", 6),
  ]
  perm, sign = build_group_mirror(terms, ctx)
  x = torch.randn(7, perm.numel())
  torch.testing.assert_close((x[:, perm] * sign)[:, perm] * sign, x)


def test_home_pose_is_mirror_invariant(joint_names):
  perm, sign = joint_mirror(joint_names)

  def default(name: str) -> float:
    for pattern, value in HOME_KEYFRAME.joint_pos.items():
      if re.fullmatch(pattern, name):
        return value
    return 0.0

  pose = np.array([default(n) for n in joint_names])
  np.testing.assert_allclose(pose[perm] * np.array(sign), pose, atol=1e-9)


def test_mirror_matches_forward_kinematics(model, joint_names):
  """Mirrored joint angles must produce the mirror image of every body position.

  Validates each joint's sign against the kinematic tree.
  """
  perm, sign = joint_mirror(joint_names)
  rng = np.random.default_rng(0)
  q = rng.uniform(-0.5, 0.5, size=len(joint_names))
  q_mirrored = q[perm] * np.array(sign)

  def body_positions(angles: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    for name, angle in zip(joint_names, angles, strict=True):
      data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_kinematics(model, data)
    return {model.body(i).name: data.xpos[i].copy() for i in range(1, model.nbody)}

  original = body_positions(q)
  mirrored = body_positions(q_mirrored)

  for name, pos in original.items():
    partner = _mirror_body_name(name)
    np.testing.assert_allclose(
      pos, mirrored[partner] * _REFLECT_Y, atol=1e-4, err_msg=f"{name} <-> {partner}"
    )
  assert len(original) > 20
