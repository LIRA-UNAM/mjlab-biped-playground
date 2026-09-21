"""Tests for the K1 left/right mirror used for symmetry data augmentation."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.sensor.raycast_sensor import GridPatternCfg
from playground.asset_zoo.robots.k1.k1_parallel_constants import (
  HOME_KEYFRAME,
  get_k1_parallel_robot_cfg,
)
from playground.tasks.velocity.config.k1.env_cfgs import OBS_JOINT_NAMES
from playground.tasks.velocity.config.k1.symmetry import (
  MirrorContext,
  build_group_mirror,
  grid_mirror_perm,
  joint_mirror,
  mirror_joint_name,
)

# Order of the "joint_pos" action term, as resolved in the live environment.
ACTION_JOINT_NAMES = (
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_A",
  "Left_Ankle_B",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_A",
  "Right_Ankle_B",
)

ACTOR_TERMS = [
  ("base_lin_vel", 3),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("joint_pos", 20),
  ("joint_vel", 20),
  ("actions", 20),
  ("command", 3),
]
CRITIC_EXTRA_TERMS = [
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


def test_mirror_joint_name():
  assert mirror_joint_name("Left_Hip_Roll") == "Right_Hip_Roll"
  assert mirror_joint_name("Right_Ankle_A") == "Left_Ankle_A"
  assert mirror_joint_name("Head_Yaw") == "Head_Yaw"


@pytest.mark.parametrize("names", [OBS_JOINT_NAMES, ACTION_JOINT_NAMES])
def test_joint_mirror_is_involution(names):
  perm, sign = joint_mirror(names)
  for i in range(len(names)):
    assert perm[perm[i]] == i
    assert sign[perm[i]] == sign[i]


def test_joint_mirror_signs():
  perm, sign = joint_mirror(ACTION_JOINT_NAMES)
  sign_of = dict(zip(ACTION_JOINT_NAMES, sign, strict=True))
  for name in ACTION_JOINT_NAMES:
    flipped = name.endswith(("_Roll", "_Yaw"))
    assert sign_of[name] == (-1.0 if flipped else 1.0), name
  # The parallel-ankle cranks both point along +y and never flip.
  assert sign_of["Left_Ankle_A"] == sign_of["Left_Ankle_B"] == 1.0


def test_joint_mirror_requires_partner():
  with pytest.raises(ValueError, match="no mirror"):
    joint_mirror(("Left_Hip_Pitch", "Right_Hip_Roll"))


def test_height_scan_perm_reverses_y(height_scan_perm):
  offsets, _ = GridPatternCfg(size=(1.6, 1.0), resolution=0.1).generate_rays(
    None, "cpu"
  )
  xy = offsets[:, :2]
  assert len(height_scan_perm) == 187
  assert sorted(height_scan_perm) == list(range(187))
  torch.testing.assert_close(xy[height_scan_perm], xy * torch.tensor([1.0, -1.0]))


@pytest.mark.parametrize("with_scan", [False, True])
def test_group_mirror_is_involution(ctx, with_scan):
  terms = ACTOR_TERMS + ([("height_scan", 187)] if with_scan else [])
  for group_terms in (terms, terms + CRITIC_EXTRA_TERMS):
    perm, sign = build_group_mirror(group_terms, ctx)
    x = torch.randn(7, perm.numel())
    torch.testing.assert_close(_apply(_apply(x, perm, sign), perm, sign), x)


def test_group_mirror_vectors(ctx):
  perm, sign = build_group_mirror(ACTOR_TERMS, ctx)
  x = torch.arange(1, perm.numel() + 1, dtype=torch.float32).unsqueeze(0)
  y = _apply(x, perm, sign)[0]
  # base_lin_vel, base_ang_vel, projected_gravity, then the command (last 3).
  assert y[0:3].tolist() == [1.0, -2.0, 3.0]
  assert y[3:6].tolist() == [-4.0, 5.0, -6.0]
  assert y[6:9].tolist() == [7.0, -8.0, 9.0]
  assert y[-3:].tolist() == [70.0, -71.0, -72.0]


def test_group_mirror_swaps_feet(ctx):
  terms = CRITIC_EXTRA_TERMS
  perm, sign = build_group_mirror(terms, ctx)
  x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10, 11, 12, 13, 14, 15]])
  y = _apply(x, perm, sign)[0].tolist()
  assert y[0:6] == [2.0, 1.0, 4.0, 3.0, 6.0, 5.0]  # height, air time, contact
  assert y[6:12] == [13.0, -14.0, 15.0, 10.0, -11.0, 12.0]  # forces


def test_group_mirror_rejects_unknown_term(ctx):
  with pytest.raises(ValueError, match="No mirror rule"):
    build_group_mirror([("mystery", 3)], ctx)


def test_group_mirror_rejects_wrong_dim(ctx):
  with pytest.raises(ValueError, match="expects"):
    build_group_mirror([("base_lin_vel", 4)], ctx)


def test_home_pose_is_mirror_invariant():
  perm, sign = joint_mirror(OBS_JOINT_NAMES)
  home = np.array([HOME_KEYFRAME.joint_pos.get(n, 0.0) for n in OBS_JOINT_NAMES])
  mirrored = home[perm] * np.array(sign)
  np.testing.assert_allclose(mirrored, home, atol=1e-9)


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
  return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]


def test_mirror_matches_forward_kinematics():
  """Mirrored joint angles must produce the mirror image of every body position.

  This validates the sign of each joint, including the parallel-ankle cranks,
  against the actual kinematic tree rather than against assumptions.
  """
  model = get_k1_parallel_robot_cfg().build().spec.compile()
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
    if name.startswith(("Left_", "left_")):
      partner = "Right_" + name[5:] if name.startswith("Left_") else "right_" + name[5:]
    elif name.startswith(("Right_", "right_")):
      partner = "Left_" + name[6:] if name.startswith("Right_") else "left_" + name[6:]
    else:
      partner = name  # Trunk, head: their own mirror image.
    np.testing.assert_allclose(
      pos, mirrored[partner] * _REFLECT_Y, atol=1e-6, err_msg=f"{name} <-> {partner}"
    )
    checked += 1
  assert checked > 20
