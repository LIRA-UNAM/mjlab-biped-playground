"""Roomie V3 constants.

Roomie V3 is a headless humanoid with a waist yaw joint and K1-style parallel
ankles: each ankle is driven by two cranks (``*_ankle_near_motor`` and
``*_ankle_far_motor``) whose rods are attached to the foot through a universal
joint (``*_rod_pitch`` + ``*_rod_roll``). The ankle pitch/roll joints are
passive and closed by four ``connect`` equalities in the MJCF.

All hinges use one axis convention: roll +X, pitch +Y, yaw +Z. Joint values in
this file are MJCF (XML) angles, not the Roomie ROS/hardware public signs.

Actuator constants come from the SteadyWin GIM8108-36 / GIM6010-36 datasheets;
the GIM6010-36 rotor inertia (and therefore its gains) is an estimate.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from playground.asset_zoo.robots.k1.actuators import (
  ActuatorConfig,
  MotorPositionActuatorCfg,
)

if TYPE_CHECKING:
  import torch

##
# MJCF and assets.
##

ROOMIE_XML: Path = Path(__file__).parent / "xmls" / "roomie.xml"
assert ROOMIE_XML.exists()

# Ground-contact proxies of every link, the foot soles and the forearm profiles.
_CONTACT_GEOMS = r"^(.*_ground_proxy|.*_foot_contact|.*_forearm_profile_wall_.*)$"


def _load_spec(self_collision: bool) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROOMIE_XML))

  for geom in list(spec.geoms):
    if not geom.name.startswith("self_collision_"):
      continue
    if self_collision:
      # Convex CAD pieces keep their XML masks: they only collide with each
      # other, never with the terrain.
      geom.group = 3
    else:
      # 2000+ convex pieces are too expensive for batched RL; drop them.
      spec.delete(geom)
  if not self_collision:
    for mesh in list(spec.meshes):
      if mesh.name.startswith("self_collision_"):
        spec.delete(mesh)

  # Keep all robot collision geoms in the collision visualization group.
  for geom in spec.geoms:
    if geom.name.endswith(("_ground_proxy", "_foot_contact")):
      geom.group = 3

  return spec


def get_spec() -> mujoco.MjSpec:
  """Roomie spec without self-collision pieces (terrain contact only)."""
  return _load_spec(self_collision=False)


def get_spec_with_self_collision() -> mujoco.MjSpec:
  """Roomie spec keeping the convex CAD self-collision pieces."""
  return _load_spec(self_collision=True)


##
# Joint layout.
##

# Passive joints of the closed loop: the two ankle DOFs the linkage drives, and
# each rod's universal joint.
_PASSIVE_ANKLE_SUFFIXES = (
  "ankle_pitch",
  "ankle_roll",
  "near_rod_pitch",
  "near_rod_roll",
  "far_rod_pitch",
  "far_rod_roll",
)


def _leg_joints(side: str) -> tuple[str, ...]:
  return (
    f"{side}_hip_pitch",
    f"{side}_hip_roll",
    f"{side}_hip_yaw",
    f"{side}_knee_pitch",
    f"{side}_ankle_pitch",
    f"{side}_ankle_roll",
    f"{side}_ankle_near_motor",
    f"{side}_near_rod_pitch",
    f"{side}_near_rod_roll",
    f"{side}_ankle_far_motor",
    f"{side}_far_rod_pitch",
    f"{side}_far_rod_roll",
  )


def _arm_joints(side: str) -> tuple[str, ...]:
  return (
    f"{side}_shoulder_pitch",
    f"{side}_shoulder_roll",
    f"{side}_shoulder_yaw",
    f"{side}_elbow_pitch",
  )


# All 33 joints in MJCF definition order. This is the column order of
# `Entity.joint_names` and `data.joint_pos`.
ROOMIE_JOINT_ORDER: tuple[str, ...] = (
  "waist_yaw",
  *_leg_joints("right"),
  *_leg_joints("left"),
  *_arm_joints("left"),
  *_arm_joints("right"),
)

# Passive joints of the closed loop, excluded from observations and rewards.
ROOMIE_PASSIVE_JOINTS: tuple[str, ...] = tuple(
  f"{side}_{suffix}" for side in ("left", "right") for suffix in _PASSIVE_ANKLE_SUFFIXES
)

# The 21 actuated joints, in MJCF definition order.
ROOMIE_ACTUATED_JOINTS: tuple[str, ...] = tuple(
  name for name in ROOMIE_JOINT_ORDER if name not in ROOMIE_PASSIVE_JOINTS
)

# Actuated joints outside the closed loop. Reset randomization is limited to
# these so the linkage starts exactly closed.
ROOMIE_FREE_RESET_JOINTS: tuple[str, ...] = tuple(
  name for name in ROOMIE_ACTUATED_JOINTS if "ankle" not in name
)


##
# Keyframe config.
##


def _linkage_pose(
  near: float, near_rod_pitch: float, far: float, far_rod_pitch: float
) -> dict[str, float]:
  """Both legs' linkage angles for a symmetric ankle pose (roll = 0)."""
  return {
    f"{side}_{joint}": value
    for side in ("left", "right")
    for joint, value in (
      ("ankle_near_motor", near),
      ("near_rod_pitch", near_rod_pitch),
      ("ankle_far_motor", far),
      ("far_rod_pitch", far_rod_pitch),
    )
  }


# Solved numerically so the keyframe closes both loops to machine precision at
# ankle pitch -0.35 (rod roll stays 0 for a symmetric pose).
_HOME_LINKAGE = _linkage_pose(-0.327708, 0.394469, -0.328636, 0.35804)

# Slightly crouched stance. The upper arms hang down and both elbows bend 90
# degrees forward (negative about +Y), clear of the torso and hip housings.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.66),
  joint_pos={
    "left_elbow_pitch": -math.pi / 2,
    "right_elbow_pitch": -math.pi / 2,
    "left_hip_pitch": -0.35,
    "left_knee_pitch": 0.7,
    "left_ankle_pitch": -0.35,
    "right_hip_pitch": -0.35,
    "right_knee_pitch": 0.7,
    "right_ankle_pitch": -0.35,
    **_HOME_LINKAGE,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# The proxy boxes were sized for ground contact and overlap each other (e.g.
# the two hip-pitch housings, rods inside the shank hulls), so they only touch
# the terrain. Equal priority with the terrain, so contact compliance mixes with
# the (randomized) terrain solref/solimp.
TERRAIN_COLLISION = CollisionCfg(
  geom_names_expr=(_CONTACT_GEOMS,),
  contype=0,
  conaffinity=1,
  solref=(0.01, 1),
  condim={r"^(left|right)_foot_contact$": 3, ".*": 1},
  priority=0,
  solmix=0.001,
  friction={r"^(left|right)_foot_contact$": (1.0,), ".*": (0.45,)},
)

# This disables all collisions except the feet.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_contact$",),
  contype=0,
  conaffinity=1,
  solref=(0.01, 1),
  condim=3,
  priority=1,
  friction=(0.6,),
)


# All Roomie articulations get the same scale rule for the joint position
# action: 0.25 * effort_limit / stiffness per joint group.
_ACTION_SCALE_FACTOR = 0.25

# Shared delay parameters for all joint groups.
ACTUATOR_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)


def _build_action_scale(
  articulation: EntityArticulationInfoCfg, factor: float = _ACTION_SCALE_FACTOR
) -> dict[str, float]:
  """Build joint-name -> scale dict from an articulation tuple.

  Each actuator group must expose ``effort_limit`` and ``stiffness``. The same
  scale (factor * effort_limit / stiffness) is applied to every joint matched
  by that group's regex.
  """
  out: dict[str, float] = {}
  for a in articulation.actuators:
    e = a.effort_limit
    s = a.stiffness
    assert e is not None
    for n in a.target_names_expr:
      out[n] = factor * e / s
  return out


##
# Roomie motor constants (one per physical motor kind).
##
# Roomie V3 uses two SteadyWin planetary actuators (36:1, 48 V, Y winding):
# GIM8108-36 with the GDZ810 driver and GIM6010-36 with the GDZ468H driver.
# Torque/speed values are the installed-driver columns of the manufacturer
# datasheets ("Parameters of GIM8108-36.pdf" / "Parameters of GIM6010-36.pdf"):
#   https://www.steadywin.cn/en/pd.jsp?fromColId=0&id=16
#   https://www.steadywin.cn/en/pd.jsp?fromColId=0&id=13
# The web page for GIM6010-36 + GDZ468H lists slightly lower figures
# (17.4 Nm rated, 51.5 Nm stall, 65/100 rpm); the datasheet is used here.
#
# effort_limit is the peak torque, matching the other robots (Asimov, K1 and G1
# all use datasheet peak values). SteadyWin datasheets only list nominal and
# stall torque, and stall is the motor's peak; the rated (continuous) torques
# are 25 Nm and 17.6 Nm. Speed limits are metadata only and the physical
# controller currently limits position moves to 50 rpm (5.24 rad/s).

_RPM_TO_RAD_S = 2.0 * math.pi / 60.0
_GEAR_RATIO = 36.0


def _knee_point_velocity(
  peak_torque: float, no_load_speed: float, rated_torque: float, rated_speed: float
) -> float:
  """Start of linear derating so the curve passes through the rated point.

  The peak torque is held up to the knee speed and falls linearly to zero at
  the no-load speed; the knee is placed so the datasheet rated torque is still
  available at the rated speed.
  """
  return no_load_speed - peak_torque * (no_load_speed - rated_speed) / rated_torque


# Rotor inertia from the manufacturer rotor archive (Ixx = 69.748 kg mm^2).
ROTOR_INERTIA_GIM8108 = 6.9748e-5
# Not published for GIM6010-36. Estimated by scaling the GIM8108 outer rotor
# with outer diameter cubed (76 mm vs 96 mm, thin ring of equal length).
# TODO: replace with the manufacturer value; treat as +-50%.
ROTOR_INERTIA_GIM6010 = ROTOR_INERTIA_GIM8108 * (76.0 / 96.0) ** 3

ARMATURE_GIM8108 = ROTOR_INERTIA_GIM8108 * _GEAR_RATIO**2
ARMATURE_GIM6010 = ROTOR_INERTIA_GIM6010 * _GEAR_RATIO**2

# PD gains from reflected inertia, as in the G1: 10 Hz natural frequency,
# damping ratio 2.
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

# GIM8108-36 peak torque, capped below the 130.2 Nm datasheet stall torque.
PEAK_TORQUE_GIM8108 = 100.0

# GIM8108-36 + GDZ810: waist yaw, hip pitch, hip roll, knee.
ACTUATOR_GIM8108 = ActuatorConfig(
  armature=ARMATURE_GIM8108,
  effort_limit=PEAK_TORQUE_GIM8108,
  velocity_limit=67 * _RPM_TO_RAD_S,
  knee_point_velocity=_knee_point_velocity(
    PEAK_TORQUE_GIM8108, 67 * _RPM_TO_RAD_S, 25.0, 61 * _RPM_TO_RAD_S
  ),
  stiffness=ARMATURE_GIM8108 * NATURAL_FREQ**2,
  damping=2.0 * DAMPING_RATIO * ARMATURE_GIM8108 * NATURAL_FREQ,
)

# GIM6010-36 + GDZ468H: hip yaw, ankle cranks, shoulders and elbows.
ACTUATOR_GIM6010 = ActuatorConfig(
  armature=ARMATURE_GIM6010,
  effort_limit=52.48,
  velocity_limit=105 * _RPM_TO_RAD_S,
  knee_point_velocity=_knee_point_velocity(
    52.48, 105 * _RPM_TO_RAD_S, 17.6, 80 * _RPM_TO_RAD_S
  ),
  stiffness=ARMATURE_GIM6010 * NATURAL_FREQ**2,
  damping=2.0 * DAMPING_RATIO * ARMATURE_GIM6010 * NATURAL_FREQ,
)


##
# Roomie articulation.
##


def _actuator_cfg(
  target_names_expr: tuple[str, ...], a: ActuatorConfig
) -> MotorPositionActuatorCfg:
  return MotorPositionActuatorCfg(
    motor=a,
    target_names_expr=target_names_expr,
    effort_limit=a.effort_limit,
    armature=a.armature,
    stiffness=a.stiffness,
    damping=a.damping,
    **ACTUATOR_DELAY,
  )


ROOMIE_ACTUATOR_GIM8108 = _actuator_cfg(
  ("waist_yaw", ".*_hip_pitch", ".*_hip_roll", ".*_knee_pitch"), ACTUATOR_GIM8108
)
# Each ankle crank is driven by its own GIM6010-36, so the crank joints get one
# motor's armature (no doubling as for a serial ankle joint).
ROOMIE_ACTUATOR_GIM6010 = _actuator_cfg(
  (
    ".*_hip_yaw",
    ".*_ankle_near_motor",
    ".*_ankle_far_motor",
    ".*_shoulder_.*",
    ".*_elbow_.*",
  ),
  ACTUATOR_GIM6010,
)


ROOMIE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ROOMIE_ACTUATOR_GIM8108,
    ROOMIE_ACTUATOR_GIM6010,
  ),
  soft_joint_pos_limit_factor=0.9,
)


ROOMIE_ACTION_SCALE = _build_action_scale(ROOMIE_ARTICULATION)


def get_roomie_robot_cfg(
  default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
  default_collisions: tuple[CollisionCfg, ...] = (TERRAIN_COLLISION,),
  self_collision: bool = False,
) -> EntityCfg:
  """Get a fresh Roomie robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    default_keyframe: Initial state keyframe to seed each environment.
    default_collisions: Collision config tuple.
    self_collision: Keep the convex CAD self-collision pieces (expensive).
  """
  return EntityCfg(
    init_state=default_keyframe,
    collisions=default_collisions,
    spec_fn=get_spec_with_self_collision if self_collision else get_spec,
    articulation=ROOMIE_ARTICULATION,
  )


def get_roomie_default_joint_pos(
  keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
  """Return the 21 actuated default joint positions as a 1D tensor."""
  import torch

  assert keyframe.joint_pos is not None
  return torch.tensor(
    [keyframe.joint_pos.get(name, 0.0) for name in ROOMIE_ACTUATED_JOINTS],
    dtype=torch.float32,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  robot = get_roomie_robot_cfg().build()

  viewer.launch(robot.spec.compile())
