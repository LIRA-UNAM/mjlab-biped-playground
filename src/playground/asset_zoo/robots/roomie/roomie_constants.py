"""Roomie V3 constants.

Roomie V3 is a headless humanoid with a waist yaw joint and K1-style parallel
ankles: each ankle is driven by two cranks (``*_ankle_near_motor`` and
``*_ankle_far_motor``) whose rods are attached to the foot through a universal
joint (``*_rod_pitch`` + ``*_rod_roll``). The ankle pitch/roll joints are
passive and closed by four ``connect`` equalities in the MJCF. The cranks are
actuated but not policy joints: they are held at 0 (see HOME_KEYFRAME).

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
from playground.asset_zoo.robots.roomie.actuators import SteadywinCascadeActuatorCfg

if TYPE_CHECKING:
  import torch

##
# MJCF and assets.
##

ROOMIE_XML: Path = Path(__file__).parent / "xmls" / "roomie.xml"
assert ROOMIE_XML.exists()

# Ground-contact proxies of every link, the foot soles and the forearm profiles.
_CONTACT_GEOMS = r"^(.*_ground_proxy|.*_foot_contact|.*_forearm_profile_wall_.*)$"

# Leg proxies (hip pitch link down to the foot, including cranks and rods).
_LEFT_LEG_GEOMS = r"^left_(hip|knee|ankle|near|far|foot)_"
_RIGHT_LEG_GEOMS = r"^right_(hip|knee|ankle|near|far|foot)_"

# The two hip-pitch housings overlap in every pose by construction; exclude them
# so leg self-collision only reports real contacts (crossed legs, knees, feet).
_PROXY_SELF_EXCLUDES: tuple[tuple[str, str], ...] = (
  ("left_hip_pitch_link", "right_hip_pitch_link"),
)


def _load_spec(cad_self_collision: bool) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROOMIE_XML))

  for geom in list(spec.geoms):
    if not geom.name.startswith("self_collision_"):
      continue
    if cad_self_collision:
      # Convex CAD pieces keep their XML masks: they only collide with each
      # other, never with the terrain.
      geom.group = 3
    else:
      # 2000+ convex pieces are too expensive for batched RL; drop them.
      spec.delete(geom)
  if not cad_self_collision:
    for mesh in list(spec.meshes):
      if mesh.name.startswith("self_collision_"):
        spec.delete(mesh)
    for body1, body2 in _PROXY_SELF_EXCLUDES:
      spec.add_exclude(bodyname1=body1, bodyname2=body2)

  # Keep all robot collision geoms in the collision visualization group.
  for geom in spec.geoms:
    if geom.name.endswith(("_ground_proxy", "_foot_contact")):
      geom.group = 3

  return spec


def get_spec() -> mujoco.MjSpec:
  """Roomie spec using the link proxies for terrain and leg-vs-leg contact."""
  return _load_spec(cad_self_collision=False)


def get_spec_with_cad_self_collision() -> mujoco.MjSpec:
  """Roomie spec keeping the convex CAD self-collision pieces (expensive)."""
  return _load_spec(cad_self_collision=True)


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


# The ankles are not policy joints: both cranks are actuated but held at 0, the
# neutral pose of the linkage (foot pitch and roll 0 relative to the shin, rods
# and passive ankle joints all 0: the loops close exactly there). The ankle
# mapping between the MJCF and the real linkage proved unreliable, while the
# motor zero is the same neutral pose on the robot.
#
# Nearly straight stance, so the flat feet are level with an upright torso
# (hip + knee + ankle = 0) and stay under the centre of mass: with the ankles
# fixed, a deeper bend moves the soles forward (hip -0.35 / knee 0.35 leaves the
# CoM 28 mm from the heel and the robot falls backward). The knee bend of 0.1
# stays inside its soft joint limit. The upper arms hang down and both elbows
# bend 90 degrees forward (negative about +Y), clear of the torso and hip
# housings. pos z puts the soles about 5 mm above the ground.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.70),
  joint_pos={
    "left_elbow_pitch": -math.pi / 2,
    "right_elbow_pitch": -math.pi / 2,
    "left_hip_pitch": -0.1,
    "left_knee_pitch": 0.1,
    "right_hip_pitch": -0.1,
    "right_knee_pitch": 0.1,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Pair with get_spec(). Every proxy touches the terrain (contype/conaffinity bit
# 1); the only self-contact is left leg vs right leg (bits 2 and 4). Contacts
# within one leg and with the torso or arms are disabled. Equal priority with
# the terrain, so contact compliance mixes with the (randomized) terrain
# solref/solimp.
LEG_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(_CONTACT_GEOMS,),
  contype={_LEFT_LEG_GEOMS: 2, _RIGHT_LEG_GEOMS: 4, ".*": 0},
  conaffinity={_LEFT_LEG_GEOMS: 1 | 4, _RIGHT_LEG_GEOMS: 1 | 2, ".*": 1},
  solref=(0.01, 1),
  condim={r"^(left|right)_foot_contact$": 3, ".*": 1},
  priority=0,
  solmix=0.001,
  friction={r"^(left|right)_foot_contact$": (1.0,), ".*": (0.45,)},
)

# Proxies touch the terrain only. Used with the CAD self-collision pieces, which
# then handle self-contact on their own.
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


# Nominal per-motor PD (datasheet armature, 10 Hz / damping ratio 2). It is no
# longer what the simulation applies; it only defines the joint position action
# scale (0.25 * effort_limit / stiffness), which the deployed policy nodes use.
ROOMIE_ACTUATOR_GIM8108 = _actuator_cfg(
  ("waist_yaw", ".*_hip_pitch", ".*_hip_roll", ".*_knee_pitch"), ACTUATOR_GIM8108
)
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
ROOMIE_NOMINAL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ROOMIE_ACTUATOR_GIM8108,
    ROOMIE_ACTUATOR_GIM6010,
  ),
  soft_joint_pos_limit_factor=0.9,
)

ROOMIE_ACTION_SCALE = _build_action_scale(ROOMIE_NOMINAL_ARTICULATION)


##
# Driver cascade (what the simulation applies).
##
# The real drivers run a position loop (P or PI) -> speed command capped at
# roomie_position_control_node's position_max_speed_rpm (50 rpm) -> speed loop,
# not a PD torque law (see actuators.py). Torque-equivalent gains per joint role
# from the Roomie V3 model_config.json profile full_body (left_hip_pitch fitted
# from a measured response, the rest family estimates from the firmware gains in
# steadywin_gains.yaml): kp (N m/rad), kv (N m s/rad), ki (N m/(rad s)).
# Each ankle crank is driven by its own GIM6010-36 (one motor's armature).
# The torque limit follows the datasheet torque-speed curve of ACTUATOR_GIM8108 /
# ACTUATOR_GIM6010 (full peak up to the knee speed, linear to 0 at no-load): at
# the 50 rpm speed cap about 70 N m (GIM8108) and 39 N m (GIM6010) remain.
CASCADE_VELOCITY_LIMIT = 50.0 * _RPM_TO_RAD_S


def _cascade_cfg(
  target_names_expr: tuple[str, ...], motor: ActuatorConfig, kp: float, kv: float,
  ki: float = 0.0, velocity_reference_timeconstant: float = 0.0,
) -> SteadywinCascadeActuatorCfg:
  return SteadywinCascadeActuatorCfg(
    target_names_expr=target_names_expr,
    effort_limit=motor.effort_limit,
    armature=motor.armature,
    stiffness=kp,
    damping=kv,
    integral_gain=ki,
    velocity_limit=CASCADE_VELOCITY_LIMIT,
    velocity_reference_timeconstant=velocity_reference_timeconstant,
    no_load_velocity=motor.velocity_limit,
    knee_point_velocity=motor.effective_knee_point_velocity,
    **ACTUATOR_DELAY,
  )


# GIM8108-36 (speed loop kv 16): firmware position_kp 1 / 6 (+ki) / 10 / 4 (+ki).
ROOMIE_CASCADE_WAIST = _cascade_cfg(("waist_yaw",), ACTUATOR_GIM8108, kp=53.33, kv=16.0)
ROOMIE_CASCADE_HIP_PITCH = _cascade_cfg(
  (".*_hip_pitch",), ACTUATOR_GIM8108, kp=320.0, kv=16.0, ki=200.0)
ROOMIE_CASCADE_HIP_ROLL = _cascade_cfg((".*_hip_roll",), ACTUATOR_GIM8108, kp=533.3, kv=16.0)
ROOMIE_CASCADE_KNEE = _cascade_cfg(
  (".*_knee_pitch",), ACTUATOR_GIM8108, kp=213.3, kv=16.0, ki=200.0)
# GIM6010-36 (speed loop kv 5.178): firmware position_kp 5 / 10 (+ki) / 1.
ROOMIE_CASCADE_HIP_YAW = _cascade_cfg((".*_hip_yaw",), ACTUATOR_GIM6010, kp=279.8, kv=5.178)
ROOMIE_CASCADE_ANKLE = _cascade_cfg(
  (".*_ankle_near_motor", ".*_ankle_far_motor"), ACTUATOR_GIM6010,
  kp=559.6, kv=5.178, ki=209.9)
ROOMIE_CASCADE_SHOULDER = _cascade_cfg(
  (".*_shoulder_.*",), ACTUATOR_GIM6010, kp=55.96, kv=5.178)
ROOMIE_CASCADE_ELBOW = _cascade_cfg(
  (".*_elbow_.*",), ACTUATOR_GIM6010, kp=55.36, kv=5.178,
  velocity_reference_timeconstant=0.02)

ROOMIE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ROOMIE_CASCADE_WAIST,
    ROOMIE_CASCADE_HIP_PITCH,
    ROOMIE_CASCADE_HIP_ROLL,
    ROOMIE_CASCADE_KNEE,
    ROOMIE_CASCADE_HIP_YAW,
    ROOMIE_CASCADE_ANKLE,
    ROOMIE_CASCADE_SHOULDER,
    ROOMIE_CASCADE_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_roomie_robot_cfg(
  default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
  default_collisions: tuple[CollisionCfg, ...] | None = None,
  cad_self_collision: bool = False,
) -> EntityCfg:
  """Get a fresh Roomie robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    default_keyframe: Initial state keyframe to seed each environment.
    default_collisions: Collision config tuple. Defaults to LEG_SELF_COLLISION,
      or TERRAIN_COLLISION when the CAD self-collision pieces are used.
    cad_self_collision: Use the convex CAD self-collision pieces (expensive)
      instead of proxy self-collision.
  """
  if default_collisions is None:
    default_collisions = (
      (TERRAIN_COLLISION,) if cad_self_collision else (LEG_SELF_COLLISION,)
    )
  return EntityCfg(
    init_state=default_keyframe,
    collisions=default_collisions,
    spec_fn=get_spec_with_cad_self_collision if cad_self_collision else get_spec,
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
