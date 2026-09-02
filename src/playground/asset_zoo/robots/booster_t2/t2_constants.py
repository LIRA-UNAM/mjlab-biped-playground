"""Booster T2 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

T2_XML: Path = Path(__file__).parent / "T2_31dof.xml"
assert T2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(T2_XML))


##
# Actuator config.
#
# Booster hasn't published a T2 motor datasheet (no rotor inertia / gear ratio
# / rated speed table like T1's Feishu wiki), so these aren't built from
# ElectricActuator(reflected_inertia(rotor_inertia, gear_ratio), ...) the way
# T1's are. Instead, reflected_inertia and effort_limit are read directly off
# each joint in T2_31dof.xml: <joint armature=...> is already the gearbox-
# reflected rotor inertia (computed upstream by Booster's own export), and
# <joint actuatorfrcrange=...> is the torque limit. velocity_limit has no
# equivalent field in the XML, so it's left at 0.0 (unused below: _kp/_kv only
# read reflected_inertia).
#
# Joints are grouped by matching (armature, actuatorfrcrange) pairs in the
# XML, which line up with what are presumably shared actuator SKUs across
# joint roles (e.g. waist_yaw/hip/knee all show the same 0.0281 / 135 N*m).
##

NECK_WRIST_ACTUATOR = ElectricActuator(
  reflected_inertia=0.007209,
  velocity_limit=0.0,
  effort_limit=22.0,
)

SHOULDER_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=0.017,
  velocity_limit=0.0,
  effort_limit=74.0,
)

ARM_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0137,
  velocity_limit=0.0,
  effort_limit=60.0,
)

WAIST_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0395142798,
  velocity_limit=0.0,
  effort_limit=138.0,
)

WAIST_ROLL_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0524116653,
  velocity_limit=0.0,
  effort_limit=144.0,
)

WAIST_YAW_HIP_KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0281,
  velocity_limit=0.0,
  effort_limit=135.0,
)

ANKLE_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0358941096,
  velocity_limit=0.0,
  effort_limit=132.0,
)

ANKLE_ROLL_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0133356060,
  velocity_limit=0.0,
  effort_limit=72.0,
)

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265  # 5 Hz
DAMPING_RATIO = 2.0


def _kp(act: ElectricActuator) -> float:
  return act.reflected_inertia * NATURAL_FREQ**2


def _kv(act: ElectricActuator) -> float:
  return 2.0 * DAMPING_RATIO * act.reflected_inertia * NATURAL_FREQ


T2_ACTUATOR_NECK_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "aa_head_yaw_joint",
    "head_pitch_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=_kp(NECK_WRIST_ACTUATOR),
  damping=_kv(NECK_WRIST_ACTUATOR),
  effort_limit=NECK_WRIST_ACTUATOR.effort_limit,
  armature=NECK_WRIST_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",),
  stiffness=_kp(SHOULDER_PITCH_ACTUATOR),
  damping=_kv(SHOULDER_PITCH_ACTUATOR),
  effort_limit=SHOULDER_PITCH_ACTUATOR.effort_limit,
  armature=SHOULDER_PITCH_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint",
  ),
  stiffness=_kp(ARM_ACTUATOR),
  damping=_kv(ARM_ACTUATOR),
  effort_limit=ARM_ACTUATOR.effort_limit,
  armature=ARM_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_WAIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint",),
  stiffness=_kp(WAIST_PITCH_ACTUATOR),
  damping=_kv(WAIST_PITCH_ACTUATOR),
  effort_limit=WAIST_PITCH_ACTUATOR.effort_limit,
  armature=WAIST_PITCH_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint",),
  stiffness=_kp(WAIST_ROLL_ACTUATOR),
  damping=_kv(WAIST_ROLL_ACTUATOR),
  effort_limit=WAIST_ROLL_ACTUATOR.effort_limit,
  armature=WAIST_ROLL_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_WAIST_YAW_HIP_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_pitch_joint",
  ),
  stiffness=_kp(WAIST_YAW_HIP_KNEE_ACTUATOR),
  damping=_kv(WAIST_YAW_HIP_KNEE_ACTUATOR),
  effort_limit=WAIST_YAW_HIP_KNEE_ACTUATOR.effort_limit,
  armature=WAIST_YAW_HIP_KNEE_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=_kp(ANKLE_PITCH_ACTUATOR),
  damping=_kv(ANKLE_PITCH_ACTUATOR),
  effort_limit=ANKLE_PITCH_ACTUATOR.effort_limit,
  armature=ANKLE_PITCH_ACTUATOR.reflected_inertia,
)

T2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=_kp(ANKLE_ROLL_ACTUATOR),
  damping=_kv(ANKLE_ROLL_ACTUATOR),
  effort_limit=ANKLE_ROLL_ACTUATOR.effort_limit,
  armature=ANKLE_ROLL_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

# pos.z is the height at which the crouched pose below clears the floor,
# found by forward-kinematics with the base fixed at the origin (feet bottom
# out around z=-0.975 in that pose; a small margin is added on top).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.98),
  joint_pos={
    "left_shoulder_roll_joint": -1.3,
    "right_shoulder_roll_joint": 1.3,
    "left_elbow_yaw_joint": -0.4,
    "right_elbow_yaw_joint": 0.4,
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_ankle_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision": 3},
  friction={_foot_regex: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
  priority=1,
)

##
# Final config.
##

T2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T2_ACTUATOR_NECK_WRIST,
    T2_ACTUATOR_SHOULDER_PITCH,
    T2_ACTUATOR_ARM,
    T2_ACTUATOR_WAIST_PITCH,
    T2_ACTUATOR_WAIST_ROLL,
    T2_ACTUATOR_WAIST_YAW_HIP_KNEE,
    T2_ACTUATOR_ANKLE_PITCH,
    T2_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t2_robot_cfg() -> EntityCfg:
  """Get a fresh T2 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=T2_ARTICULATION,
  )


T2_ACTION_SCALE: dict[str, float] = {}
for a in T2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  target = a.target_names_expr
  assert e is not None
  for n in target:
    T2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_t2_robot_cfg())

  viewer.launch(robot.spec.compile())
