"""Booster K1 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = Path(__file__).parent / "xmls" / "k1.xml"
assert K1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(K1_XML))


##
# Actuator config.
#
# Booster hasn't published a K1 motor datasheet, so (as with T2) these
# aren't built from ElectricActuator(reflected_inertia(rotor_inertia,
# gear_ratio), ...). Instead, reflected_inertia and effort_limit are read
# directly off the source K1_22dof.xml/K1_22dof_parallel.xml: <joint
# armature=...> is already the gearbox-reflected rotor inertia, and the
# <motor forcerange=...> (stripped from k1.xml, actuators are defined here
# instead) is the torque limit. velocity_limit has no equivalent field in
# the XML, so it's left at 0.0 (unused below: _kp/_kv only read
# reflected_inertia).
#
# Joints are grouped by matching (armature, forcerange) pairs in the source
# XML.
##

NECK_ACTUATOR = ElectricActuator(
  reflected_inertia=0.002,
  velocity_limit=0.0,
  effort_limit=6.0,
)

ARM_ACTUATOR = ElectricActuator(
  reflected_inertia=0.001,
  velocity_limit=0.0,
  effort_limit=14.0,
)

HIP_PITCH_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0478125,
  velocity_limit=0.0,
  effort_limit=68.0,
)

HIP_ROLL_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0339552,
  velocity_limit=0.0,
  effort_limit=43.0,
)

HIP_YAW_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0282528,
  velocity_limit=0.0,
  effort_limit=38.3,
)

KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=0.095625,
  velocity_limit=0.0,
  effort_limit=112.0,
)

ANKLE_ACTUATOR = ElectricActuator(
  reflected_inertia=0.0565,
  velocity_limit=0.0,
  effort_limit=38.3,
)

NATURAL_FREQ = 5.0 * 2.0 * 3.14159265  # 5 Hz
DAMPING_RATIO = 2.0


def _kp(act: ElectricActuator) -> float:
  return act.reflected_inertia * NATURAL_FREQ**2


def _kv(act: ElectricActuator) -> float:
  return 2.0 * DAMPING_RATIO * act.reflected_inertia * NATURAL_FREQ


K1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("aahead_yaw_joint", "aahead_pitch_joint"),
  stiffness=_kp(NECK_ACTUATOR),
  damping=_kv(NECK_ACTUATOR),
  effort_limit=NECK_ACTUATOR.effort_limit,
  armature=NECK_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "aaleft_shoulder_pitch_joint",
    "aaright_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint",
  ),
  stiffness=_kp(ARM_ACTUATOR),
  damping=_kv(ARM_ACTUATOR),
  effort_limit=ARM_ACTUATOR.effort_limit,
  armature=ARM_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=_kp(HIP_PITCH_ACTUATOR),
  damping=_kv(HIP_PITCH_ACTUATOR),
  effort_limit=HIP_PITCH_ACTUATOR.effort_limit,
  armature=HIP_PITCH_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=_kp(HIP_ROLL_ACTUATOR),
  damping=_kv(HIP_ROLL_ACTUATOR),
  effort_limit=HIP_ROLL_ACTUATOR.effort_limit,
  armature=HIP_ROLL_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=_kp(HIP_YAW_ACTUATOR),
  damping=_kv(HIP_YAW_ACTUATOR),
  effort_limit=HIP_YAW_ACTUATOR.effort_limit,
  armature=HIP_YAW_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_pitch_joint",),
  stiffness=_kp(KNEE_ACTUATOR),
  damping=_kv(KNEE_ACTUATOR),
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

K1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=_kp(ANKLE_ACTUATOR),
  damping=_kv(ANKLE_ACTUATOR),
  effort_limit=ANKLE_ACTUATOR.effort_limit,
  armature=ANKLE_ACTUATOR.reflected_inertia,
)

##
# Keyframes.
##

# pos.z is the height at which the crouched pose below clears the floor,
# found by forward-kinematics with the base fixed at the origin (lowest
# collision point, left_foot_collision, bottoms out around z=-0.543 in that
# pose; a small margin is added on top).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.57),
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_foot_collision$"

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

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_NECK,
    K1_ACTUATOR_ARM,
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE,
    K1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh K1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  target = a.target_names_expr
  assert e is not None
  for n in target:
    K1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_k1_robot_cfg())

  viewer.launch(robot.spec.compile())
