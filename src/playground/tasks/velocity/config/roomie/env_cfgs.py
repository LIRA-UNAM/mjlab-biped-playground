"""Roomie V3 velocity tracking environment configurations.

Built on mjlab's ``make_velocity_env_cfg`` and customized like the parallel-ankle
Booster K1 task: the ankles are driven by the two crank joints of each linkage,
while the ankle pitch/roll and rod joints stay passive and hidden from the policy.
The policy controls the legs and waist: the arms are servoed to their
HOME_KEYFRAME pose by a reset event that sets their position targets. Actuator gains are placeholders (see
``roomie_constants``).
"""

import math

from playground.asset_zoo.robots.roomie.roomie_constants import (
  ROOMIE_ACTION_SCALE,
  ROOMIE_ACTUATED_JOINTS,
  ROOMIE_FREE_RESET_JOINTS,
  get_roomie_robot_cfg,
)

from playground.tasks.velocity.mdp.events import hold_default_joint_targets
from playground.tasks.velocity.mdp.observations import remove_non_deployable_actor_terms
from playground.tasks.velocity.mdp.terminations import stochastic_bad_orientation
from playground.tasks.velocity.mdp.terrain import randomize_terrain_contact

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Leg actuator patterns. The ankle is driven by the two crank joints of the
# parallel linkage (the ankle pitch/roll joints are passive).
_LEG_ACTUATOR_PATTERNS = (
  ".*_hip_pitch",
  ".*_hip_roll",
  ".*_hip_yaw",
  ".*_knee_pitch",
  ".*_ankle_near_motor",
  ".*_ankle_far_motor",
)

# Arm joints: actuated, but held at HOME_KEYFRAME by hold_default_joint_targets.
_ARM_JOINT_PATTERNS = (".*_shoulder_.*", ".*_elbow_.*")

# Waist + legs: 13 motors are the action space. The arms are left out until
# their model is fixed; they hold the HOME_KEYFRAME pose.
_ACTION_JOINT_PATTERNS = ("waist_yaw",) + _LEG_ACTUATOR_PATTERNS

_ROOMIE_ACTION_SCALE = {
  k: v for k, v in ROOMIE_ACTION_SCALE.items() if k in _ACTION_JOINT_PATTERNS
}

# Joints seen by the policy/critic (joint_pos, joint_vel) and given an encoder
# bias: the 13 actuated waist and leg joints in MJCF order. The arms and the
# passive ankle pitch/roll and rod joints of the closed loop are excluded.
OBS_JOINT_NAMES = tuple(
  n for n in ROOMIE_ACTUATED_JOINTS if not any(arm in n for arm in ("_shoulder_", "_elbow_"))
)

# Joints scored by the pose reward: the serial-equivalent ankle DOFs (passive
# pitch/roll, so the std tables keep their meaning) plus the waist and other leg joints.
_POSE_JOINT_PATTERNS = (
  "waist_yaw",
  ".*_hip_pitch",
  ".*_hip_roll",
  ".*_hip_yaw",
  ".*_knee_pitch",
  ".*_ankle_pitch",
  ".*_ankle_roll",
)

_TRUNK = "torso"
_FEET = ("left_foot", "right_foot")


def roomie_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Roomie V3 rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()
  remove_non_deployable_actor_terms(cfg)

  cfg.scene.entities = {"robot": get_roomie_robot_cfg()}

  # Set raycast sensor frame to the Roomie torso.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = _TRUNK

  site_names = _FEET
  geom_names = ("left_foot_contact", "right_foot_contact")

  # Wire foot height scan to per-foot sole sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot|right_foot)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Robot self-contact: only left-leg vs right-leg proxies can touch (see
  # LEG_SELF_COLLISION).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=_TRUNK, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=_TRUNK, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Any non-foot body touching the terrain is an illegal contact (fall).
  nonfoot_ground_cfg = ContactSensorCfg(
    name="non_foot_ground_contact",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=r".*",
      exclude=_FEET,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = _ACTION_JOINT_PATTERNS
  joint_pos_action.scale = _ROOMIE_ACTION_SCALE

  # Only the waist and leg motors reach the policy: joint observations and
  # encoder bias cover the 13 action joints (no arms or passive linkage joints).
  obs_joints = SceneEntityCfg("robot", joint_names=OBS_JOINT_NAMES, preserve_order=True)
  for group in cfg.observations.values():
    for term_name in ("joint_pos", "joint_vel"):
      term = group.terms.get(term_name)
      if term is not None:
        term.params = {**term.params, "asset_cfg": obs_joints}
  cfg.events["encoder_bias"].params["asset_cfg"] = obs_joints

  # Randomizing the closed loop would start the ankles with a violated
  # constraint, so resets only perturb joints outside it.
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=ROOMIE_FREE_RESET_JOINTS
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  # The arms are outside the action space; servo them to the HOME_KEYFRAME pose
  # (resets zero every joint target).
  cfg.events["hold_arm_targets"] = EventTermCfg(
    func=hold_default_joint_targets,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_JOINT_PATTERNS)},
  )

  # Four `connect` equalities add 12 constraint rows per environment.
  assert cfg.sim.njmax is not None
  cfg.sim.njmax += 100

  cfg.viewer.body_name = _TRUNK

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0  # Approx. torso top height at HOME_KEYFRAME.

  # Fixed velocity command ranges. The base config's velocity curriculum would
  # overwrite them (up to 3 m/s forward), so it is removed below.
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  cfg.curriculum.pop("command_vel", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (_TRUNK,)

  # Sim2real domain randomization, as in the K1 task: PD gains, physically
  # consistent inertia (the torso carries the battery, so it moves further than
  # the limbs) and terrain contact compliance.
  # PD gains are resampled every episode (scaled from the defaults, so draws do
  # not compound), covering the identified-vs-actual gain mismatch of the
  # datasheet-derived stiffness/damping.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
      "operation": "scale",
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
    },
  )
  cfg.events["trunk_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(_TRUNK,)),
      "alpha_range": (-0.05, 0.05),
      "t_range": (-0.05, 0.05),
    },
  )
  cfg.events["limb_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(rf"(?!{_TRUNK}$).*",)),
      "alpha_range": (-0.05, 0.05),
      "t_range": (-0.025, 0.025),
    },
  )
  cfg.events["terrain_contact"] = EventTermCfg(
    mode="startup",
    func=randomize_terrain_contact,
    params={
      "asset_cfg": SceneEntityCfg("terrain"),
      "solref_ranges": {0: (0.006, 0.03), 1: (0.95, 1.05)},
      "solimp_ranges": {0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)},
      "shared_random": True,
    },
  )

  # Stochastic fall termination plus termination on any non-foot ground contact.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=stochastic_bad_orientation,
    params={"limit_angle": math.radians(63.0), "probability": 0.02},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Restrict the pose reward to the waist and leg joints. variable_posture
  # builds its std tensors positionally aligned to asset_cfg's resolved joint
  # list, so every joint in that list must be covered by a std dict key. The
  # arms are not policy-controlled, and the passive linkage joints are excluded.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_POSE_JOINT_PATTERNS
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r"waist_yaw": 0.1,
    r".*_hip_pitch": 0.3,
    r".*_hip_roll": 0.15,
    r".*_hip_yaw": 0.15,
    r".*_knee_pitch": 0.4,
    r".*_ankle_pitch": 0.15,
    r".*_ankle_roll": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"waist_yaw": 0.15,
    r".*_hip_pitch": 0.5,
    r".*_hip_roll": 0.2,
    r".*_hip_yaw": 0.2,
    r".*_knee_pitch": 0.6,
    r".*_ankle_pitch": 0.2,
    r".*_ankle_roll": 0.12,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (_TRUNK,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_TRUNK,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.06
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.3

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("pd_gains", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def roomie_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Roomie V3 flat terrain velocity tracking configuration."""
  cfg = roomie_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and the critic's height scan (no terrain to scan). The
  # actor never sees it (see remove_non_deployable_actor_terms).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
