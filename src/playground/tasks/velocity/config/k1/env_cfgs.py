"""Booster K1 velocity tracking environment configurations."""

import math

from playground.asset_zoo.robots.k1.k1_parallel_constants import (
  K1_PARALLEL_ACTION_SCALE,
  K1_PARALLEL_ACTUATED_JOINTS,
  K1_PARALLEL_FREE_RESET_JOINTS,
  get_k1_parallel_robot_cfg,
)

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
  ".*_Hip_Pitch",
  ".*_Hip_Roll",
  ".*_Hip_Yaw",
  ".*_Knee_Pitch",
  ".*_Ankle_A",
  ".*_Ankle_B",
)

# Arm actuator patterns. These match the keys of K1_PARALLEL_ACTION_SCALE.
_ARM_ACTUATOR_PATTERNS = (
  ".*_Shoulder_.*",
  ".*_Elbow_.*",
)

# Legs + arms are the action space. The head is intentionally excluded; it
# holds its HOME_KEYFRAME default via the entity's own PD actuator.
_ACTION_JOINT_PATTERNS = _LEG_ACTUATOR_PATTERNS + _ARM_ACTUATOR_PATTERNS

_K1_ACTION_SCALE = {
  k: v for k, v in K1_PARALLEL_ACTION_SCALE.items() if k in _ACTION_JOINT_PATTERNS
}

# Joints seen by the policy/critic (joint_pos, joint_vel) and given an encoder
# bias: the 20 actuated non-head joints in MJCF order. The passive ankle
# pitch/roll and rod joints of the closed loop are excluded.
OBS_JOINT_NAMES = tuple(
  n for n in K1_PARALLEL_ACTUATED_JOINTS if not n.startswith("Head_")
)

# Joints scored by the pose reward: the serial-equivalent ankle DOFs (passive
# pitch/roll, so the std tables keep their meaning) plus the arms.
_POSE_JOINT_PATTERNS = (
  ".*_Hip_Pitch",
  ".*_Hip_Roll",
  ".*_Hip_Yaw",
  ".*_Knee_Pitch",
  ".*_Ankle_Pitch",
  ".*_Ankle_Roll",
  *_ARM_ACTUATOR_PATTERNS,
)


def k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_k1_parallel_robot_cfg()}

  # Set raycast sensor frame to K1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "Trunk"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")

  # Wire foot height scan to per-foot sites.
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
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
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
      exclude=("left_foot_link", "right_foot_link"),
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

  # Action space: legs + arms. The head actuators keep driving their own PD
  # control toward the HOME_KEYFRAME default.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = _ACTION_JOINT_PATTERNS
  joint_pos_action.scale = _K1_ACTION_SCALE

  # The passive ankle/rod joints of the parallel linkage never reach the
  # policy: joint observations and encoder bias cover the 20 action joints.
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
    "robot", joint_names=K1_PARALLEL_FREE_RESET_JOINTS
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  # Four `connect` equalities add 12 constraint rows per environment.
  cfg.sim.njmax += 100

  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.9  # Approx. trunk height above ground at HOME_KEYFRAME.

  # Conservative velocity ranges given K1's narrow ankle-roll ROM
  # (+-0.345 rad) and asymmetric hip-roll ROM, mirroring T1's caution.
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  # Sim2real domain randomization ported from booster_mjlab: PD gains, physically
  # consistent inertia (the trunk carries the payload, so it moves further than
  # the limbs) and terrain contact compliance.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
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
      "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      "alpha_range": (-0.05, 0.05),
      "t_range": (-0.05, 0.05),
    },
  )
  cfg.events["limb_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r"(?!Trunk$).*",)),
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

  # Stochastic fall termination (gentler recovery signal than a hard 70 deg
  # cutoff) plus termination on any non-foot ground contact.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=stochastic_bad_orientation,
    params={"limit_angle": math.radians(63.0), "probability": 0.02},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Restrict the pose reward to leg + arm joints. This is required, not just
  # stylistic: variable_posture builds its std tensors positionally aligned
  # to asset_cfg's resolved joint list, and every joint in that list must be
  # covered by a std dict key or the reward crashes with a shape mismatch.
  # The head (not actuated by the policy) and the passive linkage joints are
  # excluded. Arm stds are tighter than the leg ones: arms may counter-swing a
  # little but are pulled back to the home pose, away from the trunk.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_POSE_JOINT_PATTERNS
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_Hip_Pitch": 0.3,
    r".*_Hip_Roll": 0.15,
    r".*_Hip_Yaw": 0.15,
    r".*_Knee_Pitch": 0.4,
    r".*_Ankle_Pitch": 0.15,
    r".*_Ankle_Roll": 0.1,
    r".*_Shoulder_.*": 0.08,
    r".*_Elbow_.*": 0.08,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_Hip_Pitch": 0.5,
    r".*_Hip_Roll": 0.2,
    r".*_Hip_Yaw": 0.2,
    r".*_Knee_Pitch": 0.6,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.12,
    r".*_Shoulder_.*": 0.1,
    r".*_Elbow_.*": 0.1,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.06
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.3

  # K1 defaults to FULL_COLLISION (self-collision enabled everywhere, unlike
  # Asimov's feet-only default). HOME_KEYFRAME arm angles are tuned to keep
  # arms clear of the torso/legs, and the arms are now part of the action
  # space, so penalize any self-contact (arm/torso/leg) to discourage the
  # policy from exploiting it for balance.
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


def k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain velocity tracking configuration."""
  cfg = k1_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.2)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def k1_flat_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain velocity tracking configuration for FlashSAC."""
  cfg = k1_flat_env_cfg(play=play)

  cfg.actions["joint_pos"].scale = 1.0
  return cfg


def k1_rough_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration for FlashSAC."""
  cfg = k1_rough_env_cfg(play=play)

  cfg.actions["joint_pos"].scale = 1.0
  return cfg
