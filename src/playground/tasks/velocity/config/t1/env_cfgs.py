"""Booster T1 velocity tracking environment configurations."""

from dataclasses import replace

from playground.asset_zoo.robots.booster_t1.t1_constants import (
  T1_ACTION_SCALE,
  get_t1_robot_cfg,
)

from playground.tasks.velocity.mdp.actuators import pd_gains_by_target
from playground.tasks.velocity.mdp.observations import remove_non_deployable_actor_terms

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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

_LEG_JOINT_PATTERNS = (
  ".*_Hip_Pitch",
  ".*_Hip_Roll",
  ".*_Hip_Yaw",
  ".*_Knee_Pitch",
  ".*_Ankle_Pitch",
  ".*_Ankle_Roll",
)
_UPPER_BODY_JOINT_PATTERNS = (
  "Waist",
  ".*_Shoulder_Pitch",
  ".*_Shoulder_Roll",
  ".*_Elbow_Pitch",
  ".*_Elbow_Yaw",
)
# Every joint except the neck is actuated. The neck is intentionally excluded
# from the action space; it holds its HOME_KEYFRAME default via the entity's
# own PD actuator.
_NECK_JOINT_NAMES = ("AAHead_yaw", "Head_pitch")
_ACTUATED_JOINT_PATTERNS = _LEG_JOINT_PATTERNS + _UPPER_BODY_JOINT_PATTERNS

_T1_ACTION_SCALE = {
  k: v for k, v in T1_ACTION_SCALE.items() if k not in _NECK_JOINT_NAMES
}

# Command latency between policy and motors (bus/communication), in physics steps
# of 5 ms: 10-40 ms. Same values as k1_constants.ACTUATOR_DELAY.
_ACTUATOR_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)

_ANKLE_TARGETS = (".*_Ankle_.*",)
_NON_ANKLE_TARGETS = (r"(?!.*_Ankle_).*",)


def _t1_robot_cfg_with_delay() -> EntityCfg:
  """T1 robot cfg with actuator delay on every group (velocity task only)."""
  robot_cfg = get_t1_robot_cfg()
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = replace(
    robot_cfg.articulation,
    actuators=tuple(
      replace(a, **_ACTUATOR_DELAY) for a in robot_cfg.articulation.actuators
    ),
  )
  return robot_cfg


def t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()
  remove_non_deployable_actor_terms(cfg)

  cfg.scene.entities = {"robot": _t1_robot_cfg_with_delay()}

  # Set raycast sensor frame to T1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "Trunk"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )

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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Actuate legs, waist, and arms. The neck actuators keep driving their own PD
  # control toward the HOME_KEYFRAME default.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = _ACTUATED_JOINT_PATTERNS
  joint_pos_action.scale = _T1_ACTION_SCALE

  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.2  # Approx. Trunk height above ground at HOME_KEYFRAME.

  # Command curriculum, same as the G1 velocity task. mjlab's default ranges (x up
  # to 3 m/s, y fixed at +-1 m/s for the whole run) are far too fast for the T1.
  # Steps are env steps: PPO collects 24 per iteration, so the stages switch at
  # iterations 5k and 10k (mjlab's default timing).
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.3, 0.3),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-0.9, 0.9),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-0.7, 0.7),
    },
    {"step": 10000 * 24, "lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-0.7, 0.7)},
  ]

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  # Sim2real actuator randomization. The real ankle is a parallel linkage driven
  # by two motors through a serial->parallel conversion, so its effective
  # joint-space PD gains vary with configuration and its reflected inertia is
  # larger than the single-motor value modeled here. The ankle therefore gets a
  # wider gain range than the other joints (+-20%, as in K1/G1) plus an armature
  # scale (K1 doubles ankle armature). Gains are sampled per joint.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=pd_gains_by_target,
    params={
      "target_names_expr": _NON_ANKLE_TARGETS,
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
    },
  )
  cfg.events["ankle_pd_gains"] = EventTermCfg(
    mode="startup",
    func=pd_gains_by_target,
    params={
      "target_names_expr": _ANKLE_TARGETS,
      "kp_range": (0.6, 1.4),
      "kd_range": (0.5, 1.5),
    },
  )
  cfg.events["ankle_armature"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_ANKLE_TARGETS),
      "operation": "scale",
      "ranges": (1.0, 2.5),
    },
  )

  # Ramp up the action-rate penalty once a gait has emerged, reaching full
  # strength when the first command stage widens (iteration 5k). Penalizes the
  # step-to-step action changes behind hardware jitter (cf. CAPS, Mysore et al.
  # 2021, arXiv:2012.06644).
  cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "action_rate_l2",
      "stages": [
        {"step": 0, "weight": -0.1},
        {"step": 2500 * 24, "weight": -0.25},
        {"step": 5000 * 24, "weight": -0.5},
      ],
    },
  )

  # Restrict the pose reward to the actuated joints. This is required, not just
  # stylistic: variable_posture builds its std tensors positionally aligned
  # to asset_cfg's resolved joint list, and every joint in that list must be
  # covered by a std dict key or the reward crashes with a shape mismatch.
  # The neck is not part of the action space, so it is excluded from this
  # reward. Waist/arm stds follow mjlab's upstream G1 config.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_ACTUATED_JOINT_PATTERNS
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*_Hip_Pitch": 0.3,
    r".*_Hip_Roll": 0.15,
    r".*_Hip_Yaw": 0.15,
    r".*_Knee_Pitch": 0.4,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.5,
    # Waist.
    r"Waist": 0.2,
    # Arms.
    r".*_Shoulder_Pitch": 0.15,
    r".*_Shoulder_Roll": 0.15,
    r".*_Elbow_.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*_Hip_Pitch": 0.5,
    r".*_Hip_Roll": 0.2,
    r".*_Hip_Yaw": 0.2,
    r".*_Knee_Pitch": 0.6,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.5,
    # Waist.
    r"Waist": 0.3,
    # Arms.
    r".*_Shoulder_Pitch": 0.5,
    r".*_Shoulder_Roll": 0.2,
    r".*_Elbow_.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.06
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.2
  cfg.rewards["track_linear_velocity"].weight = 2.0
  # T1 defaults to FULL_COLLISION (self-collision enabled everywhere, unlike
  # Asimov's feet-only default). HOME_KEYFRAME arm angles are tuned to keep
  # arms clear of the torso/legs, but penalize any incidental self-contact
  # (e.g. from push_robot perturbations) to discourage the policy from
  # exploiting arm/torso/leg contact for balance.
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
    cfg.curriculum = {}
    # No curriculum in play: command the final training stage.
    twist_cmd.ranges.lin_vel_x = (-1.5, 1.5)
    twist_cmd.ranges.lin_vel_y = (-0.7, 0.7)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
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


def t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat terrain velocity tracking configuration."""
  cfg = t1_rough_env_cfg(play=play)

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

def t1_flat_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat terrain velocity tracking configuration for FlashSAC."""
  cfg = t1_flat_env_cfg(play=play)

  cfg.actions["joint_pos"].scale = 1.0
  return cfg

def t1_rough_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 rough terrain velocity tracking configuration for FlashSAC."""
  cfg = t1_rough_env_cfg(play=play)

  cfg.actions["joint_pos"].scale = 1.0
  return cfg