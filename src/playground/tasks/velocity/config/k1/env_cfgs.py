"""Booster K1 velocity tracking environment configurations."""

from playground.asset_zoo.robots.k1.k1_constants import (
  K1_ACTION_SCALE,
  get_k1_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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

# Leg-only actuator/joint patterns. Neck and arm actuators are intentionally
# excluded from the action space; they hold their HOME_KEYFRAME default via
# the entity's own PD actuators, keeping the upper body static and clear of
# the legs while walking.
_LEG_JOINT_PATTERNS = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_pitch_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)

_K1_LEG_ACTION_SCALE = {
  k: v for k, v in K1_ACTION_SCALE.items() if k in _LEG_JOINT_PATTERNS
}


def k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  # Set raycast sensor frame to K1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"

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
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
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

  # Restrict the action space to legs only. Neck/arm actuators keep driving
  # their own PD control toward the HOME_KEYFRAME default, holding the upper
  # body static and out of the way of the legs.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = _LEG_JOINT_PATTERNS
  joint_pos_action.scale = _K1_LEG_ACTION_SCALE

  cfg.viewer.body_name = "trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.9  # Approx. trunk height above ground at HOME_KEYFRAME.

  # Conservative velocity ranges given K1's narrow ankle-roll ROM
  # (+-0.345 rad) and asymmetric hip-roll ROM, mirroring T1's caution.
  twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  # Restrict the pose reward to leg joints only. This is required, not just
  # stylistic: variable_posture builds its std tensors positionally aligned
  # to asset_cfg's resolved joint list, and every joint in that list must be
  # covered by a std dict key or the reward crashes with a shape mismatch.
  # Since arms/neck are not part of the action space, exclude them entirely
  # from this reward.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_LEG_JOINT_PATTERNS
  )
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_pitch_joint": 0.3,
    r".*_hip_roll_joint": 0.15,
    r".*_hip_yaw_joint": 0.15,
    r".*_knee_pitch_joint": 0.4,
    r".*_ankle_pitch_joint": 0.15,
    r".*_ankle_roll_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_pitch_joint": 0.5,
    r".*_hip_roll_joint": 0.2,
    r".*_hip_yaw_joint": 0.2,
    r".*_knee_pitch_joint": 0.6,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.12,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.06
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.3

  # K1 defaults to FULL_COLLISION (self-collision enabled everywhere, unlike
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
