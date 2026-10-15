"""Unitree G1 (23-DOF) velocity tracking environment configurations.

Structure and sim2real additions (domain randomization, stochastic fall and
illegal-contact terminations) follow the Booster K1 configuration. Robot-specific
values (pose stds, contact sensors, simulation budget) follow mjlab's upstream
G1 velocity configuration (``mjlab/tasks/velocity/config/g1/env_cfgs.py``),
restricted to the joints of the 23-DOF G1.
"""

import math

from playground.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)

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

_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")

# Leg-only action space. Waist yaw and arm actuators are excluded; they hold
# their KNEES_BENT_KEYFRAME default via the entity's own PD actuators.
_LEG_JOINT_PATTERNS = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)

_G1_LEG_ACTION_SCALE = {
  k: v for k, v in G1_ACTION_SCALE.items() if k in _LEG_JOINT_PATTERNS
}


def g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()
  remove_non_deployable_actor_terms(cfg)

  # Seven capsules per foot need a larger contact budget than the base config.
  # ccd_iterations stays at mjlab's default (50): mujoco_warp preallocates an EPA
  # buffer of nconmax * num_envs * (6 + 5 * ccd_iterations) vec3s, so upstream's
  # 500 needs ~8.6 GB at 4096 envs vs ~0.9 GB at 50.
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
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
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
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
      exclude=_FOOT_BODIES,
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

  # Action space: legs only (12 joints). Joint observations stay full-body.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = _LEG_JOINT_PATTERNS
  joint_pos_action.scale = _G1_LEG_ACTION_SCALE

  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # Command curriculum. mjlab's default ranges (x up to 3 m/s, y fixed at +-1 m/s
  # for the whole run) made the G1 walk far too fast. Yaw keeps mjlab's stages.
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
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Sim2real domain randomization (as in K1): PD gains, physically consistent
  # inertia (the torso carries the payload and most of the mass, so it moves
  # further than the limbs) and terrain contact compliance.
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
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "alpha_range": (-0.05, 0.05),
      "t_range": (-0.05, 0.05),
    },
  )
  cfg.events["limb_inertia"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r"(?!torso_link$).*",)),
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

  # Restrict the pose reward to the policy-controlled leg joints. Waist and arms
  # are PD-held at their default, so scoring them would only add reward noise.
  # variable_posture needs a std key for every joint in this list.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_LEG_JOINT_PATTERNS
  )

  # Leg pose stds from mjlab's upstream G1 config:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  # Upstream G1 steps cleanly from foot_clearance + foot_swing_height alone; an
  # air-time bonus on the heavier G1 tends to induce hopping. Raise to 0.1-0.3
  # if the policy shuffles.
  cfg.rewards["air_time"].weight = 0.0

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


def g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity tracking configuration."""
  cfg = g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

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

  # Flat policies stomp harder than rough ones: nothing on a stiff, predictable
  # plane punishes a hard touchdown. At the base -1e-5 the term is ~0.01 per
  # landing on the 34 kg G1 (negligible); -5e-4 makes reducing impact force
  # worth ~0.15 per landing without making stepping itself expensive (the term
  # also counts the body-weight share of the touchdown force).
  cfg.rewards["soft_landing"].weight = -5e-4

  return cfg


def g1_flat_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity tracking configuration for FlashSAC."""
  cfg = g1_flat_env_cfg(play=play)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 1.0
  return cfg


def g1_rough_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity tracking configuration for FlashSAC."""
  cfg = g1_rough_env_cfg(play=play)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 1.0
  return cfg
