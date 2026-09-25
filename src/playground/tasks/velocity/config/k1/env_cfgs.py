"""Booster K1 velocity tracking environment configurations.

``make_velocity_env_cfg`` below is a verbatim copy of mjlab's factory of the same
name (``mjlab/tasks/velocity/velocity_env_cfg.py``, mjlab v1.6.0), kept in this
file so the whole base configuration (sensors, observations, actions, commands,
events, rewards, terminations, curriculum) can be read next to the K1 changes.
Do not edit it: ``tests/test_k1_velocity_env_cfg.py`` checks it against mjlab's.
Only the term functions (``mdp.*``) still live in mjlab.

The ``k1_*_env_cfg`` functions after it customize that base configuration for
the Booster K1.
"""

import math
from dataclasses import replace

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
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

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


##
# Base velocity configuration: verbatim copy of mjlab's make_velocity_env_cfg.
##


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    # Critic sees the true (unbiased) joint positions as privileged information.
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 2500 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 4000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


##
# K1 customization of the base configuration.
##


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
  assert cfg.sim.njmax is not None
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
  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 20_000, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 40_000, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    )
  }
  return cfg


def k1_rough_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration for FlashSAC."""
  cfg = k1_rough_env_cfg(play=play)
  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.7, 0.7)},
        ],
      },
    )
  }
  return cfg
