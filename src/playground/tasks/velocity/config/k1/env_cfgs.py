"""Booster K1 velocity tracking environment configurations.

The whole environment is built here, one section per manager, so every
observation, action, command, event, reward, termination and curriculum term
of the K1 velocity tasks is visible in this file. Only the term functions
(``mdp.*``) come from mjlab.

``_make_k1_velocity_env_cfg`` assembles the sections. The variants differ as follows:

* rough: terrain generator, ``terrain_scan`` sensor, ``height_scan`` observation,
  terrain-level curriculum and ``out_of_terrain_bounds`` termination.
  Flat has none of these (plane terrain).
* play: infinite episodes, no observation noise, no pushes, no illegal-contact
  termination, no curriculum, fixed 5x5 terrain grid.
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

##
# K1 constants
##

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

_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
_FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")

# Max ray length of the `terrain_scan` sensor; the `height_scan` observation is
# scaled by its inverse.
_TERRAIN_SCAN_MAX_DISTANCE = 5.0


##
# Sensors
##


def _sensors(*, rough: bool) -> tuple:
  # Height map under the trunk (rough terrain only; feeds `height_scan`).
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="Trunk", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=_TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  # Height of each foot above the terrain (foot clearance / swing rewards).
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in _FOOT_SITE_NAMES),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
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

  # Foot <-> terrain contact: air time, contact flags and forces.
  feet_ground_contact = ContactSensorCfg(
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

  # Any non-foot body touching the terrain is an illegal contact (fall).
  non_foot_ground_contact = ContactSensorCfg(
    name="non_foot_ground_contact",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=r".*",
      exclude=_FOOT_BODY_NAMES,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  # Robot touching itself (arm/torso/leg); penalized by `self_collisions`.
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  return (
    *((terrain_scan,) if rough else ()),
    foot_height_scan,
    feet_ground_contact,
    non_foot_ground_contact,
    self_collision,
  )


##
# Observations
##


def _observations(*, rough: bool, play: bool) -> dict[str, ObservationGroupCfg]:
  # The passive ankle/rod joints of the parallel linkage never reach the
  # policy: joint observations and encoder bias cover the 20 action joints.
  obs_joints = SceneEntityCfg("robot", joint_names=OBS_JOINT_NAMES, preserve_order=True)

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
      params={"biased": True, "asset_cfg": obs_joints},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": obs_joints},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  if rough:
    actor_terms["height_scan"] = ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / _TERRAIN_SCAN_MAX_DISTANCE,
    )

  # The critic reuses the actor terms (its group has corruption disabled), sees
  # the true (unbiased) joint positions, and adds privileged foot terms.
  critic_terms = {
    **actor_terms,
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"asset_cfg": obs_joints}
    ),
  }
  if rough:
    critic_terms["height_scan"] = ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / _TERRAIN_SCAN_MAX_DISTANCE,
    )
  critic_terms |= {
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

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


##
# Actions
##


def _actions(*, action_scale: float | dict[str, float]) -> dict[str, ActionTermCfg]:
  # Legs + arms; the head actuators keep driving their own PD control toward
  # the HOME_KEYFRAME default.
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_ACTION_JOINT_PATTERNS,
      scale=action_scale,
      use_default_offset=True,
    )
  }


##
# Commands
##


def _commands(*, rough: bool, play: bool) -> dict[str, CommandTermCfg]:
  # Conservative velocity ranges given K1's narrow ankle-roll ROM
  # (+-0.345 rad) and asymmetric hip-roll ROM, mirroring T1's caution.
  # NOTE: the `command_vel` curriculum below overwrites lin_vel_x / ang_vel_z
  # with its own stage ranges from the first step on.
  lin_vel_x, ang_vel_z = (-0.8, 0.8), (-0.6, 0.6)
  if play and not rough:
    lin_vel_x, ang_vel_z = (-1.0, 1.2), (-0.7, 0.7)

  return {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      viz=UniformVelocityCommandCfg.VizCfg(
        z_offset=0.9  # Approx. trunk height above ground at HOME_KEYFRAME.
      ),
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=lin_vel_x,
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=ang_vel_z,
        heading=(-math.pi, math.pi),
      ),
    )
  }


##
# Events
##


def _events(*, play: bool) -> dict[str, EventTermCfg]:
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
    # Randomizing the closed loop would start the ankles with a violated
    # constraint, so resets only perturb joints outside it.
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.1, 0.1),
        "asset_cfg": SceneEntityCfg("robot", joint_names=K1_PARALLEL_FREE_RESET_JOINTS),
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
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=OBS_JOINT_NAMES, preserve_order=True
        ),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    # Sim2real domain randomization ported from booster_mjlab: PD gains,
    # physically consistent inertia (the trunk carries the payload, so it moves
    # further than the limbs) and terrain contact compliance.
    "pd_gains": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
        "operation": "scale",
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    ),
    "trunk_inertia": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
        "alpha_range": (-0.05, 0.05),
        "t_range": (-0.05, 0.05),
      },
    ),
    "limb_inertia": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(r"(?!Trunk$).*",)),
        "alpha_range": (-0.05, 0.05),
        "t_range": (-0.025, 0.025),
      },
    ),
    "terrain_contact": EventTermCfg(
      mode="startup",
      func=randomize_terrain_contact,
      params={
        "asset_cfg": SceneEntityCfg("terrain"),
        "solref_ranges": {0: (0.006, 0.03), 1: (0.95, 1.05)},
        "solimp_ranges": {0: (0.88, 0.92), 1: (0.94, 0.99), 2: (0.003, 0.01)},
        "shared_random": True,
      },
    ),
  }

  if play:
    del events["push_robot"]
    events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
  return events


##
# Rewards
##


def _rewards() -> dict[str, RewardTermCfg]:
  return {
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
        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
      },
    ),
    # Restricted to leg + arm joints. This is required, not just stylistic:
    # variable_posture builds its std tensors positionally aligned to
    # asset_cfg's resolved joint list, and every joint in that list must be
    # covered by a std dict key or the reward crashes with a shape mismatch.
    # The head (not actuated by the policy) and the passive linkage joints are
    # excluded. Arm stds are tighter than the leg ones: arms may counter-swing a
    # little but are pulled back to the home pose, away from the trunk.
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_POSE_JOINT_PATTERNS),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": {
          r".*_Hip_Pitch": 0.3,
          r".*_Hip_Roll": 0.15,
          r".*_Hip_Yaw": 0.15,
          r".*_Knee_Pitch": 0.4,
          r".*_Ankle_Pitch": 0.15,
          r".*_Ankle_Roll": 0.1,
          r".*_Shoulder_.*": 0.08,
          r".*_Elbow_.*": 0.08,
        },
        "std_running": {
          r".*_Hip_Pitch": 0.5,
          r".*_Hip_Roll": 0.2,
          r".*_Hip_Yaw": 0.2,
          r".*_Knee_Pitch": 0.6,
          r".*_Ankle_Pitch": 0.2,
          r".*_Ankle_Roll": 0.12,
          r".*_Shoulder_.*": 0.1,
          r".*_Elbow_.*": 0.1,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.06,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.3,
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
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
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
        "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITE_NAMES),
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
    # K1 defaults to FULL_COLLISION (self-collision enabled everywhere).
    # HOME_KEYFRAME arm angles are tuned to keep arms clear of the torso/legs,
    # and the arms are part of the action space, so penalize any self-contact
    # (arm/torso/leg) to discourage the policy from exploiting it for balance.
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


##
# Terminations
##


def _terminations(*, rough: bool, play: bool) -> dict[str, TerminationTermCfg]:
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Stochastic fall termination (gentler recovery signal than a hard 70 deg
    # cutoff).
    "fell_over": TerminationTermCfg(
      func=stochastic_bad_orientation,
      params={"limit_angle": math.radians(63.0), "probability": 0.02},
    ),
  }
  if rough and not play:
    terminations["out_of_terrain_bounds"] = TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    )
  if not play:
    terminations["illegal_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "non_foot_ground_contact"},
    )
  return terminations


##
# Curriculum
##


def _curriculum(*, rough: bool, play: bool) -> dict[str, CurriculumTermCfg]:
  if play:
    return {}
  curriculum = {}
  if rough:
    curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    )
  curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
      ],
    },
  )
  return curriculum


##
# Assembly
##


def _make_k1_velocity_env_cfg(
  *,
  rough: bool,
  play: bool,
  action_scale: float | dict[str, float] = _K1_ACTION_SCALE,
) -> ManagerBasedRlEnvCfg:
  if rough:
    terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=(
        replace(
          ROUGH_TERRAINS_CFG,
          curriculum=False,
          num_cols=5,
          num_rows=5,
          border_width=10.0,
        )
        if play
        else replace(ROUGH_TERRAINS_CFG, curriculum=True)
      ),
      max_init_terrain_level=5,
    )
  else:
    terrain = TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=5,
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=terrain,
      entities={"robot": get_k1_parallel_robot_cfg()},
      sensors=_sensors(rough=rough),
      num_envs=1,
      extent=2.0,
    ),
    observations=_observations(rough=rough, play=play),
    actions=_actions(action_scale=action_scale),
    commands=_commands(rough=rough, play=play),
    events=_events(play=play),
    rewards=_rewards(),
    terminations=_terminations(rough=rough, play=play),
    curriculum=_curriculum(rough=rough, play=play),
    metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="Trunk",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      # 1500 base + 100 for the four `connect` equalities of the ankle linkages
      # (12 constraint rows per environment).
      njmax=1600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    # Effectively infinite episode length when playing.
    episode_length_s=int(1e9) if play else 20.0,
  )


def k1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration."""
  return _make_k1_velocity_env_cfg(rough=True, play=play)


def k1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain velocity tracking configuration."""
  return _make_k1_velocity_env_cfg(rough=False, play=play)


def k1_flat_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain velocity tracking configuration for FlashSAC."""
  return _make_k1_velocity_env_cfg(rough=False, play=play, action_scale=1.0)


def k1_rough_env_cfg_flashsac(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 rough terrain velocity tracking configuration for FlashSAC."""
  return _make_k1_velocity_env_cfg(rough=True, play=play, action_scale=1.0)
