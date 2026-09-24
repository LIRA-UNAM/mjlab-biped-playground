"""Booster K1 getup environment configuration."""

from playground.asset_zoo.robots.k1.k1_parallel_constants import (
  K1_PARALLEL_ACTUATED_JOINTS,
  get_k1_parallel_robot_cfg,
)
from playground.tasks.getup import mdp
from playground.tasks.getup.getup_env_cfg import make_getup_env_cfg
from playground.tasks.getup.mdp.actions import SettleRelativeJointPositionActionCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

# Derived from home keyframe.
_TORSO_HEIGHT = 0.5125


def booster_k1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_k1_parallel_robot_cfg()}

  # The passive ankle/rod joints of the parallel linkage never reach the
  # policy: joint observations and encoder bias cover the 22 actuated joints.
  actuated_joints = SceneEntityCfg(
    "robot", joint_names=K1_PARALLEL_ACTUATED_JOINTS, preserve_order=True
  )
  for group in cfg.observations.values():
    for term_name in ("joint_pos", "joint_vel"):
      term = group.terms.get(term_name)
      if term is not None:
        term.params = {**term.params, "asset_cfg": actuated_joints}
  cfg.events["encoder_bias"].params["asset_cfg"] = actuated_joints

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Torso height. K1 has no separate waist body (trunk directly carries the
  # legs), so unlike T1/T2 there's no additional waist-height term needed to
  # avoid a "sitting" local minimum.
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("Trunk",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std: tight hips, medium knees and ankles, loose arms and
  # neck. Scored on the serial-equivalent ankle DOFs (passive pitch/roll), not
  # the crank drives or rod joints of the parallel linkage.
  cfg.rewards["posture"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_Hip_.*",
      ".*_Knee_Pitch",
      ".*_Ankle_Pitch",
      ".*_Ankle_Roll",
      "Head_.*",
      ".*_Shoulder_.*",
      ".*_Elbow_.*",
    ),
  )
  cfg.rewards["posture"].params["std"] = {
    r".*_Hip_Roll": 0.08,
    r".*_Hip_Yaw": 0.08,
    r".*_Hip_Pitch": 0.12,
    r".*_Knee_Pitch": 0.15,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.2,
    r"Head_.*": 0.15,
    r"(.*_Shoulder_.*|.*_Elbow_.*)": 0.5,
  }

  cfg.viewer.body_name = "Trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("Trunk",)
  )

  foot_geom_names = ("left_foot_collision", "right_foot_collision")
  cfg.events["geom_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )

  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 0.7
  # Randomizing the closed ankle loop would start it with a violated constraint.
  cfg.events["reset_fallen_or_standing"].params["hold_default_joint_names"] = (
    ".*_Ankle_.*",
  )

  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 50  # 1s at 50Hz action rate.
  cfg.terminations["energy"].params["settle_steps"] = 50
  # Actuators drive the cranks, not every joint: pair force and velocity by name.
  cfg.terminations["energy"].params["match_by_name"] = True

  cfg.curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
        ],
      },
    ),
    "energy_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 2500 * 24, "params": {"threshold": 3000.0}},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
