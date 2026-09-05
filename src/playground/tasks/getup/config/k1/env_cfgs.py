"""Booster K1 getup environment configuration."""

from playground.asset_zoo.robots.k1.k1_constants import get_k1_robot_cfg
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
_TORSO_HEIGHT = 0.57


def booster_k1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
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
    "robot", body_names=("trunk",)
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std: tight hips, medium knees and ankles, loose arms and neck.
  cfg.rewards["posture"].params["std"] = {
    r".*_hip_roll_joint": 0.08,
    r".*_hip_yaw_joint": 0.08,
    r".*_hip_pitch_joint": 0.12,
    r".*_knee_pitch_joint": 0.15,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.2,
    r"(aahead_yaw_joint|aahead_pitch_joint)": 0.15,
    r"(.*_shoulder.*|.*_elbow.*)": 0.5,
  }

  cfg.viewer.body_name = "trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
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

  assert isinstance(cfg.actions["joint_pos"], SettleRelativeJointPositionActionCfg)
  cfg.actions["joint_pos"].settle_steps = 50  # 1s at 50Hz action rate.
  cfg.terminations["energy"].params["settle_steps"] = 50

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
