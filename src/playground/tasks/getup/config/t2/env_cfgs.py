"""Booster T2 getup environment configuration."""

from playground.asset_zoo.robots.booster_t2.t2_constants import get_t2_robot_cfg
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
_TORSO_HEIGHT = 0.979
_WAIST_HEIGHT = 0.813


def booster_t2_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T2 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_t2_robot_cfg()}

  # T2's MJCF names its IMU sensors differently from T1's (imu_ang_vel/imu_lin_vel).
  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/angular-velocity"
  )
  cfg.observations["critic"].terms["base_ang_vel"].params["sensor_name"] = (
    "robot/angular-velocity"
  )
  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = (
    "robot/linear-velocity"
  )

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

  # Torso + waist height. Waist reward prevents "sitting on booty or knees" local
  # minimum where torso is high but waist (pelvis) stays near ground.
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["torso_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )
  cfg.rewards["waist_height"] = RewardTermCfg(
    func=mdp.height_reward,
    weight=1.0,
    params={
      "desired_height": _WAIST_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
    },
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std: tight hips, medium knees and ankles, loose arms/wrists and waist.
  cfg.rewards["posture"].params["std"] = {
    r".*_hip_roll_joint": 0.08,
    r".*_hip_yaw_joint": 0.08,
    r".*_hip_pitch_joint": 0.12,
    r".*_knee_pitch_joint": 0.15,
    r".*_ankle_pitch_joint": 0.2,
    r".*_ankle_roll_joint": 0.2,
    r"(aa_head_yaw_joint|head_pitch_joint)": 0.15,
    r"(waist_.*_joint|.*_shoulder.*_joint|.*_elbow.*_joint|.*_wrist.*_joint)": 0.5,
  }

  cfg.viewer.body_name = "trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("trunk",)
  )

  foot_geom_names = tuple(f"{side}_ankle_collision" for side in ("left", "right"))
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

  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 1.15

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
          # {"step": 600 * 24, "weight": -0.05},
          # {"step": 900 * 24, "weight": -0.08},
          # {"step": 1200 * 24, "weight": -0.1},
        ],
      },
    ),
    "joint_vel_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_l2",
        "stages": [
          {"step": 0, "weight": 0.0},
          # {"step": 900 * 24, "weight": -0.005},
          # {"step": 1200 * 24, "weight": -0.008},
          # {"step": 1500 * 24, "weight": -0.01},
        ],
      },
    ),
    "energy_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 2000 * 24, "params": {"threshold": 40000.0}},
          {"step": 2500 * 24, "params": {"threshold": 20000.0}},
          {"step": 2700 * 24, "params": {"threshold": 10000.0}},
        ],
      },
    ),
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg
