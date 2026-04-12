"""Asimov bipedal robot velocity tracking environment configurations."""

from mjlab.asset_zoo.robots.asimov.asimov_constants import (
    ASIMOV_ACTION_SCALE,
    ASIMOV_WALKING_REFERENCE,
    get_asimov_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationTermCfg, RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def asimov_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Asimov rough terrain velocity tracking configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_asimov_robot_cfg()}

    # Asimov feet sites
    site_names = ("left_foot", "right_foot")
    geom_names = (
        "left_ankle_roll_link_collision",
        "right_ankle_roll_link_collision",
    )

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
        primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ASIMOV_ACTION_SCALE

    cfg.viewer.body_name = "pelvis_link"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.8  # Asimov is shorter than G1

    # More conservative velocity commands due to:
    # 1. Narrow stance (11.3 cm hip width)
    # 2. Canted hip pitch (less stable)
    # 3. Limited ankle ROM
    twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

    # Remove base_lin_vel - not available on real robot IMU
    del cfg.observations["policy"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["base_lin_vel"]

    # Add gait clock observation for phase-aware locomotion
    gait_clock_cfg = ObservationTermCfg(
        func=mdp.gait_clock,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "gait_frequency": 1.25,  # Match imitation data frequency
        },
    )
    cfg.observations["policy"].terms["gait_clock"] = gait_clock_cfg
    cfg.observations["critic"].terms["gait_clock"] = gait_clock_cfg

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Larger variance for canted hip pitch (allows coupled roll/pitch motion)
        r".*hip_pitch.*": 0.5,
        # Moderate hip roll (asymmetric ranges)
        r".*hip_roll.*": 0.25,
        # Standard hip yaw
        r".*hip_yaw.*": 0.2,
        # Large knee variance (extends backwards, different from G1)
        r".*knee.*": 0.5,
        # Lower ankle variance due to limited ROM (±20° pitch, ±15° roll)
        r".*ankle_pitch.*": 0.2,
        r".*ankle_roll.*": 0.12,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Even larger variance for dynamic motion with canted hips
        r".*hip_pitch.*": 0.8,
        r".*hip_roll.*": 0.35,
        r".*hip_yaw.*": 0.3,
        r".*knee.*": 0.8,
        # Keep ankles constrained even when running
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.15,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_link",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Increase body angular velocity penalty (narrow stance = less stable)
    cfg.rewards["body_ang_vel"].weight = -0.08
    cfg.rewards["angular_momentum"].weight = -0.03
    # Enable air time reward for lighter robot (better for jumping)
    cfg.rewards["air_time"].weight = 0.5

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Imitation reward for walking reference motion
    cfg.rewards["imitation"] = RewardTermCfg(
        func=mdp.imitation_joint_pos,
        weight=1.5,
        params={
            "data_path": str(ASIMOV_WALKING_REFERENCE),
            "gait_frequency": 1.25,  # Match the reference data frequency
            "std": 0.5,  # Tolerance for joint position matching
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    # Alternating feet contact reward for proper bipedal gait
    cfg.rewards["alternating_feet"] = RewardTermCfg(
        func=mdp.alternating_feet_contact,
        weight=0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "command_threshold": 0.1,
            "gait_frequency": 1.25,  # Match imitation data frequency
        },
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def asimov_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Asimov flat terrain velocity tracking configuration."""
    cfg = asimov_rough_env_cfg(play=play)

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


def asimov_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Asimov balance-only config with reactive stepping.

    The robot learns to balance in place and take small corrective steps
    when disturbed. No velocity tracking - just stay upright.
    """
    cfg = asimov_flat_env_cfg(play=play)

    # Zero out velocity commands (not trying to go anywhere)
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

    # Disable velocity tracking rewards (not trying to go anywhere)
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_angular_velocity"].weight = 0.0

    # Keep foot rewards for reactive stepping
    # Keep upright, pose, body_ang_vel for balance

    return cfg
