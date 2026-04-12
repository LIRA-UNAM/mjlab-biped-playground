"""Asimov velocity task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import asimov_balance_env_cfg, asimov_flat_env_cfg, asimov_rough_env_cfg
from .rl_cfg import asimov_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Asimov",
    env_cfg=asimov_rough_env_cfg(),
    play_env_cfg=asimov_rough_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Asimov",
    env_cfg=asimov_flat_env_cfg(),
    play_env_cfg=asimov_flat_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Balance-Asimov",
    env_cfg=asimov_balance_env_cfg(),
    play_env_cfg=asimov_balance_env_cfg(play=True),
    rl_cfg=asimov_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
