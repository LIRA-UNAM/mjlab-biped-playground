from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import t1_flat_env_cfg, t1_rough_env_cfg
from .rl_cfg import t1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-T1",
  env_cfg=t1_rough_env_cfg(),
  play_env_cfg=t1_rough_env_cfg(play=True),
  rl_cfg=t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-T1",
  env_cfg=t1_flat_env_cfg(),
  play_env_cfg=t1_flat_env_cfg(play=True),
  rl_cfg=t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
