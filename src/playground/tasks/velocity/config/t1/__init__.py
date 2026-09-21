from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from playground.rl.flashsac import VelocityOffPolicyRunner

from .env_cfgs import t1_flat_env_cfg, t1_rough_env_cfg, t1_flat_env_cfg_flashsac, t1_rough_env_cfg_flashsac
from .rl_cfg import t1_flashsac_runner_cfg, t1_ppo_runner_cfg

register_mjlab_task(
  task_id="Pumas-Velocity-Rough-Booster-T1-PPO",
  env_cfg=t1_rough_env_cfg(),
  play_env_cfg=t1_rough_env_cfg(play=True),
  rl_cfg=t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Flat-Booster-T1-PPO",
  env_cfg=t1_flat_env_cfg(),
  play_env_cfg=t1_flat_env_cfg(play=True),
  rl_cfg=t1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Rough-Booster-T1-FlashSAC",
  env_cfg=t1_rough_env_cfg_flashsac(),
  play_env_cfg=t1_rough_env_cfg_flashsac(play=True),
  rl_cfg=t1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Flat-Booster-T1-FlashSAC",
  env_cfg=t1_flat_env_cfg_flashsac(),
  play_env_cfg=t1_flat_env_cfg_flashsac(play=True),
  rl_cfg=t1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)
