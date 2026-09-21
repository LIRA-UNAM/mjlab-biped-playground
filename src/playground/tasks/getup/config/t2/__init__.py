from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import booster_t2_getup_env_cfg
from .rl_cfg import booster_t2_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Pumas-Getup-Flat-Booster-T2",
  env_cfg=booster_t2_getup_env_cfg(),
  play_env_cfg=booster_t2_getup_env_cfg(play=True),
  rl_cfg=booster_t2_getup_ppo_runner_cfg(),
)
