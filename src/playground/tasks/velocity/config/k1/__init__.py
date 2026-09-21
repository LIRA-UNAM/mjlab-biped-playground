from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from playground.rl.flashsac import VelocityOffPolicyRunner

from .env_cfgs import (
  k1_flat_env_cfg,
  k1_rough_env_cfg,
  k1_flat_env_cfg_flashsac,
  k1_rough_env_cfg_flashsac,
)
from .rl_cfg import k1_flashsac_runner_cfg, k1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-K1-PPO",
  env_cfg=k1_rough_env_cfg(),
  play_env_cfg=k1_rough_env_cfg(play=True),
  rl_cfg=k1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-K1-PPO",
  env_cfg=k1_flat_env_cfg(),
  play_env_cfg=k1_flat_env_cfg(play=True),
  rl_cfg=k1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Booster-K1-FlashSAC",
  env_cfg=k1_rough_env_cfg_flashsac(),
  play_env_cfg=k1_rough_env_cfg_flashsac(play=True),
  rl_cfg=k1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Booster-K1-FlashSAC",
  env_cfg=k1_flat_env_cfg_flashsac(),
  play_env_cfg=k1_flat_env_cfg_flashsac(play=True),
  rl_cfg=k1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)

# Opt-in variants. DA = left/right mirror data augmentation.
for _terrain, _env_cfg in (("Rough", k1_rough_env_cfg), ("Flat", k1_flat_env_cfg)):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-{_terrain}-Booster-K1-PPO-DA",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=k1_ppo_runner_cfg(symmetry=True),
    runner_cls=VelocityOnPolicyRunner,
  )

for _terrain, _env_cfg in (
  ("Rough", k1_rough_env_cfg_flashsac),
  ("Flat", k1_flat_env_cfg_flashsac),
):
  register_mjlab_task(
    task_id=f"Mjlab-Velocity-{_terrain}-Booster-K1-FlashSAC-DA",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=k1_flashsac_runner_cfg(symmetry=True),
    runner_cls=VelocityOffPolicyRunner,
  )
