from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from playground.rl.flashsac import VelocityOffPolicyRunner

from .env_cfgs import (
  g1_flat_env_cfg,
  g1_rough_env_cfg,
  g1_flat_env_cfg_flashsac,
  g1_rough_env_cfg_flashsac,
)
from .rl_cfg import g1_flashsac_runner_cfg, g1_ppo_runner_cfg

# Rough terrain needs a longer PPO run (mjlab's G1 rough config uses 30k).
_ROUGH_PPO_ITERATIONS = 30_000
_FLAT_PPO_ITERATIONS = 6_000

register_mjlab_task(
  task_id="Pumas-Velocity-Rough-Unitree-G1-PPO",
  env_cfg=g1_rough_env_cfg(),
  play_env_cfg=g1_rough_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(max_iterations=_ROUGH_PPO_ITERATIONS),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Flat-Unitree-G1-PPO",
  env_cfg=g1_flat_env_cfg(),
  play_env_cfg=g1_flat_env_cfg(play=True),
  rl_cfg=g1_ppo_runner_cfg(max_iterations=_FLAT_PPO_ITERATIONS),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Rough-Unitree-G1-FlashSAC",
  env_cfg=g1_rough_env_cfg_flashsac(),
  play_env_cfg=g1_rough_env_cfg_flashsac(play=True),
  rl_cfg=g1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Flat-Unitree-G1-FlashSAC",
  env_cfg=g1_flat_env_cfg_flashsac(),
  play_env_cfg=g1_flat_env_cfg_flashsac(play=True),
  rl_cfg=g1_flashsac_runner_cfg(),
  runner_cls=VelocityOffPolicyRunner,
)

# Opt-in variants. DA = left/right mirror data augmentation.
for _terrain, _env_cfg, _iterations in (
  ("Rough", g1_rough_env_cfg, _ROUGH_PPO_ITERATIONS),
  ("Flat", g1_flat_env_cfg, _FLAT_PPO_ITERATIONS),
):
  register_mjlab_task(
    task_id=f"Pumas-Velocity-{_terrain}-Unitree-G1-PPO-DA",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=g1_ppo_runner_cfg(symmetry=True, max_iterations=_iterations),
    runner_cls=VelocityOnPolicyRunner,
  )

for _terrain, _env_cfg in (
  ("Rough", g1_rough_env_cfg_flashsac),
  ("Flat", g1_flat_env_cfg_flashsac),
):
  register_mjlab_task(
    task_id=f"Pumas-Velocity-{_terrain}-Unitree-G1-FlashSAC-DA",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=g1_flashsac_runner_cfg(symmetry=True),
    runner_cls=VelocityOffPolicyRunner,
  )
