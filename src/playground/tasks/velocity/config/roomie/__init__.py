from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import roomie_flat_env_cfg, roomie_rough_env_cfg
from .rl_cfg import roomie_ppo_runner_cfg

register_mjlab_task(
  task_id="Pumas-Velocity-Rough-Roomie-V3-PPO",
  env_cfg=roomie_rough_env_cfg(),
  play_env_cfg=roomie_rough_env_cfg(play=True),
  rl_cfg=roomie_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pumas-Velocity-Flat-Roomie-V3-PPO",
  env_cfg=roomie_flat_env_cfg(),
  play_env_cfg=roomie_flat_env_cfg(play=True),
  rl_cfg=roomie_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Opt-in variants. DA = left/right mirror data augmentation.
for _terrain, _env_cfg in (("Rough", roomie_rough_env_cfg), ("Flat", roomie_flat_env_cfg)):
  register_mjlab_task(
    task_id=f"Pumas-Velocity-{_terrain}-Roomie-V3-PPO-DA",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=roomie_ppo_runner_cfg(symmetry=True),
    runner_cls=VelocityOnPolicyRunner,
  )
