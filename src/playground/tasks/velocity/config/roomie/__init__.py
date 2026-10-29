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

# Actuator stiffness sweep fitted to the left hip pitch step responses
# (step_response_fit, 2026-10-28): the robot matches kp x4.0 / kv x2.25 of the
# training gains best. Damping stays at the fitted x2.25; stiffness brackets it.
ROOMIE_KV_SCALE_FIT = 2.25
ROOMIE_KP_SCALE_SWEEP = (2.0, 3.0, 3.5, 4.0, 5.0, 6.0)
for _kp in ROOMIE_KP_SCALE_SWEEP:
  register_mjlab_task(
    task_id=f"Pumas-Velocity-Flat-Roomie-V3-PPO-Kp{_kp:g}",
    env_cfg=roomie_flat_env_cfg(kp_scale=_kp, kv_scale=ROOMIE_KV_SCALE_FIT),
    play_env_cfg=roomie_flat_env_cfg(play=True, kp_scale=_kp, kv_scale=ROOMIE_KV_SCALE_FIT),
    rl_cfg=roomie_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
