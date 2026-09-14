"""RL configuration for Booster T1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from playground.rl.flashsac import (
  RslRlFlashSacActorCfg,
  RslRlFlashSacAlgorithmCfg,
  RslRlFlashSacCriticCfg,
  RslRlFlashSacRunnerCfg,
)


def t1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster T1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Larger network than Asimov's: only legs are actuated, but
      # observations still carry full-body (arms/waist/neck) proprioception.
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="t1_velocity",
    wandb_project="t1_velocity",
    save_interval=300,
    num_steps_per_env=24,
    max_iterations=3_000,
  )


def t1_flashsac_runner_cfg() -> RslRlFlashSacRunnerCfg:
  """Create FlashSAC (off-policy) RL runner configuration for Booster T1 velocity task."""
  return RslRlFlashSacRunnerCfg(
    actor=RslRlFlashSacActorCfg(
      num_blocks=2,
      hidden_dim=256,
    ),
    critic=RslRlFlashSacCriticCfg(
      num_blocks=2,
      hidden_dim=256,
    ),
    algorithm=RslRlFlashSacAlgorithmCfg(
      replay_buffer_size=1_000_000,
      buffer_min_length=100_000,
      num_mini_batches=2,
      mini_batch_size=2048,
      n_steps=3,
      gamma=0.99,
      critic_target_update_tau=0.01,
      # Interpreted in the tanh-normalized [-1, 1] action space (we use identity
      # action_bias/action_scale), not true radians, since the actor's affine
      # scaling buffers are left at identity. The paper's own 0.15 default
      # (insensitive across 0.05-0.25 in their setup) assumes radians via a
      # real action_bias/action_scale; in our normalized space that same value
      # let entropy collapse by ~step 15k, capping exploration to small
      # in-place oscillations that satisfy yaw tracking but not full
      # forward/lateral strides. Raised further to sustain exploration longer.
      temp_target_sigma=0.15,
    ),
    experiment_name="t1_velocity_flashsac",
    wandb_project="t1_velocity",
    save_interval=7_500,
    num_steps_per_env=1,
    max_iterations=75_000,
  )
