"""RL configuration for Booster K1 velocity task."""

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


def k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster K1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Larger network than Asimov's: only legs are actuated, but
      # observations still carry full-body (arms/neck) proprioception.
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
    experiment_name="k1_velocity",
    wandb_project="mjlab_playground",
    save_interval=300,
    num_steps_per_env=24,
    max_iterations=3_000,
  )


def k1_flashsac_runner_cfg() -> RslRlFlashSacRunnerCfg:
  """Create FlashSAC (off-policy) RL runner configuration for Booster K1 velocity task."""
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
      replay_buffer_size=10_000_000,
      buffer_min_length=100_000,
      num_mini_batches=2,
      mini_batch_size=2048,
      n_steps=3,
      gamma=0.99,
      critic_target_update_tau=0.01,
      # See t1_flashsac_runner_cfg: interpreted in the tanh-normalized [-1, 1]
      # action space (identity action_bias/action_scale), not radians. Doubled
      # from the 0.15 default to sustain exploration past the point where it
      # was collapsing entropy too early on T1's equivalent config.
      temp_target_sigma=0.3,
    ),
    experiment_name="k1_velocity_flashsac",
    wandb_project="k1_velocity_flashsac",
    save_interval=3_000,
    num_steps_per_env=1,
    max_iterations=75_000,
  )
