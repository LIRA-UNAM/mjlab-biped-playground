"""RL configuration for Booster T1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from playground.rl.config import (
  RslRlSymmetryCfg,
)
from playground.rl.config import (
  RslRlPpoAlgorithmCfg as RslRlSymPpoAlgorithmCfg,
)
from playground.rl.flashsac import (
  RslRlFlashSacActorCfg,
  RslRlFlashSacAlgorithmCfg,
  RslRlFlashSacCriticCfg,
  RslRlFlashSacRunnerCfg,
)

# Left/right mirror used for data augmentation (see symmetry.py). Referenced by
# its "module:function" path so the runner config stays serializable.
_SYMMETRY_FUNC = "playground.tasks.velocity.config.t1.symmetry:augment_symmetries"


def t1_ppo_runner_cfg(symmetry: bool = False) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Booster T1 velocity task.

  Args:
    symmetry: Augment every PPO mini-batch with its left/right mirror.
  """
  algorithm_kwargs: dict = dict(
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
  )
  if symmetry:
    algorithm_kwargs["symmetry_cfg"] = RslRlSymmetryCfg(
      use_data_augmentation=True,
      data_augmentation_func=_SYMMETRY_FUNC,
    )
  if symmetry:
    algorithm = RslRlSymPpoAlgorithmCfg(**algorithm_kwargs)
  else:
    algorithm = RslRlPpoAlgorithmCfg(**algorithm_kwargs)

  experiment_name = "t1_velocity_da" if symmetry else "t1_velocity"

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Larger network than Asimov's: legs, waist, and arms are actuated, and
      # observations carry full-body (incl. neck) proprioception.
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
    algorithm=algorithm,
    experiment_name=experiment_name,
    wandb_project="t1_velocity",
    save_interval=2_000,
    num_steps_per_env=24,
    # Command curriculum stages switch at iterations 5k and 10k (see env_cfgs).
    max_iterations=15_000,
  )


def t1_flashsac_runner_cfg(symmetry: bool = False) -> RslRlFlashSacRunnerCfg:
  """Create FlashSAC (off-policy) RL runner configuration for Booster T1 velocity task.

  Args:
    symmetry: Augment every replay mini-batch with its left/right mirror.
  """
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
      num_mini_batches=8,
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
      symmetry_cfg=(
        {
          "data_augmentation_func": _SYMMETRY_FUNC,
          "use_data_augmentation": True,
          "use_mirror_loss": False,
        }
        if symmetry
        else None
      ),
    ),
    experiment_name="t1_velocity_flashsac_da" if symmetry else "t1_velocity_flashsac",
    wandb_project="t1_velocity",
    save_interval=2_000,
    num_steps_per_env=1,
    max_iterations=15_000,
  )
