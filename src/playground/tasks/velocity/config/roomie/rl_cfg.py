"""RL configuration for Roomie V3 velocity task."""

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

# Left/right mirror used for data augmentation (see symmetry.py). Referenced by
# its "module:function" path so the runner config stays serializable.
_SYMMETRY_FUNC = "playground.tasks.velocity.config.roomie.symmetry:augment_symmetries"


def roomie_ppo_runner_cfg(symmetry: bool = False) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Roomie V3 velocity task.

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
    algorithm = RslRlSymPpoAlgorithmCfg(**algorithm_kwargs)
  else:
    algorithm = RslRlPpoAlgorithmCfg(**algorithm_kwargs)

  experiment_name = "roomie_velocity_da" if symmetry else "roomie_velocity"

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
    wandb_project="roomie_velocity",
    save_interval=1_000,
    num_steps_per_env=24,
    max_iterations=15_000,
  )
