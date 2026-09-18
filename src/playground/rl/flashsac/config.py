"""Runner configuration for FlashSAC, mirroring mjlab.rl.config's plain-dataclass style
(FlashSAC's original config lives in isaaclab_flashsac.rl_cfg as isaaclab @configclass
dataclasses, which aren't available outside Isaac Lab)."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class RslRlFlashSacActorCfg:
  """Config for the FlashSAC actor network."""

  num_blocks: int = 2
  """Number of residual blocks in the actor trunk."""
  hidden_dim: int = 128
  """Hidden dimension of each residual block."""
  log_std_min: float = -10.0
  """Minimum log standard deviation of the Tanh-Gaussian policy."""
  log_std_max: float = 2.0
  """Maximum log standard deviation of the Tanh-Gaussian policy."""
  class_name: str = "FlashSACActor"
  """Actor class name, resolved within playground.rl.flashsac then rsl_rl."""


@dataclass
class RslRlFlashSacCriticCfg:
  """Config for the FlashSAC (distributional, double) critic network."""

  num_blocks: int = 2
  """Number of residual blocks in the critic trunk."""
  hidden_dim: int = 256
  """Hidden dimension of each residual block."""
  num_bins: int = 101
  """Number of bins for the categorical (C51-style) value distribution."""
  min_v: float = -5.0
  """Minimum value of the categorical value distribution support."""
  max_v: float = 5.0
  """Maximum value of the categorical value distribution support."""
  num_qs: int = 2
  """Number of critics in the ensemble."""
  class_name: str = "FlashSACCritic"
  """Critic class name, resolved within playground.rl.flashsac then rsl_rl."""


@dataclass
class RslRlFlashSacAlgorithmCfg:
  """Config for the FlashSAC algorithm."""

  replay_buffer_size: int = 1_000_000
  """Max replay buffer size in total transitions across all environments."""
  buffer_min_length: int = 10_000
  """Minimum number of transitions before updates start."""
  buffer_optimize_memory_usage: bool = True
  """Use the memory-efficient buffer (stores observations once, reconstructs
  next observations by index) instead of the flat TorchUniformBuffer."""
  buffer_device: str | None = None
  """Device for the buffer storage; None uses the training device."""
  buffer_obs_dtype: str | None = None
  """Optional torch dtype name for observation storage, e.g. "bfloat16"."""
  num_learning_epochs: int = 1
  """How many epochs to run each update."""
  num_mini_batches: int = 1
  """How many mini-batches (gradient updates) to run per epoch."""
  mini_batch_size: int = 2048
  """Mini-batch size for updates, drawn from the replay buffer."""
  learning_rate_init: float = 3e-4
  """Initial learning rate before warmup."""
  learning_rate_peak: float = 3e-4
  """Peak learning rate after warmup."""
  learning_rate_end: float = 1.5e-4
  """Final learning rate after cosine decay."""
  learning_rate_warmup_steps: int = 0
  """Number of warmup steps before the cosine decay starts."""
  learning_rate_decay_steps: int | None = None
  """Number of cosine-decay steps. If None, derived from
  ``max_iterations * num_learning_epochs * num_mini_batches``."""
  actor_bc_alpha: float = 0.0
  """Behavior-cloning regularization coefficient for the actor."""
  actor_noise_zeta_mu: float = 2.0
  """Mean of the zeta distribution used to sample exploration-noise repeat lengths."""
  actor_noise_zeta_max: int = 16
  """Maximum exploration-noise repeat length."""
  actor_update_period: int = 2
  """Number of critic updates between each actor update."""
  critic_target_update_tau: float = 0.01
  """EMA coefficient for the critic target network update."""
  temp_initial_value: float = 0.01
  """Initial value of the learnable temperature."""
  temp_target_sigma: float = 0.15
  """Target policy stochasticity used to derive the temperature loss."""
  temp_target_entropy: float | None = None
  """Target entropy for the temperature loss. If None, derived from action dim."""
  gamma: float = 0.99
  """The discount factor."""
  n_steps: int = 1
  """Number of steps for n-step return aggregation in the replay buffer."""
  normalize_reward: bool = True
  """Whether to apply running-return-variance-based reward normalization."""
  normalized_G_max: float = 5.0
  """Max normalized return magnitude; should match the critic's min_v/max_v support."""
  use_compile: bool = False
  """Whether to torch.compile the actor/critic forwards and EMA/weight-normalize
  functions. Defaults to False here: the upstream default of True targets a
  CUDA-graph workaround specific to running inside Isaac Sim, which mjlab does
  not use. Safe to enable per-run once verified to work under mjlab/mujoco_warp."""
  compile_mode: str = "auto"
  """torch.compile mode, only used when use_compile is True."""
  use_amp: bool = True
  """Whether to use fp16 automatic mixed precision for actor/critic updates."""
  symmetry_cfg: dict[str, Any] | None = None
  """Optional data-augmentation symmetry config, resolved by
  ``rsl_rl.extensions.resolve_symmetry_config`` (which injects the env under
  ``symmetry_cfg["env"]``) before algorithm construction. None disables it."""
  class_name: str = "FlashSAC"
  """Algorithm class name, resolved within playground.rl.flashsac then rsl_rl."""


@dataclass
class RslRlFlashSacRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration for training with FlashSAC.

  Subclasses ``RslRlBaseRunnerCfg`` directly (like ``RslRlOnPolicyRunnerCfg``
  does for PPO) since off-policy training needs the same shared fields:
  ``obs_groups`` (FlashSAC uses the same "actor"/"critic" convention),
  ``num_steps_per_env`` (here: env steps collected per off-policy iteration,
  typically 1), ``max_iterations``, ``logger``, ``wandb_project``,
  ``upload_model``, ``clip_actions``, and the resume/load fields.
  """

  class_name: str = "OffPolicyRunner"
  """Runner class name. Only informative here: the concrete runner class is
  selected via ``runner_cls`` in ``register_mjlab_task`` (see
  ``playground.rl.flashsac.runner.VelocityOffPolicyRunner``)."""
  actor: RslRlFlashSacActorCfg = field(default_factory=RslRlFlashSacActorCfg)
  """The actor configuration."""
  critic: RslRlFlashSacCriticCfg = field(default_factory=RslRlFlashSacCriticCfg)
  """The critic configuration."""
  algorithm: RslRlFlashSacAlgorithmCfg = field(
    default_factory=RslRlFlashSacAlgorithmCfg
  )
  """The algorithm configuration."""
  start_training: int = 0
  """Iteration at which learning updates start (before this, actions are
  sampled randomly to warm up the replay buffer; see
  ``FlashSAC.can_start_training``)."""
  log_interval: int = 20
  """Number of iterations between console/writer log updates. Off-policy
  iterations are short (typically one env step), so logging is windowed over
  this many iterations to keep output and writer traffic sane."""
