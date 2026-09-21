"""PPO algorithm configs for options mjlab's own config does not expose."""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.rl.config import RslRlPpoAlgorithmCfg as MjlabRslRlPpoAlgorithmCfg


@dataclass
class RslRlSymmetryCfg:
  """Left/right symmetry configuration, passed through to rsl-rl's ``Symmetry``.

  ``data_augmentation_func`` is called as ``func(env=, obs=, actions=)`` and must
  return ``(obs, actions)`` with the mirrored samples appended after the
  originals (``[original; mirrored]``). Either input may be ``None``, in which
  case the matching output is ``None``.
  """

  use_data_augmentation: bool = False
  """Append mirrored samples to every mini-batch."""

  use_mirror_loss: bool = False
  """Add an auxiliary loss penalizing the policy for disagreeing with itself on
  mirrored observations."""

  data_augmentation_func: Callable | str | None = None
  """The mirror function, or its ``"module:function"`` path."""

  mirror_loss_coeff: float = 0.0
  """Weight of the mirror loss. Only used when ``use_mirror_loss`` is True."""


@dataclass
class RslRlPpoAlgorithmCfg(MjlabRslRlPpoAlgorithmCfg):
  """mjlab's PPO config plus an optional symmetry config."""

  symmetry_cfg: RslRlSymmetryCfg | None = None


@dataclass
class RslRlMuonPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO with hybrid Muon (matrices) / Adam (everything else) updates."""

  class_name: str = "playground.rl.muon:MuonPPO"
  muon_weight_decay: float = 0.0
  muon_momentum: float = 0.95
  muon_ns_steps: int = 5
