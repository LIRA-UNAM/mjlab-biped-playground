"""Left/right mirror symmetry for the K1 velocity tasks.

The mirror is a signed permutation of every observation group and of the
action vector, derived from the live observation layout (term names and
dimensions) rather than hard-coded offsets. An unknown observation term raises
instead of being silently left unmirrored.

Mirroring is across the robot's sagittal plane (y -> -y in the base frame), so:

* body-frame vectors flip their y component, angular velocities flip x and z;
* left/right joints swap, and joints whose axis is not in the sagittal plane
  (roll and yaw) flip sign. Pitch joints and the parallel-ankle cranks (both
  along +y) keep their sign;
* the velocity command flips vy and wz;
* per-foot quantities swap feet, and the ``height_scan`` grid reverses in y.

``foot_contact_forces`` is the one approximation: the contact sensor reports it
in the world frame, so flipping y is exact only when the base heading is along
world x. The vertical component and the foot swap are exact. The term only
feeds the privileged critic.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .env_cfgs import OBS_JOINT_NAMES

_FLIP_Y = (1.0, -1.0, 1.0)
_FLIP_XZ = (-1.0, 1.0, -1.0)
_COMMAND_SIGN = (1.0, -1.0, -1.0)  # (vx, vy, wz)


@dataclass
class MirrorContext:
  """Everything needed to mirror the observation terms of the K1 velocity task."""

  obs_joint_names: Sequence[str]
  action_joint_names: Sequence[str]
  height_scan_perm: Sequence[int] | None = None


def mirror_joint_name(name: str) -> str:
  """Return the name of the joint on the opposite side (head joints map to themselves)."""
  if name.startswith("Left_"):
    return "Right_" + name[len("Left_") :]
  if name.startswith("Right_"):
    return "Left_" + name[len("Right_") :]
  return name


def joint_mirror(names: Sequence[str]) -> tuple[list[int], list[float]]:
  """Signed permutation mirroring a vector ordered like ``names``.

  ``mirrored[i] = sign[i] * x[perm[i]]``. Roll and yaw joints change sign; pitch
  joints and the parallel-ankle cranks (A/B) do not.
  """
  index = {name: i for i, name in enumerate(names)}
  perm: list[int] = []
  sign: list[float] = []
  for name in names:
    other = mirror_joint_name(name)
    if other not in index:
      raise ValueError(f"Joint {name!r} has no mirror {other!r} in the joint list.")
    perm.append(index[other])
    sign.append(-1.0 if name.endswith(("_Roll", "_Yaw")) else 1.0)
  return perm, sign


def grid_mirror_perm(offsets_xy: torch.Tensor) -> list[int]:
  """Permutation reversing y of a ray grid: ``mirrored[i] = x[perm[i]]``."""
  mirrored = offsets_xy * offsets_xy.new_tensor([1.0, -1.0])
  # The default matmul-based mode is too imprecise for near-zero distances.
  dist = torch.cdist(mirrored, offsets_xy, compute_mode="donot_use_mm_for_euclid_dist")
  nearest = dist.min(dim=1)
  if nearest.values.max() > 1e-4:
    raise ValueError("Ray grid is not symmetric in y.")
  return nearest.indices.tolist()


def _term_mirror(
  name: str, dim: int, ctx: MirrorContext
) -> tuple[list[int], list[float]]:
  """Signed permutation for a single observation term."""

  def vector(sign: Sequence[float]) -> tuple[list[int], list[float]]:
    return list(range(3)), list(sign)

  if name == "base_lin_vel" or name == "projected_gravity":
    perm, sign = vector(_FLIP_Y)
  elif name == "base_ang_vel":
    perm, sign = vector(_FLIP_XZ)
  elif name == "command":
    perm, sign = vector(_COMMAND_SIGN)
  elif name in ("joint_pos", "joint_vel"):
    perm, sign = joint_mirror(ctx.obs_joint_names)
  elif name == "actions":
    perm, sign = joint_mirror(ctx.action_joint_names)
  elif name == "height_scan":
    if ctx.height_scan_perm is None:
      raise ValueError("height_scan observed but no ray grid permutation was given.")
    perm, sign = list(ctx.height_scan_perm), [1.0] * dim
  elif name in ("foot_height", "foot_air_time", "foot_contact"):
    perm, sign = [1, 0], [1.0, 1.0]
  elif name == "foot_contact_forces":
    perm, sign = [3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
  else:
    raise ValueError(f"No mirror rule for observation term {name!r}.")
  if len(perm) != dim:
    raise ValueError(
      f"Observation term {name!r} has dim {dim}, but its mirror rule expects {len(perm)}."
    )
  return perm, sign


def build_group_mirror(
  terms: Sequence[tuple[str, int]], ctx: MirrorContext
) -> tuple[torch.Tensor, torch.Tensor]:
  """Signed permutation for a concatenated observation group.

  Args:
    terms: ``(name, dim)`` of each term, in concatenation order.

  Returns:
    ``(perm, sign)`` such that ``mirrored = obs[:, perm] * sign``.
  """
  perm: list[int] = []
  sign: list[float] = []
  offset = 0
  for name, dim in terms:
    term_perm, term_sign = _term_mirror(name, dim, ctx)
    perm += [offset + p for p in term_perm]
    sign += term_sign
    offset += dim
  return torch.tensor(perm, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


def build_action_mirror(ctx: MirrorContext) -> tuple[torch.Tensor, torch.Tensor]:
  perm, sign = joint_mirror(ctx.action_joint_names)
  return torch.tensor(perm, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


@dataclass
class _EnvMirror:
  groups: dict[str, tuple[torch.Tensor, torch.Tensor]]
  action: tuple[torch.Tensor, torch.Tensor]


_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _env_mirror(env: VecEnv) -> _EnvMirror:
  """Build (and cache) the mirror maps from the environment's live layout."""
  unwrapped = env.unwrapped  # type: ignore[attr-defined]
  cached = _CACHE.get(unwrapped)
  if cached is not None:
    return cached

  action_term = unwrapped.action_manager.get_term("joint_pos")
  height_scan_perm = None
  if "terrain_scan" in unwrapped.scene.sensors:
    pattern = unwrapped.scene["terrain_scan"].cfg.pattern
    offsets, _ = pattern.generate_rays(None, "cpu")
    height_scan_perm = grid_mirror_perm(offsets[:, :2])
  ctx = MirrorContext(
    obs_joint_names=OBS_JOINT_NAMES,
    action_joint_names=tuple(action_term.target_names),
    height_scan_perm=height_scan_perm,
  )

  om = unwrapped.observation_manager
  groups = {
    group: build_group_mirror(
      [
        (name, int(dim[-1]) if len(dim) == 1 else int(torch.Size(dim).numel()))
        for name, dim in zip(
          om.active_terms[group], om.group_obs_term_dim[group], strict=True
        )
      ],
      ctx,
    )
    for group in om.active_terms
  }
  mirror = _EnvMirror(groups=groups, action=build_action_mirror(ctx))
  _CACHE[unwrapped] = mirror
  return mirror


def augment_symmetries(
  env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append the left/right mirror of ``obs`` and ``actions`` after the originals.

  Follows the contract of ``rsl_rl.extensions.Symmetry``: outputs have twice the
  batch size, laid out as ``[original; mirrored]``.
  """
  mirror = _env_mirror(env)

  if obs is not None:
    augmented = {}
    for group in obs.keys():
      if group not in mirror.groups:
        raise ValueError(f"No mirror map for observation group {group!r}.")
      perm, sign = mirror.groups[group]
      x = obs[group]
      perm, sign = perm.to(x.device), sign.to(x.device, x.dtype)
      if x.shape[-1] != perm.numel():
        raise ValueError(
          f"Observation group {group!r} has dim {x.shape[-1]}, expected {perm.numel()}."
        )
      augmented[group] = torch.cat([x, x[:, perm] * sign], dim=0)
    obs = TensorDict(augmented, batch_size=(2 * obs.batch_size[0],))

  if actions is not None:
    perm, sign = mirror.action
    perm, sign = perm.to(actions.device), sign.to(actions.device, actions.dtype)
    actions = torch.cat([actions, actions[:, perm] * sign], dim=0)

  return obs, actions
