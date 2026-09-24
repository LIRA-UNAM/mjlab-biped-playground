"""The K1 env_cfgs.py keeps a verbatim copy of mjlab's ``make_velocity_env_cfg``.

These tests fail when mjlab changes its factory (for example after a version
bump). Re-sync the copy in ``env_cfgs.py`` with mjlab's ``velocity_env_cfg.py``
and update the K1 overrides if needed.
"""

import dataclasses
import difflib
import enum
import inspect

from mjlab.tasks.velocity import velocity_env_cfg as mjlab_velocity_env_cfg
from playground.tasks.velocity.config.k1 import env_cfgs as k1_env_cfgs


def _canonical(obj):
  """Structure that compares by value.

  ``==`` on the cfg dataclasses is not enough: mjlab's ``EntityCfg`` creates a
  fresh lambda per instance, so two cfgs built by the very same factory never
  compare equal. Callables are therefore compared by qualified name, and dict
  order is kept because manager term order matters.
  """
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return {
      "__type__": type(obj).__qualname__,
      **{f.name: _canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
    }
  if isinstance(obj, dict):
    return ("dict", [(repr(k), _canonical(v)) for k, v in obj.items()])
  if isinstance(obj, (list, tuple)):
    return ("seq", [_canonical(v) for v in obj])
  if isinstance(obj, enum.Enum):
    return ("enum", str(obj))
  if callable(obj):
    return (
      "callable",
      getattr(obj, "__module__", "?"),
      getattr(obj, "__qualname__", repr(obj)),
    )
  return obj


def test_factory_source_matches_mjlab():
  local = inspect.getsource(k1_env_cfgs.make_velocity_env_cfg)
  upstream = inspect.getsource(mjlab_velocity_env_cfg.make_velocity_env_cfg)
  diff = "\n".join(
    difflib.unified_diff(
      upstream.splitlines(),
      local.splitlines(),
      "mjlab",
      "k1 env_cfgs",
      lineterm="",
      n=1,
    )
  )
  assert local == upstream, f"make_velocity_env_cfg differs from mjlab's:\n{diff}"


def test_factory_values_match_mjlab():
  local = _canonical(k1_env_cfgs.make_velocity_env_cfg())
  upstream = _canonical(mjlab_velocity_env_cfg.make_velocity_env_cfg())
  assert local == upstream
