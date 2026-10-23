"""SteadyWin driver cascade (position loop -> speed loop) as an mjlab actuator.

The Roomie V3 GIM8108-36 / GIM6010-36 drivers do not apply a PD torque law. Each
runs a position loop that produces a speed command, capped at the controller's
position_max_speed_rpm, and a speed loop that turns the speed error into current.
In torque-equivalent units (Roomie V3 model_config.json, profile full_body,
driver_cascade.parameter_basis):

  e      = q_target - q
  v_ref  = clip(kp/kv * e + ki/kv * integral(e), -v_max, v_max)   position (PI) loop
  v_ref  = first-order filter of v_ref (velocity_reference_timeconstant, 0 = none)
  tau    = clip(kv * (v_ref - q_dot), -tau_max, tau_max)            speed (P) loop

With ki = 0 and no saturation this is exactly the PD law tau = kp e - kv q_dot.
The integral only accumulates while neither the speed command nor the torque is
saturated in the direction of the error (conditional anti-windup), and it is
cleared on reset.

tau_max follows the motor torque-speed curve: the full effort_limit while
braking (torque against the joint velocity) or below knee_point_velocity, then
falling linearly to 0 at no_load_velocity while motoring.

Gearbox backlash: the driver's encoder sits on the rotor side, so q is replaced
by a play operator of the joint position, q_m = clamp(q_m, q - b/2, q + b/2),
with b the per-joint backlash (0 by default; set per episode by
randomize_steadywin_cascade). The gap starts at a random point of the band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

RPM_TO_RAD_S = 2.0 * math.pi / 60.0


@dataclass(kw_only=True)
class SteadywinCascadeActuatorCfg(IdealPdActuatorCfg):
  """Driver cascade. stiffness = kp (N m/rad), damping = kv (N m s/rad)."""

  integral_gain: float = 0.0
  """ki (N m/(rad s)) of the position loop; 0 = proportional position loop."""
  velocity_limit: float = 50.0 * RPM_TO_RAD_S
  """Speed command cap (rad/s): roomie_position_control_node position_max_speed_rpm."""
  velocity_reference_timeconstant: float = 0.0
  """First-order filter on the speed command (s); 0 = none."""
  no_load_velocity: float | None = None
  """Motor no-load speed (rad/s); None = flat torque limit."""
  knee_point_velocity: float | None = None
  """Start of the linear torque derating (rad/s); None = no_load_velocity."""

  def __post_init__(self) -> None:
    super().__post_init__()
    assert self.damping > 0.0, "the speed loop needs a positive kv (damping)."
    assert self.integral_gain >= 0.0, "integral_gain must be non-negative."
    assert self.velocity_limit > 0.0, "velocity_limit must be positive."
    assert self.velocity_reference_timeconstant >= 0.0
    if self.no_load_velocity is not None:
      assert self.no_load_velocity > 0.0, "no_load_velocity must be positive."
      knee = self.knee_point_velocity
      assert knee is None or 0.0 <= knee < self.no_load_velocity, (
        "knee_point_velocity must be in [0, no_load_velocity)."
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> SteadywinCascadeActuator:
    return SteadywinCascadeActuator(self, entity, target_ids, target_names)


class SteadywinCascadeActuator(IdealPdActuator[SteadywinCascadeActuatorCfg]):
  """Stateful cascade (overrides compute, so it is never fused).

  stiffness, damping and force_limit stay the IdealPd tensors, so dr.pd_gains
  and dr.effort_limits randomize kp, kv and the torque limit (the torque-speed
  curve scales with it) as for an ideal PD actuator.
  """

  def __init__(
    self,
    cfg: SteadywinCascadeActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.integral_gain: torch.Tensor | None = None
    self.velocity_limit: torch.Tensor | None = None
    self._integral: torch.Tensor | None = None
    self._velocity_reference: torch.Tensor | None = None
    self._pending_error: torch.Tensor | None = None
    self._integrate: torch.Tensor | None = None
    self.backlash: torch.Tensor | None = None
    self._motor_pos: torch.Tensor | None = None
    self._motor_pos_valid: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    shape = (data.nworld, len(self._target_names))
    self.integral_gain = torch.full(shape, self.cfg.integral_gain, device=device)
    self.velocity_limit = torch.full(shape, self.cfg.velocity_limit, device=device)
    self._integral = torch.zeros(shape, device=device)
    self._velocity_reference = torch.zeros(shape, device=device)
    self._pending_error = torch.zeros(shape, device=device)
    self._integrate = torch.zeros(shape, dtype=torch.bool, device=device)
    self.backlash = torch.zeros(shape, device=device)
    self._motor_pos = torch.zeros(shape, device=device)
    self._motor_pos_valid = torch.zeros(data.nworld, dtype=torch.bool, device=device)

  def _torque_limit(self, torque: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    """Available torque magnitude at this velocity, in the torque's direction."""
    assert self.force_limit is not None
    v0 = self.cfg.no_load_velocity
    if v0 is None:
      return self.force_limit
    knee = self.cfg.knee_point_velocity
    if knee is None:
      fraction = (vel.abs() < v0).to(vel.dtype)
    else:
      fraction = torch.clamp((v0 - vel.abs()) / (v0 - knee), 0.0, 1.0)
    motoring = torque * vel > 0.0
    return torch.where(motoring, self.force_limit * fraction, self.force_limit)

  def _driver_position(self, pos: torch.Tensor) -> torch.Tensor:
    """Rotor-side position seen by the driver (play operator of the joint)."""
    assert self.backlash is not None and self._motor_pos is not None
    assert self._motor_pos_valid is not None
    half = 0.5 * self.backlash
    # First step after a reset: start at a random point of the gap (branch-free,
    # no host sync).
    start = pos + half * (2.0 * torch.rand_like(pos) - 1.0)
    self._motor_pos = torch.where(self._motor_pos_valid[:, None], self._motor_pos, start)
    self._motor_pos_valid[:] = True
    self._motor_pos = torch.minimum(torch.maximum(self._motor_pos, pos - half), pos + half)
    return self._motor_pos

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None and self.damping is not None
    assert self.force_limit is not None and self.integral_gain is not None
    assert self.velocity_limit is not None and self._integral is not None
    assert self._velocity_reference is not None
    error = cmd.position_target - self._driver_position(cmd.pos)
    raw_speed = (self.stiffness * error + self.integral_gain * self._integral) / self.damping
    speed = torch.clamp(raw_speed, -self.velocity_limit, self.velocity_limit)
    tau_s = self.cfg.velocity_reference_timeconstant
    if tau_s > 0.0:
      # One physics step of a first-order lag (dt from the last update call).
      alpha = 1.0 - math.exp(-self._dt / tau_s) if self._dt > 0.0 else 1.0
      self._velocity_reference += alpha * (speed - self._velocity_reference)
      speed = self._velocity_reference
    raw_torque = self.damping * (speed - cmd.vel) + cmd.effort_target
    torque_limit = self._torque_limit(raw_torque, cmd.vel)
    torque = torch.clamp(raw_torque, -torque_limit, torque_limit)
    # Conditional anti-windup: integrate unless a saturation pushes the same way
    # as the error.
    speed_sat = (raw_speed.abs() >= self.velocity_limit) & (raw_speed * error > 0.0)
    torque_sat = (raw_torque.abs() >= torque_limit) & (raw_torque * error > 0.0)
    self._pending_error = error
    self._integrate = ~(speed_sat | torque_sat)
    return torque

  _dt: float = 0.0

  def update(self, dt: float) -> None:
    super().update(dt)
    self._dt = dt
    assert self._integral is not None and self._pending_error is not None
    assert self._integrate is not None
    self._integral += torch.where(
      self._integrate, self._pending_error * dt, torch.zeros_like(self._pending_error)
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self._integral is None:
      return
    ids = slice(None) if env_ids is None else env_ids
    self._integral[ids] = 0.0
    assert self._velocity_reference is not None and self._pending_error is not None
    self._velocity_reference[ids] = 0.0
    self._pending_error[ids] = 0.0
    assert self._motor_pos_valid is not None
    self._motor_pos_valid[ids] = False
