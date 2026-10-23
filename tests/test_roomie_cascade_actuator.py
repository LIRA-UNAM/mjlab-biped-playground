"""Tests for the SteadyWin driver cascade actuator used by the Roomie V3 task."""

import math

import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd
from playground.asset_zoo.robots.roomie.actuators import (
  SteadywinCascadeActuator,
  SteadywinCascadeActuatorCfg,
)

DT = 0.005


def _actuator(kp=320.0, kv=16.0, ki=0.0, effort=100.0, vmax=50 * 2 * math.pi / 60, tau=0.0,
              n=1):
  cfg = SteadywinCascadeActuatorCfg(
    target_names_expr=("j",), stiffness=kp, damping=kv, integral_gain=ki,
    effort_limit=effort, velocity_limit=vmax, velocity_reference_timeconstant=tau)
  act = SteadywinCascadeActuator(cfg, entity=None, target_ids=list(range(n)),
                                 target_names=[f"j{i}" for i in range(n)])
  shape = (1, n)
  act.stiffness = torch.full(shape, kp)
  act.damping = torch.full(shape, kv)
  act.force_limit = torch.full(shape, effort)
  act.integral_gain = torch.full(shape, ki)
  act.velocity_limit = torch.full(shape, vmax)
  act._integral = torch.zeros(shape)
  act._velocity_reference = torch.zeros(shape)
  act._pending_error = torch.zeros(shape)
  act._integrate = torch.zeros(shape, dtype=torch.bool)
  return act


def _cmd(target, pos, vel, effort=0.0):
  t = lambda v: torch.tensor([[float(v)]])  # noqa: E731
  return ActuatorCmd(position_target=t(target), velocity_target=t(0.0),
                     effort_target=t(effort), pos=t(pos), vel=t(vel))


def test_unsaturated_proportional_cascade_is_pd():
  act = _actuator(kp=533.3, kv=16.0)
  tau = act.compute(_cmd(0.01, 0.0, 0.1))
  assert tau.item() == pytest.approx(533.3 * 0.01 - 16.0 * 0.1, rel=1e-6)


def test_speed_command_is_capped():
  act = _actuator(kp=320.0, kv=16.0)
  # kp/kv * e = 20 * 1.0 = 20 rad/s > 5.236 rad/s cap.
  tau = act.compute(_cmd(1.0, 0.0, 0.0))
  assert tau.item() == pytest.approx(16.0 * 50 * 2 * math.pi / 60, rel=1e-6)


def test_torque_is_limited():
  act = _actuator(kp=560.0, kv=100.0, effort=52.48)
  assert act.compute(_cmd(0.0, 0.0, 5.0)).item() == pytest.approx(-52.48)


def test_integral_removes_a_steady_error_and_resets():
  act = _actuator(kp=320.0, kv=16.0, ki=200.0)
  e = 0.02
  first = act.compute(_cmd(e, 0.0, 0.0)).item()
  for _ in range(200):  # 1 s at a constant error
    act.compute(_cmd(e, 0.0, 0.0))
    act.update(DT)
  later = act.compute(_cmd(e, 0.0, 0.0)).item()
  assert first == pytest.approx(320.0 * e)
  # The integral term has added ki * e * t on top of the proportional torque.
  assert later == pytest.approx(320.0 * e + 200.0 * e * 1.0, rel=0.02)
  act.reset()
  assert act.compute(_cmd(e, 0.0, 0.0)).item() == pytest.approx(320.0 * e)


def test_no_windup_while_the_speed_command_is_saturated():
  act = _actuator(kp=320.0, kv=16.0, ki=200.0)
  for _ in range(100):  # large error: the speed command stays capped
    act.compute(_cmd(1.0, 0.0, 0.0))
    act.update(DT)
  assert act._integral.abs().max().item() == 0.0


def test_speed_reference_filter_lags_the_command():
  act = _actuator(kp=55.36, kv=5.178, tau=0.02)
  act.update(DT)  # sets dt
  tau1 = act.compute(_cmd(0.05, 0.0, 0.0)).item()
  unfiltered = 55.36 * 0.05
  alpha = 1.0 - math.exp(-DT / 0.02)
  assert tau1 == pytest.approx(alpha * unfiltered, rel=1e-5)
