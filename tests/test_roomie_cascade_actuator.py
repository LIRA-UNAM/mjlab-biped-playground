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
              n=1, v0=None, knee=None, backlash=0.0):
  cfg = SteadywinCascadeActuatorCfg(
    target_names_expr=("j",), stiffness=kp, damping=kv, integral_gain=ki,
    effort_limit=effort, velocity_limit=vmax, velocity_reference_timeconstant=tau,
    no_load_velocity=v0, knee_point_velocity=knee)
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
  act.backlash = torch.full(shape, backlash)
  act._motor_pos = torch.zeros(shape)
  act._motor_pos_valid = torch.zeros(1, dtype=torch.bool)
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


# GIM8108-36 curve: 100 N m up to the knee, linear to 0 at the no-load speed.
V0, KNEE = 7.016, 4.503


def test_full_torque_below_the_knee():
  act = _actuator(kp=2000.0, kv=100.0, v0=V0, knee=KNEE)
  # Large positive speed error while moving forward below the knee: motoring.
  assert act.compute(_cmd(1.0, 0.0, 4.0)).item() == pytest.approx(100.0)


def test_torque_is_derated_at_the_speed_cap():
  act = _actuator(kp=20000.0, kv=1000.0, vmax=100.0, v0=V0, knee=KNEE)
  vel = 50 * 2 * math.pi / 60
  expected = 100.0 * (V0 - vel) / (V0 - KNEE)
  assert act.compute(_cmd(1.0, 0.0, vel)).item() == pytest.approx(expected, rel=1e-5)
  # No torque left at the no-load speed.
  assert act.compute(_cmd(1.0, 0.0, V0 + 0.1)).item() == pytest.approx(0.0, abs=1e-6)


def test_braking_keeps_the_full_torque():
  act = _actuator(kp=320.0, kv=1000.0, vmax=100.0, v0=V0, knee=KNEE)
  # Torque against the motion (target behind, moving forward fast).
  assert act.compute(_cmd(-1.0, 0.0, 6.5)).item() == pytest.approx(-100.0)


def test_no_windup_while_the_derated_torque_is_saturated():
  act = _actuator(kp=320.0, kv=16.0, ki=200.0, vmax=100.0, v0=V0, knee=KNEE)
  # Motoring at 6 rad/s leaves 100 * (V0 - 6) / (V0 - KNEE) ~ 40 N m; the speed
  # loop asks for 16 * (20 * 0.5 - 6) = 64 N m, so the torque is saturated.
  for _ in range(50):
    act.compute(_cmd(0.5, 0.0, 6.0))
    act.update(DT)
  assert act._integral.abs().max().item() == 0.0


def test_backlash_hides_motion_inside_the_band():
  b = math.radians(0.25)
  act = _actuator(kp=533.3, kv=16.0, backlash=b)
  act._motor_pos_valid[:] = True  # gap centred: driver position 0
  # Moving the joint by less than half the gap does not move the driver's reading.
  assert act.compute(_cmd(0.0, 0.4 * b, 0.0)).item() == pytest.approx(0.0, abs=1e-9)
  # Past the band the reading lags the joint by b/2.
  tau = act.compute(_cmd(0.0, 2.0 * b, 0.0)).item()
  assert tau == pytest.approx(-533.3 * 1.5 * b, rel=1e-5)


def test_reset_restarts_the_gap_inside_the_band():
  b = 0.004
  act = _actuator(backlash=b)
  for _ in range(20):
    act.reset()
    act.compute(_cmd(0.0, 0.3, 0.0))
    assert abs(act._motor_pos.item() - 0.3) <= 0.5 * b + 1e-9


def test_randomization_event_writes_only_the_reset_envs():
  from types import SimpleNamespace

  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from playground.tasks.velocity.mdp.events import randomize_steadywin_cascade

  act = _actuator(ki=200.0, n=3)
  shape = (4, 3)
  act.integral_gain = torch.full(shape, 200.0)
  act.backlash = torch.zeros(shape)
  env = SimpleNamespace(
    num_envs=4, device="cpu",
    scene={"robot": SimpleNamespace(actuators=[act])})
  randomize_steadywin_cascade(
    env, torch.tensor([1, 3]), integral_gain_range=(0.5, 1.5),
    backlash_range=(0.0, 0.004363), asset_cfg=SceneEntityCfg("robot"))
  for untouched in (0, 2):
    assert torch.all(act.integral_gain[untouched] == 200.0)
    assert torch.all(act.backlash[untouched] == 0.0)
  for reset in (1, 3):
    assert torch.all((act.integral_gain[reset] >= 100.0) & (act.integral_gain[reset] <= 300.0))
    assert torch.all((act.backlash[reset] >= 0.0) & (act.backlash[reset] <= 0.004363))
