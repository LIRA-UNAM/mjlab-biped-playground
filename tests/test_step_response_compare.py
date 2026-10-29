"""Tests for the Roomie step response comparison tool."""

import math

import numpy as np
import pytest
from playground.scripts.step_response_compare import (
  HangingRoomie,
  compare,
  simulate_segment,
  step_metrics,
)

SLOTS = [f"{side}_{axis}" for side in ("left", "right") for axis in (
  "hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")]


def test_metrics_of_a_first_order_response():
  tau, t = 0.05, np.arange(0.0, 1.0, 0.001)
  pos = 1.0 + 0.2 * (1.0 - np.exp(-t / tau))
  vel = 0.2 / tau * np.exp(-t / tau)
  m = step_metrics(t, pos, vel, start=1.0, goal=1.2)
  assert m.rise_time == pytest.approx(tau * math.log(9.0), abs=2e-3)
  assert m.delay == pytest.approx(tau * -math.log(0.95), abs=2e-3)
  assert m.overshoot == 0.0
  assert m.settling_time == pytest.approx(tau * math.log(20.0), abs=2e-3)
  assert m.peak_speed == pytest.approx(4.0, rel=1e-3)
  assert abs(m.final_error) < 1e-6


def test_metrics_report_overshoot():
  t = np.arange(0.0, 1.0, 0.001)
  pos = 1.0 - np.exp(-8 * t) * np.cos(30 * t)  # underdamped, start 0 goal 1
  m = step_metrics(t, pos, np.gradient(pos, t), start=0.0, goal=1.0)
  assert m.overshoot > 10.0


@pytest.fixture(scope="module")
def robot():
  return HangingRoomie()


def test_simulated_knee_step_reaches_the_goal(robot):
  start = {"left_hip_pitch": -0.35, "left_knee_pitch": 0.7}
  t, pos, vel = simulate_segment(
    robot, "left_knee_pitch", start, np.array([0.0, 0.5]), np.array([0.7, 0.75]), 0.3, 1.5)
  assert pos[-1] == pytest.approx(0.75, abs=2e-3)
  assert np.abs(vel).max() < 50 * 2 * math.pi / 60 + 1e-3  # 50 rpm speed cap


def test_a_recording_of_the_model_itself_matches(robot):
  # A recording synthesized with the simulator must compare equal to its replay.
  start = [-0.35, 0.0, 0.0, 0.7, 0.0, 0.0, -0.35, 0.0, 0.0, 0.7, 0.0, 0.0]
  j = SLOTS.index("left_hip_pitch")
  goal_t = np.arange(0.0, 1.6, 0.01)
  goal_v = np.where(goal_t >= 0.5, start[j] + 0.3, start[j])
  t, pos, vel = simulate_segment(robot, "left_hip_pitch", dict(zip(SLOTS, start)),
                                 goal_t, goal_v, 0.0, 1.6)
  samples = []
  for ti, p, v in zip(t, pos, vel, strict=True):
    sp, sv = list(start), [0.0] * 12
    sp[j], sv[j] = float(p), float(v)
    samples.append({"t": float(ti), "stamp": float(ti), "pos": sp, "vel": sv})
  goals = []
  for ti, gv in zip(goal_t, goal_v, strict=True):
    g = list(start)
    g[j] = float(gv)
    goals.append({"t": float(ti), "goal": g})
  recording = {"leg_slots": SLOTS, "goals": goals, "samples": samples, "segments": [
    {"joint": "left_hip_pitch", "amplitude": 0.3, "from": start[j], "to": start[j] + 0.3,
     "t_step": 0.5, "t_end": 1.6}]}
  (result,) = compare(recording, output_dir=None, delay=0.0)
  for key in ("rise_time", "overshoot", "peak_speed", "final_error"):
    assert result["robot"][key] == pytest.approx(result["sim"][key], rel=0.05, abs=0.01), key


def test_gain_scales_change_the_simulated_response():
  start = {"left_hip_pitch": -0.35, "left_knee_pitch": 0.7}
  args = ("left_hip_pitch", start, np.array([0.0, 0.5]), np.array([-0.35, -0.30]), 0.3, 1.0)
  t, soft, v_soft = simulate_segment(HangingRoomie(), *args)
  _, stiff, v_stiff = simulate_segment(HangingRoomie(kp_scale=4.0, kv_scale=2.25), *args)
  after = t >= 0.5
  assert np.abs(v_stiff[after]).max() > 1.5 * np.abs(v_soft[after]).max()
