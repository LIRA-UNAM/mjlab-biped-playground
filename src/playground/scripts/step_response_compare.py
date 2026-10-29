"""Compare recorded Roomie joint step responses with the training actuator model.

Reads a recording made on the robot with roomie_ppowalk's
step_response_record.py (goals and /joint_states while one leg joint at a time
is stepped, robot hanging) and replays the same goal timeline on the mjlab
model the policy was trained with: fixed base (hanging), training physics
(5 ms, implicitfast) and the SteadyWin cascade actuators of
ROOMIE_ARTICULATION (gains, 50 rpm speed cap, torque-speed curve, armature).

For every step it prints delay, rise time, overshoot, peak speed, settling
time and steady-state error for the robot and the simulation, and saves a plot
(position: goal, robot, sim; velocity: robot, sim).

  uv run step_response_compare ~/step_left.json --output-dir step_plots
  uv run step_response_compare ~/step_left.json --delay-ms 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.entity import Entity

from playground.asset_zoo.robots.roomie.actuators import (
  SteadywinCascadeActuator,
  SteadywinCascadeActuatorCfg,
)
from playground.asset_zoo.robots.roomie.roomie_constants import (
  HOME_KEYFRAME,
  ROOMIE_ARTICULATION,
  get_roomie_robot_cfg,
  scale_cascade_gains,
)

SIM_DT = 0.005
# Public ROS sign of the steppable leg joints (public = sign * MJCF): all +1, as
# in roomie_ppowalk_policy.yaml.
PUBLIC_SIGN = {
  f"{side}_{axis}": 1.0
  for side in ("left", "right")
  for axis in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch")
}
# Leg joints whose recorded start position is copied into the simulation.
_LEG_JOINTS = tuple(PUBLIC_SIGN)


# Metrics.


@dataclass
class StepMetrics:
  delay: float  # s until 5 % of the step
  rise_time: float  # s from 10 % to 90 %
  overshoot: float  # % of the step
  peak_speed: float  # rad/s
  settling_time: float  # s until within 5 % of the step (and staying)
  final_error: float  # rad, mean of the last 0.3 s (goal - position)


def step_metrics(t: np.ndarray, pos: np.ndarray, vel: np.ndarray, start: float,
                 goal: float) -> StepMetrics:
  """Metrics of a step response; t is relative to the step (t >= 0 only)."""
  amplitude = goal - start
  progress = (pos - start) / amplitude if amplitude != 0.0 else np.zeros_like(pos)

  def first_time(level: float) -> float:
    idx = np.nonzero(progress >= level)[0]
    return float(t[idx[0]]) if idx.size else math.nan

  t05, t10, t90 = first_time(0.05), first_time(0.10), first_time(0.90)
  outside = np.nonzero(np.abs(progress - 1.0) > 0.05)[0]
  if outside.size == 0:
    settling = 0.0
  elif outside[-1] + 1 < len(t):
    settling = float(t[outside[-1] + 1])
  else:
    settling = math.nan
  tail = t >= t[-1] - 0.3
  return StepMetrics(
    delay=t05,
    rise_time=t90 - t10,
    overshoot=max(0.0, float(progress.max()) - 1.0) * 100.0,
    peak_speed=float(np.nanmax(np.abs(vel))) if np.isfinite(vel).any() else math.nan,
    settling_time=settling,
    final_error=float(np.mean(goal - pos[tail])),
  )


# Simulation.


def _fixed_base_model() -> mujoco.MjModel:
  cfg = get_roomie_robot_cfg()
  floating = cfg.spec_fn

  def spec_fn() -> mujoco.MjSpec:
    spec = floating()
    # Defaults, so attaching does not report the XML's own <option>; the
    # training physics is set on the compiled model below.
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    for joint in list(spec.joints):
      if joint.type == mujoco.mjtJoint.mjJNT_FREE:
        spec.delete(joint)
    return spec

  hanging = replace(cfg.init_state, pos=(0.0, 0.0, 1.0))
  with warnings.catch_warnings():
    # The XML's own <option> differs from the training one, set below.
    warnings.simplefilter("ignore", UserWarning)
    model = Entity(replace(cfg, spec_fn=spec_fn, init_state=hanging)).compile()
  model.opt.timestep = SIM_DT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return model


def _standalone(cfg: SteadywinCascadeActuatorCfg, n: int) -> SteadywinCascadeActuator:
  """A cascade actuator for one environment, driven outside an mjlab env."""
  act = SteadywinCascadeActuator(cfg, entity=None, target_ids=list(range(n)),
                                 target_names=[f"j{i}" for i in range(n)])
  shape = (1, n)
  act.stiffness = torch.full(shape, cfg.stiffness)
  act.damping = torch.full(shape, cfg.damping)
  act.force_limit = torch.full(shape, cfg.effort_limit)
  act.integral_gain = torch.full(shape, cfg.integral_gain)
  act.velocity_limit = torch.full(shape, cfg.velocity_limit)
  act.backlash = torch.zeros(shape)
  act._integral = torch.zeros(shape)
  act._velocity_reference = torch.zeros(shape)
  act._pending_error = torch.zeros(shape)
  act._integrate = torch.zeros(shape, dtype=torch.bool)
  act._motor_pos = torch.zeros(shape)
  act._motor_pos_valid = torch.zeros(1, dtype=torch.bool)
  return act


class HangingRoomie:
  """Fixed-base Roomie with the training cascade actuators on every joint.

  kp_scale / kv_scale scale every actuator's gains as scale_cascade_gains does
  for the training task (1.0 = ROOMIE_ARTICULATION).
  """

  def __init__(self, kp_scale: float = 1.0, kv_scale: float = 1.0) -> None:
    self.model = _fixed_base_model()
    self.data = mujoco.MjData(self.model)
    m = self.model
    joint_actuator = {}
    for a in range(m.nu):
      joint_actuator[m.joint(int(m.actuator_trnid[a, 0])).name] = a
    self.groups = []  # (actuator, joint names, qpos adr, dof adr, actuator ids)
    articulation = scale_cascade_gains(ROOMIE_ARTICULATION, kp_scale, kv_scale)
    for cfg in articulation.actuators:
      assert isinstance(cfg, SteadywinCascadeActuatorCfg)
      names = [n for n in joint_actuator
               if any(re.fullmatch(e, n) for e in cfg.target_names_expr)]
      self.groups.append((
        _standalone(cfg, len(names)), names,
        [m.jnt_qposadr[m.joint(n).id] for n in names],
        [m.jnt_dofadr[m.joint(n).id] for n in names],
        [joint_actuator[n] for n in names],
      ))

  def reset(self, leg_pose: dict[str, float]) -> dict[str, float]:
    """HOME_KEYFRAME with the given MJCF leg joint angles; returns all targets."""
    mujoco.mj_resetData(self.model, self.data)
    targets = {}
    for _, names, qadr, _, _ in self.groups:
      for n, adr in zip(names, qadr):
        value = leg_pose.get(n, HOME_KEYFRAME.joint_pos.get(n, 0.0))
        targets[n] = value
    for name, value in HOME_KEYFRAME.joint_pos.items():
      self.data.joint(name).qpos[0] = value
    for name, value in leg_pose.items():
      self.data.joint(name).qpos[0] = value
    mujoco.mj_forward(self.model, self.data)
    for act, *_ in self.groups:
      act.reset()
    return targets

  def step(self, targets: dict[str, float]) -> None:
    d = self.data
    for act, names, qadr, dadr, aids in self.groups:
      t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
      cmd = ActuatorCmd(
        position_target=t([targets[n] for n in names]),
        velocity_target=torch.zeros(1, len(names)),
        effort_target=torch.zeros(1, len(names)),
        pos=t([d.qpos[a] for a in qadr]),
        vel=t([d.qvel[a] for a in dadr]),
      )
      torque = act.compute(cmd)[0].numpy()
      d.ctrl[aids] = torque
    mujoco.mj_step(self.model, d)
    for act, *_ in self.groups:
      act.update(SIM_DT)


def simulate_segment(robot: HangingRoomie, joint: str, start_pose: dict[str, float],
                     goal_t: np.ndarray, goal_v: np.ndarray, t_from: float, t_to: float,
                     settle: float = 1.5, delay: float = 0.0):
  """Simulated MJCF position/velocity of ``joint`` for a public goal timeline.

  start_pose holds the public leg positions at the segment start (they are the
  held goals of the other joints); goal_t/goal_v the recorded goal of ``joint``
  (zero-order hold, shifted by ``delay``).
  """
  sign = PUBLIC_SIGN[joint]
  mjcf_pose = {n: PUBLIC_SIGN[n] * start_pose[n] for n in _LEG_JOINTS if n in start_pose}
  first_goal = sign * float(np.interp(t_from, goal_t, goal_v))
  mjcf_pose[joint] = first_goal
  targets = robot.reset(mjcf_pose)
  for _ in range(int(settle / SIM_DT)):
    robot.step(targets)
  times, pos, vel = [], [], []
  jid = robot.model.joint(joint).id
  qadr, dadr = robot.model.jnt_qposadr[jid], robot.model.jnt_dofadr[jid]
  idx_sorted = np.argsort(goal_t)
  goal_t, goal_v = goal_t[idx_sorted], goal_v[idx_sorted]
  t = t_from
  while t <= t_to:
    k = np.searchsorted(goal_t, t - delay, side="right") - 1
    targets[joint] = sign * float(goal_v[max(k, 0)])
    robot.step(targets)
    t += SIM_DT
    times.append(t)
    pos.append(sign * robot.data.qpos[qadr])
    vel.append(sign * robot.data.qvel[dadr])
  return np.array(times), np.array(pos), np.array(vel)


# Report.


def _fmt(m: StepMetrics) -> str:
  def f(v: float, scale: float = 1.0, digits: int = 0) -> str:
    return "  n/a" if not math.isfinite(v) else f"{v * scale:5.{digits}f}"
  return (f"delay {f(m.delay, 1e3)} ms  rise {f(m.rise_time, 1e3)} ms  "
          f"overshoot {f(m.overshoot, 1, 1)} %  peak {f(m.peak_speed, 1, 2)} rad/s  "
          f"settle {f(m.settling_time, 1e3)} ms  final err {f(m.final_error, 1e3, 1)} mrad")


def compare(recording: dict, output_dir: Path | None, delay: float, kp_scale: float = 1.0,
            kv_scale: float = 1.0) -> list[dict]:
  slots = recording["leg_slots"]
  goals_t = np.array([g["t"] for g in recording["goals"]])
  goals = np.array([g["goal"] for g in recording["goals"]])
  samp_t = np.array([s["t"] for s in recording["samples"]])
  samp_pos = np.array([s["pos"] for s in recording["samples"]], dtype=float)
  samp_vel = np.array([s["vel"] for s in recording["samples"]], dtype=float)
  robot = HangingRoomie(kp_scale, kv_scale)
  results = []
  for k, seg in enumerate(recording["segments"]):
    joint = seg["joint"]
    j = slots.index(joint)
    t_step, t_end = seg["t_step"], seg["t_end"]
    t_from = t_step - 0.2
    # Positions of all leg slots just before the step: the held goals.
    before = np.nonzero(samp_t <= t_step)[0]
    start_pose = dict(zip(slots, samp_pos[before[-1]])) if before.size else {}
    window = (samp_t >= t_from) & (samp_t <= t_end)
    rt, rpos, rvel = samp_t[window], samp_pos[window, j], samp_vel[window, j]
    st, spos, svel = simulate_segment(
      robot, joint, start_pose, goals_t, goals[:, j], t_from, t_end, delay=delay)
    after_r, after_s = rt >= t_step, st >= t_step
    robot_m = step_metrics(rt[after_r] - t_step, rpos[after_r], rvel[after_r],
                           seg["from"], seg["to"])
    sim_m = step_metrics(st[after_s] - t_step, spos[after_s], svel[after_s],
                         seg["from"], seg["to"])
    print(f"[{k}] {joint} {seg['from']:+.3f} -> {seg['to']:+.3f} rad ({seg['amplitude']:+.3f})")
    print(f"    robot  {_fmt(robot_m)}")
    print(f"    sim    {_fmt(sim_m)}")
    results.append({"segment": k, "joint": joint, "amplitude": seg["amplitude"],
                    "robot": robot_m.__dict__, "sim": sim_m.__dict__})
    if output_dir is not None:
      _plot(output_dir / f"{k:02d}_{joint}_{seg['amplitude']:+.3f}.png", joint, seg,
            goals_t - t_step, goals[:, j], rt - t_step, rpos, rvel, st - t_step, spos, svel)
  return results


def _plot(path, joint, seg, gt, gv, rt, rpos, rvel, st, spos, svel) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, (ax_p, ax_v) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
  shown = (gt >= rt[0]) & (gt <= rt[-1]) if len(rt) else slice(None)
  ax_p.step(gt[shown], gv[shown], where="post", color="0.5", label="goal")
  ax_p.plot(rt, rpos, label="robot")
  ax_p.plot(st, spos, "--", label="sim (training model)")
  ax_p.set_ylabel("position (rad)")
  ax_p.legend(loc="best")
  ax_p.set_title(f"{joint}: {seg['from']:+.3f} -> {seg['to']:+.3f} rad")
  ax_v.plot(rt, rvel, label="robot")
  ax_v.plot(st, svel, "--", label="sim")
  ax_v.set_ylabel("velocity (rad/s)")
  ax_v.set_xlabel("time from step (s)")
  ax_v.legend(loc="best")
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=110)
  plt.close(fig)


# Gain fit.


def _arrays(recording: dict):
  goals_t = np.array([g["t"] for g in recording["goals"]])
  goals = np.array([g["goal"] for g in recording["goals"]])
  samp_t = np.array([s["t"] for s in recording["samples"]])
  samp_pos = np.array([s["pos"] for s in recording["samples"]], dtype=float)
  return goals_t, goals, samp_t, samp_pos


def valid_segments(recording: dict, joint: str, min_progress: float = 0.7) -> list[dict]:
  """Steps of ``joint`` the robot actually followed (moved at least
  ``min_progress`` of the step within the segment and its feedback updated)."""
  slots = recording["leg_slots"]
  j = slots.index(joint)
  _, _, samp_t, samp_pos = _arrays(recording)
  out = []
  for seg in recording["segments"]:
    if seg["joint"] != joint:
      continue
    window = (samp_t >= seg["t_step"]) & (samp_t <= seg["t_end"])
    pos = samp_pos[window, j]
    if pos.size < 10 or not np.isfinite(pos).all():
      continue
    progress = (pos[-10:].mean() - seg["from"]) / seg["amplitude"]
    if progress >= min_progress:
      out.append(seg)
  return out


def fit_cost(robot: HangingRoomie, recordings: list[dict], joint: str, horizon: float = 0.6,
             delay: float = 0.0) -> float:
  """Mean normalized RMS position error of the valid steps of ``joint``."""
  costs = []
  for recording in recordings:
    slots = recording["leg_slots"]
    j = slots.index(joint)
    goals_t, goals, samp_t, samp_pos = _arrays(recording)
    for seg in valid_segments(recording, joint):
      t_step = seg["t_step"]
      t_to = min(seg["t_end"], t_step + horizon)
      before = np.nonzero(samp_t <= t_step)[0]
      start_pose = dict(zip(slots, samp_pos[before[-1]]))
      st, spos, _ = simulate_segment(robot, joint, start_pose, goals_t, goals[:, j],
                                     t_step - 0.05, t_to, delay=delay)
      window = (samp_t >= t_step) & (samp_t <= t_to)
      sim_at_robot = np.interp(samp_t[window], st, spos)
      err = samp_pos[window, j] - sim_at_robot
      costs.append(float(np.sqrt(np.mean(err**2))) / abs(seg["amplitude"]))
  if not costs:
    raise ValueError(f"no valid steps of {joint} in the recordings")
  return float(np.mean(costs))


def fit_gains(recordings: list[dict], joint: str, kp_scales, kv_scales, delay: float = 0.0):
  """Grid search of (kp_scale, kv_scale); returns (best, {(kp, kv): cost})."""
  table = {}
  for kv in kv_scales:
    for kp in kp_scales:
      table[(kp, kv)] = fit_cost(HangingRoomie(kp, kv), recordings, joint, delay=delay)
  best = min(table, key=table.get)
  return best, table


def fit_main() -> None:
  parser = argparse.ArgumentParser(
    description="Fit kp/kv scales of the training actuators to recorded steps.")
  parser.add_argument("recordings", type=Path, nargs="+")
  parser.add_argument("--joint", default="left_hip_pitch")
  parser.add_argument("--kp-scales", type=float, nargs="+",
                      default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
  parser.add_argument("--kv-scales", type=float, nargs="+",
                      default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
  parser.add_argument("--delay-ms", type=float, default=0.0)
  args = parser.parse_args()
  recordings = [json.loads(p.expanduser().read_text()) for p in args.recordings]
  n = sum(len(valid_segments(r, args.joint)) for r in recordings)
  print(f"{n} valid {args.joint} steps")
  best, table = fit_gains(recordings, args.joint, args.kp_scales, args.kv_scales,
                          delay=args.delay_ms / 1e3)
  print("normalized RMS error (rows kv_scale, columns kp_scale)")
  print("kv\\kp " + " ".join(f"{kp:6.2f}" for kp in args.kp_scales))
  for kv in args.kv_scales:
    print(f"{kv:6.2f} " + " ".join(f"{table[(kp, kv)]:6.3f}" for kp in args.kp_scales))
  print(f"best: kp_scale {best[0]}, kv_scale {best[1]} (cost {table[best]:.3f}; "
        f"1.0/1.0 = {table.get((1.0, 1.0), float('nan')):.3f})")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("recording", type=Path)
  parser.add_argument("--output-dir", type=Path, default=Path("step_response_plots"))
  parser.add_argument("--delay-ms", type=float, default=0.0,
                      help="extra command delay applied to the simulated goals")
  parser.add_argument("--json", type=Path, help="also write the metrics as JSON")
  parser.add_argument("--kp-scale", type=float, default=1.0)
  parser.add_argument("--kv-scale", type=float, default=1.0)
  args = parser.parse_args()
  recording = json.loads(args.recording.expanduser().read_text())
  results = compare(recording, args.output_dir, args.delay_ms / 1e3, args.kp_scale,
                    args.kv_scale)
  print(f"plots in {args.output_dir}/")
  if args.json:
    args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
