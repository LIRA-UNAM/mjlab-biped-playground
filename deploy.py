"""
Asimov Sim2Sim Deployment for mjlab-trained policies.

This script:
1. Loads an ONNX policy exported from mjlab
2. Extracts KP/KD/action_scale from ONNX metadata
3. Builds observations matching mjlab's velocity task
4. Runs sim2sim in MuJoCo with video recording and analysis

Observation tensor layout (47 dims - mjlab Asimov with gait_clock):
  0-2:   base_ang_vel (3)    - IMU angular velocity in body frame
  3-5:   projected_gravity (3) - Gravity projected into body frame
  6-8:   command (3)         - Velocity command (vx, vy, wz)
  9-20:  joint_pos (12)      - Joint positions relative to default
  21-32: joint_vel (12)      - Joint velocities
  33-44: actions (12)        - Last action sent
  45-46: gait_clock (2)      - [cos(phase), sin(phase)] for gait timing
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import csv
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

# Robot model (same XML used for training)
XML_PATH = "src/mjlab/asset_zoo/robots/asimov/xmls/asimov.xml"

# Simulation parameters
SIMULATION_DURATION = 20.0  # seconds
SIMULATION_DT = 0.005       # 200 Hz physics (matches mjlab)
CONTROL_DECIMATION = 4      # Policy runs at 50 Hz

# Video parameters
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30

# Number of observations and actions
NUM_OBS = 47  # base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(12) + joint_vel(12) + actions(12) + gait_clock(2)
NUM_ACTIONS = 12

# Gait clock frequency (Hz) - must match training config
GAIT_FREQ = 1.25  # Matches mjlab training config

# Joint limits for plotting
JOINT_LIMITS = {
    "L_hip_pitch":   (-2.094, 1.000),
    "L_hip_roll":    (-0.785, 0.785),
    "L_hip_yaw":     (-0.785, 0.785),
    "L_knee":        (-1.5, 1.5),
    "L_ankle_pitch": (-0.5, 0.5),
    "L_ankle_roll":  (-0.1, 0.1),
    "R_hip_pitch":   (-1.000, 2.094),
    "R_hip_roll":    (-0.785, 0.785),
    "R_hip_yaw":     (-0.785, 0.785),
    "R_knee":        (-1.5, 1.5),
    "R_ankle_pitch": (-0.5, 0.5),
    "R_ankle_roll":  (-0.1, 0.1),
}


# ============================================================================
# Helper Functions
# ============================================================================

def parse_csv_floats(s: str) -> list[float]:
    """Parse comma-separated float string."""
    return [float(x) for x in s.split(",")]


def load_onnx_with_metadata(onnx_path: str) -> dict:
    """Load ONNX model and extract metadata."""
    model = onnx.load(onnx_path)
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = prop.value
    return metadata


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller for joint torques (used when actuators are motors)."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def create_position_actuators(spec, joint_names, kps, kds, effort_limits, armatures):
    """Modify existing actuators to be position actuators (matching mjlab training)."""
    # Modify existing actuators to position type with KP/KD
    for i, joint_name in enumerate(joint_names):
        # Find the existing actuator for this joint
        act_name = f"{joint_name}_ctrl"
        act = None
        for a in spec.actuators:
            if a.name == act_name:
                act = a
                break

        if act is None:
            # Create new actuator if not found
            act = spec.add_actuator()
            act.name = f"{joint_name}_pos"
            # Find joint by iterating through spec.joints
            joint = None
            for j in spec.joints:
                if j.name == joint_name:
                    joint = j
                    break
            if joint is not None:
                act.target = joint.full_name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT

        # Configure as position actuator
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kps[i]  # kp gain
        act.biasprm[0] = 0.0     # bias offset
        act.biasprm[1] = -kps[i]  # -kp (position bias)
        act.biasprm[2] = -kds[i]  # -kd (velocity bias)
        if effort_limits is not None:
            act.forcerange = (-effort_limits[i], effort_limits[i])

        # Set armature on joint
        joint = None
        for j in spec.joints:
            if j.name == joint_name:
                joint = j
                break
        if joint is not None:
            joint.armature = armatures[i]


def get_projected_gravity(quat):
    """
    Get gravity vector projected into body frame.

    MuJoCo quaternion format: (w, x, y, z)
    Gravity in world frame: (0, 0, -1)

    This computes: R^T @ [0, 0, -1] where R is rotation from body to world.
    Uses the same formula as mjlab's quat_apply_inverse.
    """
    w, x, y, z = quat
    # Rotate [0, 0, -1] by inverse quaternion (matching mjlab quat_apply_inverse)
    gx = 2.0 * (w * y - x * z)
    gy = -2.0 * (w * x + y * z)
    gz = -1.0 + 2.0 * (x * x + y * y)
    return np.array([gx, gy, gz], dtype=np.float32)


def get_imu_data(data):
    """Get IMU sensor data (velocimeter and gyro in body frame)."""
    # MuJoCo velocimeter and gyro sensors output in body frame
    lin_vel = data.sensor("imu_lin_vel").data.copy().astype(np.float32)
    ang_vel = data.sensor("imu_ang_vel").data.copy().astype(np.float32)
    return lin_vel, ang_vel


def analyze_results(csv_path, joint_names, output_prefix):
    """Analyze simulation results and generate plots."""
    df = pd.read_csv(csv_path)
    time_arr = df['time'].values

    short_names = []
    for name in joint_names:
        short = name.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
        short_names.append(short)

    print("\n" + "=" * 70)
    print("TRACKING ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nStability Metrics:")
    print(f"  Pelvis height: {df['pelvis_z'].mean():.3f} +/- {df['pelvis_z'].std():.4f} m")
    print(f"  Gravity Z:     {df['grav_z'].mean():.3f} +/- {df['grav_z'].std():.4f} (upright = -1.0)")

    print(f"\n{'Joint':<14} {'RMS (rad)':<12} {'RMS (deg)':<12} {'Max (rad)':<12}")
    print("-" * 50)

    settled = df[df['time'] > 1.0]
    for short in short_names:
        if f"err_{short}" in df.columns:
            err = settled[f"err_{short}"]
            rms = np.sqrt(np.mean(err**2))
            max_err = np.abs(err).max()
            print(f"{short:<14} {rms:<12.4f} {np.degrees(rms):<12.2f} {max_err:<12.4f}")

    # Plot 1: Target vs Actual Joint Positions
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Target vs Actual Joint Positions", fontsize=14, fontweight='bold')

    for i, (ax, short) in enumerate(zip(axes, short_names)):
        if f"tgt_{short}" not in df.columns:
            continue
        tgt = df[f"tgt_{short}"].values
        act = df[f"act_{short}"].values

        ax.plot(time_arr, tgt, 'b-', linewidth=1.5, label='Target', alpha=0.8)
        ax.plot(time_arr, act, 'g-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        if short in JOINT_LIMITS:
            lim_min, lim_max = JOINT_LIMITS[short]
            ax.axhline(y=lim_min, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=lim_max, color='r', linestyle='--', linewidth=1, alpha=0.7)
            ax.fill_between(time_arr, lim_min, lim_max, color='green', alpha=0.1)

        ax.set_title(short, fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Angle (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plot1_path = f"{output_prefix}_joints.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {plot1_path}")

    # Plot 2: Stability Metrics
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig2.suptitle("Robot Stability Metrics", fontsize=14, fontweight='bold')

    ax1.plot(time_arr, df['pelvis_z'].values, 'b-', linewidth=2)
    ax1.axhline(y=0.75, color='r', linestyle='--', alpha=0.7, label='Target height')
    ax1.set_ylabel("Pelvis Height (m)", fontsize=11)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 0.9])

    ax2.plot(time_arr, df['grav_z'].values, 'g-', linewidth=2)
    ax2.axhline(y=-1.0, color='r', linestyle='--', alpha=0.7, label='Upright (-1.0)')
    ax2.set_ylabel("Gravity Z (body frame)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-1.1, 0.5])

    plt.tight_layout()
    plot2_path = f"{output_prefix}_stability.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")

    # Plot 3: IMU Data
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8))
    fig3.suptitle("IMU Data", fontsize=14, fontweight='bold')

    ax = axes3[0]
    ax.plot(time_arr, df['quat_w'].values, 'b-', linewidth=1.5, label='w', alpha=0.8)
    ax.plot(time_arr, df['quat_x'].values, 'r-', linewidth=1.5, label='x', alpha=0.8)
    ax.plot(time_arr, df['quat_y'].values, 'g-', linewidth=1.5, label='y', alpha=0.8)
    ax.plot(time_arr, df['quat_z'].values, 'm-', linewidth=1.5, label='z', alpha=0.8)
    ax.set_ylabel("Quaternion", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.1, 1.1])

    ax = axes3[1]
    ax.plot(time_arr, df['lin_vel_x'].values, 'r-', linewidth=1.5, label='vx', alpha=0.8)
    ax.plot(time_arr, df['lin_vel_y'].values, 'g-', linewidth=1.5, label='vy', alpha=0.8)
    ax.plot(time_arr, df['lin_vel_z'].values, 'b-', linewidth=1.5, label='vz', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Linear Velocity (m/s)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes3[2]
    ax.plot(time_arr, df['ang_vel_x'].values, 'r-', linewidth=1.5, label='wx', alpha=0.8)
    ax.plot(time_arr, df['ang_vel_y'].values, 'g-', linewidth=1.5, label='wy', alpha=0.8)
    ax.plot(time_arr, df['ang_vel_z'].values, 'b-', linewidth=1.5, label='wz', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot3_path = f"{output_prefix}_imu.png"
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot3_path}")


# ============================================================================
# Main Simulation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Asimov sim2sim deployment for mjlab policies")
    parser.add_argument("--policy", type=str, required=True, help="Path to ONNX policy file")
    parser.add_argument("--cmd_vx", type=float, default=0.5, help="X velocity command (m/s)")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Y velocity command (m/s)")
    parser.add_argument("--cmd_wz", type=float, default=0.0, help="Yaw rate command (rad/s)")
    parser.add_argument("--duration", type=float, default=SIMULATION_DURATION, help="Simulation duration (s)")
    parser.add_argument("--push_force", type=float, default=0.0, help="Push force in N (applied at push_time)")
    parser.add_argument("--push_time", type=float, default=2.0, help="Time to apply push (s)")
    parser.add_argument("--push_duration", type=float, default=0.1, help="Duration of push (s)")
    parser.add_argument("--push_dir", type=str, default="x", choices=["x", "y", "-x", "-y"], help="Push direction (x=forward, y=left, -x=back, -y=right)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    video_output = os.path.join(log_dir, "sim2sim.mp4")
    csv_output = os.path.join(log_dir, "sim2sim.csv")
    output_prefix = os.path.join(log_dir, "sim2sim")

    # Load ONNX metadata
    print("=" * 60)
    print("Asimov Sim2Sim - mjlab Policy Deployment")
    print("=" * 60)
    print(f"Policy: {args.policy}")

    metadata = load_onnx_with_metadata(args.policy)

    joint_names = metadata["joint_names"].split(",")
    kps = np.array(parse_csv_floats(metadata["joint_stiffness"]), dtype=np.float32)
    kds = np.array(parse_csv_floats(metadata["joint_damping"]), dtype=np.float32)
    default_joint_pos = np.array(parse_csv_floats(metadata["default_joint_pos"]), dtype=np.float32)
    action_scale = np.array(parse_csv_floats(metadata["action_scale"]), dtype=np.float32)
    observation_names = metadata["observation_names"].split(",")

    print(f"\nExtracted from ONNX metadata:")
    print(f"  Joint names: {joint_names}")
    print(f"  KP: {kps}")
    print(f"  KD: {kds}")
    print(f"  Default pos: {default_joint_pos}")
    print(f"  Action scale: {action_scale}")
    print(f"  Observations: {observation_names}")

    # Torque limits (from motor specs)
    torque_limits = np.array([120, 90, 60, 75, 36, 36, 120, 90, 60, 75, 36, 36], dtype=np.float32)

    # Velocity command
    cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float32)
    print(f"\nVelocity command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, wz={cmd[2]:.2f}")
    if args.push_force > 0:
        print(f"Push: {args.push_force:.0f}N at t={args.push_time:.1f}s for {args.push_duration:.2f}s")

    # Load ONNX policy
    print("\nLoading ONNX policy...")
    ort_session = ort.InferenceSession(args.policy)
    input_name = ort_session.get_inputs()[0].name
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {ort_session.get_inputs()[0].shape}")

    # Load MuJoCo model using MjSpec to create position actuators (matching mjlab training)
    print("\nLoading MuJoCo model with position actuators...")
    os.chdir(script_dir)
    xml_path = os.path.join(script_dir, XML_PATH)

    # Load XML into MjSpec so we can modify actuators
    spec = mujoco.MjSpec.from_file(xml_path)

    # Armature values from asimov_constants.py (reflected inertia)
    armatures = np.array([
        0.0652, 0.100, 0.0343, 0.0330, 0.0236, 0.0236,  # left leg
        0.0652, 0.100, 0.0343, 0.0330, 0.0236, 0.0236,  # right leg
    ], dtype=np.float32)

    # Create position actuators (replaces motor actuators from XML)
    create_position_actuators(spec, joint_names, kps, kds, torque_limits, armatures)

    # Compile the modified spec
    m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    # Get joint indices (actuators now match joint_names order)
    actuated_joint_names = joint_names
    print("\nActuated Joint Order (position actuators):")
    for i, name in enumerate(actuated_joint_names):
        print(f"  {i:02d}: {name}")

    # Find joint qpos/qvel indices
    qpos_ids = []
    qvel_ids = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qpos_ids.append(m.jnt_qposadr[jid])
        qvel_ids.append(m.jnt_dofadr[jid])
    qpos_ids = np.array(qpos_ids)
    qvel_ids = np.array(qvel_ids)

    short_names = []
    for name in actuated_joint_names:
        short = name.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
        short_names.append(short)

    # Initialize robot state
    d.qpos[7:7+NUM_ACTIONS] = default_joint_pos
    d.qpos[2] = 0.75  # Starting height
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    # Get pelvis body id for applying push force
    pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    push_start_step = int(args.push_time / SIMULATION_DT)
    push_end_step = int((args.push_time + args.push_duration) / SIMULATION_DT)

    # Setup video recording
    print(f"\nSetting up video recording ({VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps)...")
    renderer = mujoco.Renderer(m, VIDEO_HEIGHT, VIDEO_WIDTH)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Setup CSV logging
    csv_file = open(csv_output, "w", newline="")
    csv_writer = csv.writer(csv_file)

    header = ["step", "time", "pelvis_x", "pelvis_y", "pelvis_z"]
    header += ["quat_w", "quat_x", "quat_y", "quat_z"]
    header += ["lin_vel_x", "lin_vel_y", "lin_vel_z"]
    header += ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    header += ["grav_x", "grav_y", "grav_z"]
    for short in short_names:
        header += [f"tgt_{short}", f"act_{short}", f"err_{short}"]
    csv_writer.writerow(header)

    # Initialize state variables
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = default_joint_pos.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)

    # Gait clock state - tracks phase in [0, 1) like mjlab
    gait_phase = 0.0
    control_dt = SIMULATION_DT * CONTROL_DECIMATION  # 0.02s at 50Hz

    counter = 0
    frame_counter = 0
    frames_per_video_frame = int(1.0 / (VIDEO_FPS * SIMULATION_DT))
    total_steps = int(args.duration / SIMULATION_DT)

    print(f"\nRunning simulation for {args.duration} seconds...")
    print(f"Output: {log_dir}/")
    start_time = time.time()

    for step_idx in range(total_steps):
        # Get current joint state
        q = d.qpos[qpos_ids]
        dq = d.qvel[qvel_ids]

        # With position actuators, send position targets directly
        # MuJoCo's built-in PD controller handles torque computation
        d.ctrl[:] = target_dof_pos

        # Apply external push force if within push window
        if args.push_force > 0 and push_start_step <= step_idx < push_end_step:
            # Apply force in specified direction
            if args.push_dir == "x":
                d.xfrc_applied[pelvis_body_id, 0] = args.push_force
            elif args.push_dir == "-x":
                d.xfrc_applied[pelvis_body_id, 0] = -args.push_force
            elif args.push_dir == "y":
                d.xfrc_applied[pelvis_body_id, 1] = args.push_force
            elif args.push_dir == "-y":
                d.xfrc_applied[pelvis_body_id, 1] = -args.push_force
        else:
            d.xfrc_applied[pelvis_body_id, :] = 0

        mujoco.mj_step(m, d)

        counter += 1

        # Policy inference at control frequency
        if counter % CONTROL_DECIMATION == 0:
            # Get observations
            qj = d.qpos[qpos_ids]
            dqj = d.qvel[qvel_ids]
            pelvis_quat = d.qpos[3:7]  # (w, x, y, z)

            # Get body frame velocities from MuJoCo sensors
            lin_vel, ang_vel = get_imu_data(d)

            # Projected gravity
            grav = get_projected_gravity(pelvis_quat)

            # Build observation vector (mjlab Asimov order - 47 dims):
            # [base_ang_vel(3), projected_gravity(3), command(3),
            #  joint_pos(12), joint_vel(12), actions(12), gait_clock(2)]

            # Update gait phase - only advances when command is active (like mjlab)
            cmd_magnitude = np.sqrt(cmd[0]**2 + cmd[1]**2) + abs(cmd[2])
            if cmd_magnitude > 0.1:  # command_threshold from training config
                gait_phase = (gait_phase + control_dt * GAIT_FREQ) % 1.0

            # Convert phase to cos/sin (phase is in [0,1), multiply by 2π)
            phase_2pi = 2.0 * np.pi * gait_phase

            obs[0:3] = ang_vel                          # base_ang_vel
            obs[3:6] = grav                             # projected_gravity
            obs[6:9] = cmd                              # command
            obs[9:21] = qj - default_joint_pos          # joint_pos (relative)
            obs[21:33] = dqj                            # joint_vel
            obs[33:45] = action                         # last action
            obs[45:47] = [np.cos(phase_2pi), np.sin(phase_2pi)]  # gait_clock [cos, sin]

            # Run ONNX inference
            obs_input = obs.reshape(1, -1).astype(np.float32)
            action = ort_session.run(None, {input_name: obs_input})[0].squeeze()

            # Compute target position
            target_dof_pos = default_joint_pos + action * action_scale

            # Log to CSV
            sim_time = step_idx * SIMULATION_DT
            pelvis_pos = d.qpos[0:3]

            row = [counter, f"{sim_time:.3f}"]
            row += [f"{pelvis_pos[0]:.4f}", f"{pelvis_pos[1]:.4f}", f"{pelvis_pos[2]:.4f}"]
            row += [f"{pelvis_quat[0]:.4f}", f"{pelvis_quat[1]:.4f}", f"{pelvis_quat[2]:.4f}", f"{pelvis_quat[3]:.4f}"]
            row += [f"{lin_vel[0]:.4f}", f"{lin_vel[1]:.4f}", f"{lin_vel[2]:.4f}"]
            row += [f"{ang_vel[0]:.4f}", f"{ang_vel[1]:.4f}", f"{ang_vel[2]:.4f}"]
            row += [f"{grav[0]:.4f}", f"{grav[1]:.4f}", f"{grav[2]:.4f}"]

            for i in range(NUM_ACTIONS):
                tgt = target_dof_pos[i]
                act = qj[i]
                err = tgt - act
                row += [f"{tgt:.4f}", f"{act:.4f}", f"{err:.4f}"]
            csv_writer.writerow(row)

        # Record video frame
        if counter % frames_per_video_frame == 0:
            renderer.update_scene(d, camera="side_camera")
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            sim_time = step_idx * SIMULATION_DT
            pelvis_z = d.qpos[2]
            grav_z = get_projected_gravity(d.qpos[3:7])[2]
            status = "STANDING" if (pelvis_z > 0.4 and grav_z < -0.5) else "FALLEN!"

            cv2.putText(frame_bgr, f"Time: {sim_time:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Cmd: vx={cmd[0]:.2f} vy={cmd[1]:.2f} wz={cmd[2]:.2f}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Height: {pelvis_z:.2f}m | {status}",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            video_writer.write(frame_bgr)
            frame_counter += 1

        # Progress update
        if step_idx % int(0.5 / SIMULATION_DT) == 0:
            sim_time = step_idx * SIMULATION_DT
            pelvis_x, pelvis_y, pelvis_z = d.qpos[0:3]
            grav = get_projected_gravity(d.qpos[3:7])
            is_upright = grav[2] < -0.5
            status = "STANDING" if (pelvis_z > 0.4 and is_upright) else "FALLEN!"
            print(f"  t={sim_time:5.1f}s | pos=({pelvis_x:5.2f},{pelvis_y:5.2f},{pelvis_z:.2f}) | {status}")

    # Cleanup
    video_writer.release()
    renderer.close()
    csv_file.close()

    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f}s (real time)")

    print(f"\n{'=' * 60}")
    print(f"Simulation complete!")
    print(f"  Total frames: {frame_counter}")
    print(f"  Video saved: {video_output}")
    print(f"  CSV saved: {csv_output}")
    print(f"{'=' * 60}")

    # Generate analysis plots
    print("\nGenerating analysis plots...")
    analyze_results(csv_output, actuated_joint_names, output_prefix)

    print("\nDone!")


if __name__ == "__main__":
    main()
