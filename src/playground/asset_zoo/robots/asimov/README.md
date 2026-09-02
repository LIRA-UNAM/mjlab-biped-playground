# Asimov Robot Configuration

Asimov is a 12-DOF bipedal robot with 6 joints per leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll.

## Motor Specifications

Motors are from Synapticon EC-A series:

| Joint | Motor Model | Gear Ratio | Peak Torque | Rated Torque |
|-------|-------------|------------|-------------|--------------|
| Hip Pitch | EC-A6416-P2-25 | 25:1 | 120 Nm | 55 Nm |
| Hip Roll | EC-A5013-H17-100 | 100:1 | 90 Nm | 45 Nm |
| Hip Yaw | EC-A3814-H14-107 | 107:1 | 60 Nm | 30 Nm |
| Knee | EC-A4315-P2-36 | 36:1 | 75 Nm | 50 Nm |
| Ankle Pitch/Roll | EC-A4310-P2-36 | 36:1 | 36 Nm | 18 Nm |

## Armature (Reflected Inertia)

In MuJoCo, `armature` adds to a joint's effective inertia, modeling the reflected inertia of the motor and gearbox. This is critical for accurate dynamics simulation.

### Formula

For a geared motor system:

```
J_reflected = J_rotor × (gear_ratio)²
```

Where:
- `J_rotor` is the rotor inertia (from motor datasheet, in kg·m²)
- `gear_ratio` is the reduction ratio

The squared term comes from energy conservation: when the output rotates by angle θ, the motor rotates by θ × gear_ratio, so the kinetic energy contribution is:

```
KE = ½ × J_rotor × (gear_ratio × θ̇)² = ½ × (J_rotor × gear_ratio²) × θ̇²
```

### Asimov Armature Values

Using rotor inertias from Synapticon datasheets:

| Joint | J_rotor (kg·mm²) | Gear Ratio | Armature (kg·m²) |
|-------|------------------|------------|------------------|
| Hip Pitch | 104.395 | 25 | 0.0652 |
| Hip Roll | 10.0 | 100 | 0.100 |
| Hip Yaw | 3.0 | 107 | 0.0343 |
| Knee | 25.5 | 36 | 0.0330 |
| Ankle | 18.2 | 36 | 0.0236 |

Example calculation for hip roll:
```
J_reflected = 10.0 kg·mm² × 100² = 10.0 × 10⁻⁶ kg·m² × 10000 = 0.100 kg·m²
```

### Comparison with Unitree G1

G1 uses two-stage planetary gearboxes, requiring a more complex formula that accounts for inertia contributions at each stage. Asimov uses single-stage or harmonic drives where the manufacturer provides total rotor inertia.

| Joint | G1 Armature | Asimov Armature | Ratio |
|-------|-------------|-----------------|-------|
| Hip Pitch | 0.010 | 0.065 | 6.4x |
| Hip Roll | 0.025 | 0.100 | 4.0x |
| Hip Yaw | 0.015 | 0.034 | 2.3x |
| Knee | 0.025 | 0.033 | 1.3x |
| Ankle | 0.007 | 0.024 | 3.3x |

Asimov's higher armatures are due to higher gear ratios (e.g., hip_roll uses 100:1 vs G1's ~22:1).

## Physics-Based PD Gains

We use physics-based gains derived from the reflected inertia:

### Stiffness (KP)

```
KP = J_reflected × ω_n²
```

Where ω_n = 10 Hz × 2π ≈ 62.83 rad/s (natural frequency)

This ensures the joint behaves like a critically damped second-order system with 10 Hz bandwidth.

### Damping (KD)

The physics-based formula is:
```
KD = 2 × ζ × J_reflected × ω_n
```

Where ζ is the damping ratio (typically 1.0-2.0 for critical/overdamped).

However, we cap KD at **5.0 Nm·s/rad** for all joints due to hardware limitations. The Synapticon motors have a maximum KD of 5.0 that can be achieved in firmware.

### Resulting Gains

| Joint | KP (Nm/rad) | KD (Nm·s/rad) |
|-------|-------------|---------------|
| Hip Pitch | 257.4 | 5.0 |
| Hip Roll | 394.8 | 5.0 |
| Hip Yaw | 135.4 | 5.0 |
| Knee | 130.3 | 5.0 |
| Ankle | 93.2 | 5.0 |

## Why This Matters for Sim2Real

1. **Armature mismatch causes sim2sim failure**: If the deployment XML has different armature values than training, the robot will fall immediately. The dynamics are fundamentally different.

2. **KD capping for hardware**: Using physics-optimal KD (~8-12 for large joints) in simulation would learn policies that can't be deployed. Capping at 5.0 ensures sim2real compatibility.

3. **Consistent gains**: Extracting KP/KD from ONNX metadata ensures the same gains are used in deployment as in training.

## Files

- `asimov_constants.py` - Robot configuration with gains and armatures
- `xmls/asimov.xml` - MuJoCo XML with armature values in joint definitions
- `assets/meshes/` - Visual and collision meshes
