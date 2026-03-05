# Asimov Motor Parameters Reference

## Motor Assignments

| Joint | Side | Motor ID | CAN ID | Actuator Model | Gear Ratio | Gear Type |
|-------|------|----------|--------|----------------|------------|-----------|
| hip_pitch | L | 0 | 0x01 | EC-A6416-P2-25 | 25:1 | Planetary (P2) |
| hip_pitch | R | 6 | 0x07 | EC-A6416-P2-25 | 25:1 | Planetary (P2) |
| hip_roll | L | 1 | 0x02 | EC-A5013-H17-100 | 100:1 | Harmonic (H17) |
| hip_roll | R | 7 | 0x08 | EC-A5013-H17-100 | 100:1 | Harmonic (H17) |
| hip_yaw | L | 2 | 0x03 | EC-A3814-H14-107 | 107:1 | Harmonic (H14) |
| hip_yaw | R | 8 | 0x09 | EC-A3814-H14-107 | 107:1 | Harmonic (H14) |
| knee | L | 3 | 0x04 | EC-A4315-P2-36 | 36:1 | Planetary (P2) |
| knee | R | 9 | 0x0A | EC-A4315-P2-36 | 36:1 | Planetary (P2) |
| ankle_pitch | L | 4 | 0x05 | EC-A4310-P2-36 | 36:1 | Planetary (P2) |
| ankle_pitch | R | 10 | 0x0B | EC-A4310-P2-36 | 36:1 | Planetary (P2) |
| ankle_roll | L | 5 | 0x06 | EC-A4310-P2-36 | 36:1 | Planetary (P2) |
| ankle_roll | R | 11 | 0x0C | EC-A4310-P2-36 | 36:1 | Planetary (P2) |

---

## Complete Motor Specifications (from ENCOS Spec Sheets)

### EC-A6416-P2-25 (Hip Pitch)

| Parameter | Value | Unit |
|-----------|-------|------|
| Reduction Ratio | 25 | - |
| Rated Voltage | 48 | V |
| Rated Bus Current | 13 | A |
| Rated Phase Current | 17.5 | A |
| Rated Power | 624 | W |
| Rated Speed | 107 | RPM |
| Rated Torque | 40 | Nm |
| Peak Speed | 120 | RPM |
| Peak Phase Current | 60 | A |
| Peak Torque | 120 | Nm |
| Efficiency | 73 | % |
| Torque Constant (KT) | 2.74 | Nm/A |
| Weight | 803 | g |
| Backlash | 15 | Arcmin |
| Motor Size (OD × Thickness) | 88 × 67.5 | mm |
| Communication | CAN | - |
| Baud Rate | 1M | - |
| Double Encoder | Yes | - |
| Axial/Radial Load | 2.14 | kN |
| Bending Moment Resistance | 40 | Nm |
| Cross Roller Bearing Type | 4005 | - |
| **Rotor Inertia** | **104.395** | **kg·mm²** |
| Torque Density | 149.44 | Nm/kg |

### EC-A5013-H17-100 (Hip Roll)

| Parameter | Value | Unit |
|-----------|-------|------|
| Reduction Ratio | 100 | - |
| Rated Voltage | 48 | V |
| Rated Bus Current | 2.5 | A |
| Rated Phase Current | 3.0 | A |
| Rated Power | 120 | W |
| Rated Speed | 33 | RPM |
| Rated Torque | 30 | Nm |
| Peak Speed | 38 | RPM |
| Peak Phase Current | 20 | A |
| Peak Torque | 90 | Nm |
| Efficiency | 31 | % |
| Torque Constant (KT) | 5.9 | Nm/A |
| Weight | 640 | g |
| Backlash | 10 | Arcsec |
| Motor Size (OD × Thickness) | 63 × 81.5 | mm |
| Communication | CAN | - |
| Baud Rate | 1M | - |
| Double Encoder | Yes | - |
| Axial/Radial Load | 1.89 | kN |
| Bending Moment Resistance | 28 | Nm |
| Cross Roller Bearing Type | 3005 | - |
| **Rotor Inertia** | **10** | **kg·mm²** |
| Torque Density | 140.625 | Nm/kg |

### EC-A3814-H14-107 (Hip Yaw)

| Parameter | Value | Unit |
|-----------|-------|------|
| Reduction Ratio | 107 | - |
| Rated Voltage | 48 | V |
| Rated Bus Current | 2.8 | A |
| Rated Phase Current | 4.0 | A |
| Rated Power | 134.4 | W |
| Rated Speed | 47 | RPM |
| Rated Torque | 20 | Nm |
| Peak Speed | 52 | RPM |
| Peak Phase Current | 10 | A |
| Peak Torque | 60 | Nm |
| Efficiency | 51 | % |
| Torque Constant (KT) | 4.2 | Nm/A |
| Weight | 400 | g |
| Backlash | 10 | Arcsec |
| Motor Size (OD × Thickness) | 53 × 78.5 | mm |
| Communication | CAN | - |
| Baud Rate | 1M | - |
| Double Encoder | Yes | - |
| Axial/Radial Load | 1.89 | kN |
| Bending Moment Resistance | 28 | Nm |
| Cross Roller Bearing Type | 3005 | - |
| **Rotor Inertia** | **3** | **kg·mm²** |
| Torque Density | 150 | Nm/kg |

### EC-A4315-P2-36 (Knee)

| Parameter | Value | Unit |
|-----------|-------|------|
| Reduction Ratio | 36 | - |
| Rated Voltage | 48 | V |
| Rated Bus Current | 8.28 | A |
| Rated Phase Current | 10.22 | A |
| Rated Power | 397 | W |
| Rated Speed | 109 | RPM |
| Rated Torque | 25 | Nm |
| Peak Speed | 117 | RPM |
| Peak Phase Current | 30 | A |
| Peak Torque | 75 | Nm |
| Efficiency | 74 | % |
| Torque Constant (KT) | 2.8 | Nm/A |
| Weight | 455 | g |
| Backlash | 10 | Arcmin |
| Motor Size (OD × Thickness) | 56 × 69.5 | mm |
| Communication | CAN | - |
| Baud Rate | 1M | - |
| Double Encoder | Yes | - |
| Axial/Radial Load | 1.89 | kN |
| Bending Moment Resistance | 28 | Nm |
| Cross Roller Bearing Type | 3005 | - |
| **Rotor Inertia** | **25.5** | **kg·mm²** |
| Torque Density | 164.84 | Nm/kg |

### EC-A4310-P2-36 (Ankle Pitch & Roll)

| Parameter | Value | Unit |
|-----------|-------|------|
| Reduction Ratio | 36 | - |
| Rated Voltage | 24 (supports 48V) | V |
| Rated Bus Current | 6.5 | A |
| Rated Phase Current | 7.8 | A |
| Rated Power | 156 | W |
| Rated Speed | 75 | RPM |
| Rated Torque | 12 | Nm |
| Peak Speed | 89 | RPM |
| Peak Phase Current | 30 | A |
| Peak Torque | 36 | Nm |
| Efficiency | 71 | % |
| Torque Constant (KT) | 1.4 | Nm/A |
| Weight | 384 | g |
| Backlash | 10 | Arcmin |
| Motor Size (OD × Thickness) | 56 × 60.5 | mm |
| Communication | CAN | - |
| Baud Rate | 1M | - |
| Double Encoder | Yes | - |
| Axial/Radial Load | 1.89 | kN |
| Bending Moment Resistance | TBD | Nm |
| Cross Roller Bearing Type | TBD | - |
| **Rotor Inertia** | **18.2** | **kg·mm²** |
| Torque Density | TBD | Nm/kg |

---

## Summary Table

| Motor | Joint | Gear Ratio | KT (Nm/A) | Rated Torque | Peak Torque | Weight | Rotor Inertia (kg·mm²) | Efficiency |
|-------|-------|------------|-----------|--------------|-------------|--------|------------------------|------------|
| EC-A6416-P2-25 | Hip Pitch | 25:1 | 2.74 | 40 Nm | 120 Nm | 803g | **104.395** | 73% |
| EC-A5013-H17-100 | Hip Roll | 100:1 | 5.9 | 30 Nm | 90 Nm | 640g | **10** | 31% |
| EC-A3814-H14-107 | Hip Yaw | 107:1 | 4.2 | 20 Nm | 60 Nm | 400g | **3** | 51% |
| EC-A4315-P2-36 | Knee | 36:1 | 2.8 | 25 Nm | 75 Nm | 455g | **25.5** | 74% |
| EC-A4310-P2-36 | Ankle | 36:1 | 1.4 | 12 Nm | 36 Nm | 384g | **18.2** | 71% |

---

## CAN Protocol Ranges (CRITICAL!)

**EC系列力位混控协议范围 (EC Series Force/Position Mixed Control Protocol Range) - 2025.11.28**

### Complete Protocol Specification

| Motor Model | Gear Type | KP Range | KD Range | SPD Range (rad/s) | POS Range (rad) | Torque Range (Nm) | Current Range (A) | KT (Nm/A) |
|-------------|-----------|----------|----------|-------------------|-----------------|-------------------|-------------------|-----------|
| **EC-A2806-P2-36** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -12~12 | -10~10 | 1.4 |
| **EC-A4310-P2-36** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -30~30 | -30~30 | 1.4 |
| **EC-A4315-P2-36** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -70~70 | -30~30 | 2.8 |
| **EC-A6408-P2-25** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -60~60 | -60~60 | 2.35 |
| **EC-A6416-P2-25** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -120~120 | -60~60 | 2.74 |
| **EC-A8112-P1-18** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -90~90 | -60~60 | 2.1 |
| **EC-A8116-P1-18** | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -150~150 | -70~70 | 2.35 |
| EC-A10020-P1-12/6 | Planetary | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -150~150 | -70~70 | 2.5/1.7 |
| EC-A13720-P1-11.4 | Planetary | 0~500 | 0~50 | -18~18 | -12.5~12.5 | -400~400 | -220~220 | 2.5 |
| EC-A10020-P2-24 | Planetary | 0~500 | 0~50 | -18~18 | -12.5~12.5 | -300~300 | -140~140 | 2.6 |
| EC-A13715-P1-12.67 | Planetary | 0~500 | 0~50 | -18~18 | -12.5~12.5 | -320~320 | -220~220 | 2.5 |
| **EC-A3814-H14-107** | Harmonic | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -60~60 | -20~20 | 4.2 |
| **EC-A5013-H17-100** | Harmonic | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -90~90 | -30~30 | 5.9 |
| **EC-A6013-H20-100** | Harmonic | 0~500 | 0~5 | -18~18 | -12.5~12.5 | -130~130 | -35~35 | 5.6 |

**Note:** From 2025/1/15, motors in red (larger models) have default KD range changed from 0~5 to 0~50.

### Asimov Motor CAN Ranges (Extracted)

| Joint | Motor Model | KP Range | KD Range | Torque (Nm) | Current (A) | KT (Nm/A) |
|-------|-------------|----------|----------|-------------|-------------|-----------|
| **hip_pitch** | EC-A6416-P2-25 | 0~500 | 0~5 | ±120 | ±60 | 2.74 |
| **hip_roll** | EC-A5013-H17-100 | 0~500 | 0~5 | ±90 | ±30 | 5.9 |
| **hip_yaw** | EC-A3814-H14-107 | 0~500 | 0~5 | ±60 | ±20 | 4.2 |
| **knee** | EC-A4315-P2-36 | 0~500 | 0~5 | ±70 | ±30 | 2.8 |
| **ankle** | EC-A4310-P2-36 | 0~500 | 0~5 | ±30 | ±30 | 1.4 |

---

## CAN Current Feedback Scaling

The CAN protocol reports current as 12-bit value (0-4095). The mapping to Amps depends on the actuator model:

| Actuator Model | Current Range (A) | Scale Factor |
|----------------|-------------------|--------------|
| EC-A2806-P2-36 | -10 ~ +10 | 20.0/4095 |
| EC-A4310-P2-36 | -30 ~ +30 | 60.0/4095 |
| EC-A4315-P2-36 | -30 ~ +30 | 60.0/4095 |
| EC-A6408-P2-25 | -60 ~ +60 | 120.0/4095 |
| EC-A6416-P2-25 | -60 ~ +60 | 120.0/4095 |
| EC-A8112-P1-18 | -60 ~ +60 | 120.0/4095 |
| EC-A8116-P1-18 | -70 ~ +70 | 140.0/4095 |
| EC-A10020-P1-12/6 | -70 ~ +70 | 140.0/4095 |
| EC-A10020-P2-24 | -140 ~ +140 | 280.0/4095 |
| EC-A13720-P1-11.4 | -220 ~ +220 | 440.0/4095 |
| EC-A13715-P1-12.67 | -220 ~ +220 | 440.0/4095 |
| EC-A3814-H14-107 | -20 ~ +20 | 40.0/4095 |
| EC-A5013-H17-100 | -30 ~ +30 | 60.0/4095 |
| EC-A6013-H20-100 | -35 ~ +35 | 70.0/4095 |

**Verified Current Ranges for Asimov Motors (from 2025.11.28 protocol table):**

| Joint | Actuator | CAN Current Range | Notes |
|-------|----------|-------------------|-------|
| hip_pitch | EC-A6416-P2-25 | ±60 A | Confirmed |
| hip_roll | EC-A5013-H17-100 | ±30 A | Confirmed (was 20A in old spec) |
| hip_yaw | EC-A3814-H14-107 | ±20 A | Confirmed |
| knee | EC-A4315-P2-36 | ±30 A | Confirmed |
| ankle | EC-A4310-P2-36 | ±30 A | Confirmed |

**Note:** The 2025.11.28 protocol table supersedes earlier spec sheets. Hip Roll is ±30A (not ±20A).

## Current Decoding Formula

```python
# Generic formula
current_a = (cur_raw * (2 * MAX_CURRENT / 4095.0)) - MAX_CURRENT

# Per-joint MAX_CURRENT values (from 2025.11.28 protocol table):
CURRENT_RANGE = {
    "hip_pitch": 60.0,   # EC-A6416-P2-25
    "hip_roll": 30.0,    # EC-A5013-H17-100
    "hip_yaw": 20.0,     # EC-A3814-H14-107
    "knee": 30.0,        # EC-A4315-P2-36
    "ankle_pitch": 30.0, # EC-A4310-P2-36
    "ankle_roll": 30.0,  # EC-A4310-P2-36
}
```

---

## PD Gains (Firmware)

**Current RL-trained gains:**

| Joint | KP | KD |
|-------|-----|-----|
| hip_pitch | 12.8 | 0.8 |
| hip_roll | 328.0 | 5.0 |
| hip_yaw | 212.2 | 5.0 |
| knee | 64.2 | 2.7 |
| ankle_pitch | 19.3 | 3.3 |
| ankle_roll | 18.1 | 0.9 |

**CAN Protocol Gain Ranges:**
- KP: 0-4095 -> 0-500 (12-bit)
- KD: 0-511 -> 0-5.0 (9-bit) or 0-50 (depends on motor, see Table 9-1)

---

## Gain Identification Results

**Comparison: Identified vs Firmware Gains (with corrected current scaling):**

| Joint | Identified KP | Firmware KP | Ratio | Status |
|-------|--------------|-------------|-------|--------|
| hip_pitch | 14.3 | 12.8 | 1.12x | OK |
| hip_roll | 316.0 | 328.0 | 0.96x | OK |
| hip_yaw | 217.6 | 212.2 | 1.03x | OK |
| knee | 59.2 | 64.2 | 0.92x | OK |
| ankle_pitch | 15.0 | 19.3 | 0.78x | ~OK |
| ankle_roll | 16.4 | 18.1 | 0.91x | OK |

---

## Reflected Inertia Calculations

For physics-based gain tuning, reflected inertia at joint output:

```
J_reflected = J_rotor * gear_ratio^2
```

| Joint | J_rotor (kg·mm²) | Gear Ratio | J_reflected (kg·m²) |
|-------|------------------|------------|---------------------|
| hip_pitch | 104.395 | 25 | 0.0652 |
| hip_roll | 10 | 100 | 0.100 |
| hip_yaw | 3 | 107 | 0.0343 |
| knee | 25.5 | 36 | 0.0330 |
| ankle | 18.2 | 36 | 0.0236 |

---

## Important Notes

1. **Current scaling mismatch**: The DBC parser currently uses fixed -30~+30A range for all motors. The `analyze_motor_dynamics.py` script rescales currents to the correct per-motor ranges.

2. **Hip Yaw anomaly**: The empirically verified current range (+/-40A) differs significantly from the spec sheet (+/-10A). This requires further investigation with the motor manufacturer.

3. **Harmonic vs Planetary drives**: Hip Roll (H17) and Hip Yaw (H14) use harmonic drives with very low backlash (10 arcsec). The planetary motors (P2) have higher backlash (10-15 arcmin).

4. **Efficiency impact**: Motor efficiency varies significantly (31% for hip_roll, 73-74% for hip_pitch/knee). This affects the relationship between electrical current and mechanical torque.

5. **Rotor inertia**: All motors now have rotor inertia values. Harmonic drives from ENCOS team (EC-5013=10, EC-3814=3 kg·mm²), ankle EC-4310=18.2 kg·mm².

---

## References

- ENCOS Protocol V1.9 Manual
- Motor spec sheets:
  - EC-A6416-P2-25 (Hip Pitch)
  - EC-A5013-H17-100 (Hip Roll)
  - EC-A3814-H14-107 (Hip Yaw)
  - EC-A4315-P2-36 (Knee)
  - EC-A4310-P2-36 (Ankle)
- Table 9-1: EC Series Force/Position Mixed Control Protocol Range (2025.01.15)