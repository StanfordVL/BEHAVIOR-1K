# 90° Wrist-Adapter Experiment — Revert Checkpoint

## State at checkpoint (2026-05-07)

Goal: rotate the SharpaWave hand 90° relative to `panda_link7` via a URDF
edit (path A from the user's question — pure transform, no new mesh).

### Files snapshotted in this directory

- `franka_mounted_sharpa_right.urdf.before`
  Full copy of the live URDF before any edits, taken from
  `/home/eeg/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/urdf/franka_mounted_sharpa_right.urdf`

### Current value of the mount joint (URDF lines ~1051-1055)

```xml
<joint name="panda_right_hand_C_MC_joint" type="fixed">
    <parent link="panda_link7" />
    <child link="right_hand_C_MC" />
    <origin xyz="0.0 0.0 0.107" rpy="0.0 0.0 0.0" />
</joint>
```

### Revert command

```bash
cp /home/eeg/Desktop/BEHAVIOR-1K/hand_teleop/_revert_state/franka_mounted_sharpa_right.urdf.before \
   /home/eeg/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/urdf/franka_mounted_sharpa_right.urdf
# If USD was rebuilt, delete the rebuilt one to force re-import on next launch:
# rm /home/eeg/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda
# (also restore the .before USD if needed — not snapshotted by default because
#  it's 175 MB; only delete if you can re-import.)
```

### What changed at this checkpoint

**Goal:** add a virtual 90°-around-Y rotation between `panda_link7` and
`right_hand_C_MC` (analogous to a physical 90° wrist adapter — tilts the
hand forward).

**Edits:**

1. URDF (`...franka_mounted_sharpa_right.urdf`, lines ~1051-1055):
   ```diff
   - <origin xyz="0.0 0.0 0.107" rpy="0.0 0.0 0.0" />
   + <origin xyz="0.0 0.0 0.107" rpy="0.0 1.5708 0.0" />
   ```
2. USD (`...franka_mounted_sharpa_right.usda`, line ~2414, inside the
   `panda_right_hand_C_MC_joint` def):
   ```diff
   - quatf physics:localRot0 = (1, 0, 0, 0)
   + quatf physics:localRot0 = (0.70710678, 0, 0.70710678, 0)
   ```
   (USD quaternion format is **wxyz**, identity = `(1,0,0,0)`,
    90° around Y = `(cos45°, 0, sin45°, 0)`.)

**To revert without rebuilding the USD:**

```bash
cp /home/eeg/Desktop/BEHAVIOR-1K/hand_teleop/_revert_state/franka_mounted_sharpa_right.urdf.before \
   /home/eeg/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/urdf/franka_mounted_sharpa_right.urdf

# Restore the single USD line in-place (no full backup needed since edit is one line):
sed -i 's/quatf physics:localRot0 = (0\.70710678, 0, 0\.70710678, 0)/quatf physics:localRot0 = (1, 0, 0, 0)/' \
   /home/eeg/Desktop/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda
```

The USD line is unique (joint is named uniquely + the surrounding context
of `panda_right_hand_C_MC_joint`) so the sed is safe.
