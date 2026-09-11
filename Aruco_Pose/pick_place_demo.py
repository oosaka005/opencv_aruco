# Pick-and-place demo: Lite6 xArm + hand-eye calibrated overhead camera.
#
# At start it detects BOTH markers and converts them to robot base coordinates
# (via cam2base): marker 37 = pickup, marker 35 = drop-off. Each gives an x/y/yaw;
# every Z is hardcoded, and the transit midpoints + home are hardcoded. Both markers
# are REQUIRED before any motion.
#
# Cycle:
#   initial -> midpoints -> above-pick -> pick -> [close] -> above-pick
#           -> above-drop -> drop -> [open] -> above-drop -> midpoints(rev) -> initial
#
# SEPARATE offsets define where the target sits relative to each marker:
#   PICKUP_OFFSET_XY/_YAW   — cup grasp point relative to marker 37
#   DROPOFF_OFFSET_XY/_YAW  — holder seat relative to marker 35
# XY is in the marker/object frame (so it rotates with the marker); yaw is added to
# the marker's yaw. Tune them with marker_position.py until the gripper centre lands
# where you want for pickup, and the cup seats correctly at drop-off.
# NOTE: these offsets still need future work to stay exact as marker orientation
# changes (rotating vs fixed-base components) — see the reverted ref-yaw attempt.
#
# SAFETY: this MOVES the arm. It prints the full planned path and waits for explicit
# confirmation before enabling motion. Keep the e-stop in reach; start slow.
#
# Run from Aruco_Pose in the arm venv, Pi camera service up:  python pick_place_demo.py

import time
import numpy as np

from aruco_processing import ArucoProcessor
from hand_eye_calibration import ROBOT_IP, PI_URL, MARKER_LENGTH_M, NUM_CAPTURES
from xarm.wrapper import XArmAPI

# ── Markers ───────────────────────────────────────────────────────────────────
PICKUP_MARKER_ID  = 37
DROPOFF_MARKER_ID = 35

# ── Offsets: target relative to its marker (separate for pickup and drop-off) ─
PICKUP_OFFSET_XY   = (7.3, -0.6)   # (dx, dy) mm, marker/object frame — grasp vs marker 37
PICKUP_OFFSET_YAW  = 0.0          # deg, added to pickup marker yaw
DROPOFF_OFFSET_XY  = (0.0, -3.5)   # (dx, dy) mm, BASE frame (dx->X, dy->Y) — seat vs marker 35
DROPOFF_OFFSET_YAW = 0.0          # deg, added to drop-off marker yaw

# ── Motion parameters ─────────────────────────────────────────────────────────
APPROACH_SPEED = 30       # mm/s for normal movements
SLOW_SPEED     = 20       # mm/s for fine positioning (pickup/placement)
RETREAT_SPEED  = 30       # mm/s for moving away
JOINT_SPEED    = 30       # deg/s for the joint-space return to the home configuration

# ── Tool ──────────────────────────────────────────────────────────────────────
TOOL_ROLL  = -178.8        # deg — tool pointing straight down
TOOL_PITCH = 0.6

# ── Hardcoded Z heights (mm, base frame) — SET THESE to your setup ────────────
Z_ABOVE_PICK = 250.0
Z_PICK       = 163.3
Z_ABOVE_DROP = 250.0
Z_DROP       = 64.0

# ── Home + transit waypoints ([x, y, z, roll, pitch, yaw], mm/deg) ────────────
INITIAL_POSE = [169, -32, 160.0, TOOL_ROLL, TOOL_PITCH, 90]
MIDPOINTS = [                                   # traversed out, then reversed back
    [110, -129.0, 250.0, TOOL_ROLL, TOOL_PITCH, 90],
]


def _current_yaw(arm):
    """Current TCP yaw (deg) from the arm, or None."""
    ret = arm.get_position()
    pose = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else ret
    return pose[5] if pose is not None else None


def _get_joints(arm):
    """Current joint angles [J1..J6] (deg) from the arm, or None."""
    ret = arm.get_servo_angle()
    ang = ret[1] if isinstance(ret, (tuple, list)) and len(ret) == 2 else ret
    return list(ang) if ang is not None else None


def _shortest_yaw(target, current):
    """Return the equivalent of `target` (same orientation, ±360k) that is closest to
    `current` — the shortest-path wrist move — clamped into the joint-6 range
    [-360, 360]. Keeps the TRUE orientation (the grasp is not symmetric); it only
    removes redundant full turns so joint 6 takes the short way and never winds past
    its limit. Full turns are the same orientation, so this never changes the grasp."""
    y = target + 360.0 * round((current - target) / 360.0)   # nearest equivalent to current
    if y > 360.0:                                            # stay within J6 +/-360
        y -= 360.0
    elif y < -360.0:
        y += 360.0
    return y


def connect_arm(ip):
    arm = XArmAPI(ip, enable_heartbeat=True)
    time.sleep(0.5)
    arm.clean_error(); arm.clean_warn()
    arm.set_mode(0);      time.sleep(0.3)   # position mode
    arm.motion_enable(True); time.sleep(0.3)
    arm.set_state(0);     time.sleep(0.5)   # ready
    return arm


def move(arm, pose, label, speed=APPROACH_SPEED):
    cur = _current_yaw(arm)
    yaw = pose[5] if cur is None else _shortest_yaw(pose[5], cur)   # shortest path, within J6 +/-360
    print(f"  -> {label:14s} {[round(v, 1) for v in pose[:5]] + [round(yaw, 1)]}  @ {speed} mm/s")
    code = arm.set_position(x=pose[0], y=pose[1], z=pose[2],
                            roll=pose[3], pitch=pose[4], yaw=yaw,
                            speed=speed, wait=True)
    if code != 0:
        raise RuntimeError(f"set_position failed (code {code}) at '{label}'")


def target_from_marker(proc, marker_id, offset_xy, offset_yaw, offset_in_base=False):
    """Detect marker_id, map to base frame, apply its offset. Returns
    ((x, y, yaw) target, (x, y, yaw) raw marker); None if not seen.

    offset_in_base=False: offset is in the marker/object frame (rotates with the
      marker's yaw) — correct for a movable object like the cup on pickup.
    offset_in_base=True:  offset is applied straight in base X/Y (no rotation) —
      intuitive when the marker sits at a fixed, non-zero yaw (e.g. the holder)."""
    res = proc.capture_marker_pose_camera(marker_id, NUM_CAPTURES, marker_length=MARKER_LENGTH_M)
    if res is None:
        return None
    b = proc._marker_camera_to_base(*res)          # (x, y, z, yaw) base frame
    if b is None:
        print("  cam2base not loaded — run hand_eye_calibration.py first."); return None
    mx, my, _mz, myaw = b
    dx, dy = offset_xy
    if offset_in_base:
        tx, ty = mx + dx, my + dy                                   # base-frame: dx->X, dy->Y
    else:
        c, s = np.cos(np.radians(myaw)), np.sin(np.radians(myaw))   # rotate offset into base
        tx = mx + c * dx - s * dy
        ty = my + s * dx + c * dy
    tyaw = myaw + offset_yaw          # true orientation; shortest-path is handled at move time
    return (tx, ty, tyaw), (mx, my, myaw)


def main():
    print("Setting up camera...")
    proc = ArucoProcessor(pi_url=PI_URL)

    print(f"Detecting pickup marker #{PICKUP_MARKER_ID} and drop-off marker #{DROPOFF_MARKER_ID} "
          "(both required before any motion)...")
    pick_t = target_from_marker(proc, PICKUP_MARKER_ID, PICKUP_OFFSET_XY, PICKUP_OFFSET_YAW)
    if pick_t is None:
        print(f"Pickup marker {PICKUP_MARKER_ID} not detected — aborting, arm never engaged."); return
    drop_t = target_from_marker(proc, DROPOFF_MARKER_ID, DROPOFF_OFFSET_XY, DROPOFF_OFFSET_YAW,
                                offset_in_base=True)
    if drop_t is None:
        print(f"Drop-off marker {DROPOFF_MARKER_ID} not detected — aborting, arm never engaged."); return

    (px, py, pyaw), (pmx, pmy, pmyaw) = pick_t
    (dx_, dy_, dyaw), (dmx, dmy, dmyaw) = drop_t
    print(f"  pickup  marker(base) x={pmx:.1f} y={pmy:.1f} yaw={pmyaw:.1f}  ->  target x={px:.1f} y={py:.1f} yaw={pyaw:.1f}")
    print(f"  dropoff marker(base) x={dmx:.1f} y={dmy:.1f} yaw={dmyaw:.1f}  ->  target x={dx_:.1f} y={dy_:.1f} yaw={dyaw:.1f}")

    above_pick = [px, py, Z_ABOVE_PICK, TOOL_ROLL, TOOL_PITCH, pyaw]
    pick       = [px, py, Z_PICK,       TOOL_ROLL, TOOL_PITCH, pyaw]
    above_drop = [dx_, dy_, Z_ABOVE_DROP, TOOL_ROLL, TOOL_PITCH, dyaw]
    drop       = [dx_, dy_, Z_DROP,       TOOL_ROLL, TOOL_PITCH, dyaw]

    print("\nPlanned cycle:")
    for p, lbl in ([(INITIAL_POSE, "initial")] +
                   [(mp, f"midpoint {i+1}") for i, mp in enumerate(MIDPOINTS)] +
                   [(above_pick, "above pick"), (pick, "PICK (close)"), (above_pick, "above pick"),
                    (above_drop, "above drop"), (drop, "DROP (open)"), (above_drop, "above drop")] +
                   [(mp, f"midpoint {len(MIDPOINTS)-i}") for i, mp in enumerate(reversed(MIDPOINTS))] +
                   [(INITIAL_POSE, "initial")]):
        print(f"    {lbl:14s} {[round(v, 1) for v in p]}")

    if input("\nProceed and MOVE the arm? [y/N] ").strip().lower() != 'y':
        print("Aborted, no motion."); return

    print("\nConnecting + enabling arm...")
    arm = connect_arm(ROBOT_IP)
    try:
        move(arm, INITIAL_POSE, "initial", APPROACH_SPEED)
        home_angles = _get_joints(arm)   # capture the home joint config (incl. J6) to return to
        for i, mp in enumerate(MIDPOINTS):
            move(arm, mp, f"midpoint {i+1}", APPROACH_SPEED)
        move(arm, above_pick, "above pick", APPROACH_SPEED)
        print("  [gripper] open"); arm.open_lite6_gripper(); time.sleep(0.6)
        move(arm, pick, "pick", SLOW_SPEED)
        print("  [gripper] close"); arm.close_lite6_gripper(); time.sleep(0.9)
        move(arm, above_pick, "above pick", RETREAT_SPEED)
        move(arm, above_drop, "above drop", APPROACH_SPEED)
        move(arm, drop, "drop", SLOW_SPEED)
        print("  [gripper] open"); arm.open_lite6_gripper(); time.sleep(0.9)
        move(arm, above_drop, "above drop", RETREAT_SPEED)
        for i, mp in enumerate(reversed(MIDPOINTS)):
            move(arm, mp, f"midpoint {len(MIDPOINTS) - i}", APPROACH_SPEED)
        # Return to the EXACT starting joint configuration so joint 6 resets to where
        # it began and can't accumulate across cycles — rather than a Cartesian move
        # whose IK might pick a wound wrist solution.
        if home_angles is not None:
            print(f"  -> initial (joints) {[round(a, 1) for a in home_angles]}  @ {JOINT_SPEED} deg/s")
            code = arm.set_servo_angle(angle=home_angles, speed=JOINT_SPEED, wait=True)
            if code != 0:
                raise RuntimeError(f"set_servo_angle failed (code {code}) returning to initial joints")
        else:
            move(arm, INITIAL_POSE, "initial", APPROACH_SPEED)
        print("\nCycle complete.")
    except Exception as e:
        print(f"\n[ERROR] {e}\n  stopping arm (set_state 4).")
        arm.set_state(4)
    finally:
        arm.stop_lite6_gripper()
        arm.disconnect()
        print("Disconnected.")


if __name__ == '__main__':
    main()
