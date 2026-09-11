# Eye-to-hand calibration: Lite6 xArm (moving marker) + fixed overhead camera.
#
# The camera is fixed and watches an ArUco marker rigidly mounted on the arm's TCP.
# We collect (gripper->base, marker->camera) pose pairs at several arm poses and
# solve for the fixed camera->base transform, so any future marker pose from the
# camera can be expressed in robot base coordinates.
#
# WORKFLOW (marker NOT commanded here — you jog the arm yourself):
#   1. Start the camera service on the Pi (systemd) and confirm /capture works.
#   2. Run this script.  For each pose: jog the arm with your control software to a
#      new position, keep the marker visible + in focus, then press Enter to record.
#   3. Aim for >= ~12 poses spanning different X/Y, different depths, and — most
#      importantly — different YAW/orientations (rotation diversity is what makes
#      the solve well-conditioned).  Type 'd' when done, 's' to drop the last one.
#
# Run from the Aruco_Pose directory in the arm venv (has xarm, cv2, scipy, numpy).

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from scipy.spatial.transform import Rotation

from aruco_processing import ArucoProcessor
from xarm.wrapper import XArmAPI

# ── Config ───────────────────────────────────────────────────────────────────
ROBOT_IP        = "192.168.1.169"
PI_URL          = "http://192.168.10.141:8000"
MARKER_ID       = 0               # printed marker ID (0-50)
MARKER_LENGTH_M = 0.040          # printed marker side length (metres)
NUM_CAPTURES    = 5              # high-res images averaged per pose
MIN_POSES       = 8              # refuse to solve below this; 12-15 is better
OUTPUT_NPY      = os.path.join(os.path.dirname(__file__), 'calibration_results', 'cam2base.npy')
OUTPUT_JSON     = os.path.join(os.path.dirname(__file__), 'calibration_results', 'cam2base.json')

METHODS = {
    'TSAI':       cv2.CALIB_HAND_EYE_TSAI,
    'PARK':       cv2.CALIB_HAND_EYE_PARK,
    'HORAUD':     cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF':    cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ── Pose helpers (all matrices are 4x4, translations in millimetres) ──────────

def xarm_pose_to_matrix(pose):
    """xArm [x,y,z,roll,pitch,yaw] (mm, deg) -> 4x4 base_T_gripper (gripper in base).

    xArm RPY = rotation about fixed base X (roll), Y (pitch), Z (yaw), i.e.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll) = scipy extrinsic 'xyz'. If validation shows
    a large residual, this convention is the first thing to revisit.
    """
    x, y, z, roll, pitch, yaw = pose[:6]
    R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [x, y, z]
    return T


def rtvec_to_matrix(rvec, tvec_m):
    """OpenCV (rvec, tvec in metres) -> 4x4 cam_T_marker (marker in camera), mm."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec_m.flatten() * 1000.0   # metres -> mm to match the robot
    return T


def rvec_to_rpy_deg(rvec):
    """Rodrigues rotation vector -> [roll, pitch, yaw] degrees (display/logging only)."""
    R, _ = cv2.Rodrigues(rvec)
    return Rotation.from_matrix(R).as_euler('xyz', degrees=True)


def _score(gripper2base, marker2cam, cam2base):
    """Residual metric: if cam2base (=base_T_cam) is correct, the implied
    marker->gripper transform must be the SAME for every pose (the marker is bolted
    to the TCP). Return the mean per-axis std of that translation across poses (mm)."""
    ts = []
    for G, M in zip(gripper2base, marker2cam):
        g_T_m = np.linalg.inv(G) @ cam2base @ M
        ts.append(g_T_m[:3, 3])
    return float(np.std(np.array(ts), axis=0).mean())


def solve(gripper2base, marker2cam):
    """Try both robot-pose conventions and all OpenCV methods; keep whichever gives
    the most self-consistent result (lowest marker->gripper spread). This sidesteps
    the eye-to-hand inversion/method choice by letting the data decide."""
    best = None
    for invert in (True, False):
        Ra, ta, Rb, tb = [], [], [], []
        for G, M in zip(gripper2base, marker2cam):
            A = np.linalg.inv(G) if invert else G   # invert=True -> base_T_gripper
            Ra.append(A[:3, :3]); ta.append(A[:3, 3])
            Rb.append(M[:3, :3]); tb.append(M[:3, 3])
        for name, m in METHODS.items():
            try:
                Rx, tx = cv2.calibrateHandEye(Ra, ta, Rb, tb, method=m)
            except cv2.error as e:
                print(f"  invert={invert} method={name:11s} -> failed ({e})")
                continue
            X = np.eye(4)
            X[:3, :3] = Rx
            X[:3, 3]  = np.asarray(tx).flatten()
            s = _score(gripper2base, marker2cam, X)
            print(f"  invert={invert!s:5s} method={name:11s} marker->gripper std = {s:8.3f} mm")
            if best is None or s < best[0]:
                best = (s, X, name, invert)
    return best


# ── Robot ─────────────────────────────────────────────────────────────────────

def connect_arm(ip):
    arm = XArmAPI(ip, enable_heartbeat=True)
    time.sleep(0.5)
    arm.clean_error()
    arm.clean_warn()
    # Read-only use: we never command motion here, so we don't enable/arm the robot.
    return arm


def read_pose(arm):
    """Return [x,y,z,roll,pitch,yaw] (mm, deg) or None on error."""
    ret = arm.get_position()
    code, pose = (ret if isinstance(ret, (tuple, list)) and len(ret) == 2 else (0, ret))
    if code != 0 or pose is None:
        print(f"  arm.get_position error (code={code})")
        return None
    return list(pose)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to arm...")
    arm = connect_arm(ROBOT_IP)
    print("Setting up camera...")
    proc = ArucoProcessor(pi_url=PI_URL)   # no start(); we only use /capture

    gripper2base, marker2cam, records = [], [], []

    print("\n" + "=" * 64)
    print("Jog the arm to a new pose, then press Enter to record it.")
    print("Vary YAW and DEPTH across poses. Keep marker #%d visible + in focus." % MARKER_ID)
    print("Commands:  [Enter]=record   s=drop last   d=done & solve   q=quit")
    print("=" * 64)

    while True:
        cmd = input(f"\n[{len(gripper2base)} recorded] > ").strip().lower()
        if cmd == 'q':
            print("Aborted, nothing saved."); arm.disconnect(); return
        if cmd == 'd':
            break
        if cmd == 's':
            if gripper2base:
                gripper2base.pop(); marker2cam.pop(); records.pop()
                print("  dropped last pose")
            continue

        pose = read_pose(arm)
        if pose is None:
            continue
        res = proc.capture_marker_pose_camera(MARKER_ID, NUM_CAPTURES, marker_length=MARKER_LENGTH_M)
        if res is None:
            print("  marker not detected — reposition / refocus and try again")
            continue
        rvec, tvec = res
        G = xarm_pose_to_matrix(pose)
        M = rtvec_to_matrix(rvec, tvec)
        gripper2base.append(G)
        marker2cam.append(M)

        marker_xyz = tvec.flatten() * 1000.0
        marker_rpy = rvec_to_rpy_deg(rvec)
        records.append({
            'robot_pose_mmdeg':   [round(float(v), 3) for v in pose[:6]],
            'marker_cam_xyz_mm':  [round(float(v), 3) for v in marker_xyz],
            'marker_cam_rpy_deg': [round(float(v), 3) for v in marker_rpy],
        })
        # Full 6-DOF (arm rpy + marker rotation + full marker xyz) IS captured and fed
        # to the solver below; these two lines just surface it live for sanity-checking.
        print(f"  OK  arm    xyz={np.round(pose[:3], 1).tolist()}  rpy={np.round(pose[3:6], 1).tolist()}")
        print(f"      marker xyz={np.round(marker_xyz, 1).tolist()}  rpy={np.round(marker_rpy, 1).tolist()}  (cam frame)")

    n = len(gripper2base)
    if n < MIN_POSES:
        print(f"\nOnly {n} poses (need >= {MIN_POSES}). Not solving."); arm.disconnect(); return

    # ── Rotation-diversity diagnostic (the #1 cause of a bad hand-eye solve) ──
    # Measure how much the marker's facing direction changed across poses, using the
    # angle between marker normals (its z-axis in the camera frame). Pure yaw / in-plane
    # spin leaves the normal fixed -> translation along the view axis is unobservable
    # -> nonsense cam2base translation. This is wrap-free, unlike comparing raw angles.
    normals = np.array([M[:3, 2] for M in marker2cam])
    dots = np.clip(normals @ normals.T, -1.0, 1.0)
    max_tilt = float(np.degrees(np.arccos(dots.min())))
    print(f"\nMarker tilt diversity: max angle between marker facings = {max_tilt:.0f} deg")
    if max_tilt < 25:
        print("  [WARN] the marker barely changed which way it faces (mostly yaw/in-plane).")
        print("         That is the DEGENERATE case for hand-eye — the translation solve is")
        print("         under-constrained and cam2base translation will be garbage.")
        print("         Re-collect while TILTING the marker toward different directions")
        print("         (~30-45 deg each way, about at least two axes), not just spinning it.")

    print(f"\nSolving from {n} poses...")
    best = solve(gripper2base, marker2cam)
    if best is None:
        print("All solver attempts failed."); arm.disconnect(); return

    score, cam2base, method, invert = best
    print("\n" + "=" * 64)
    print(f"Best: method={method}, invert_robot={invert}, residual={score:.3f} mm")
    print("cam2base (base_T_cam, mm):")
    print(np.array2string(cam2base, precision=3, suppress_small=True))
    cam_pos = cam2base[:3, 3]
    print(f"\nCamera position in base frame: x={cam_pos[0]:.1f}  y={cam_pos[1]:.1f}  z={cam_pos[2]:.1f} mm")
    print("  ^ sanity-check this against where the camera physically sits above the base.")
    if score > 3.0:
        print("\n[WARN] residual > 3 mm — likely too little rotation diversity, a wrong")
        print("       marker size, a shifting marker mount, or the RPY convention. Add more")
        print("       varied-orientation poses and re-run before trusting this.")

    np.save(OUTPUT_NPY, cam2base)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            'timestamp':        datetime.now().isoformat(),
            'type':             'eye_to_hand',
            'units':            'mm',
            'cam2base':         cam2base.tolist(),
            'residual_mm':      score,
            'method':           method,
            'invert_robot':     invert,
            'num_poses':        n,
            'marker_id':        MARKER_ID,
            'marker_length_m':  MARKER_LENGTH_M,
            'poses':            records,
        }, f, indent=2)
    print(f"\nSaved -> {OUTPUT_NPY}")
    print(f"        {OUTPUT_JSON}")
    arm.disconnect()


if __name__ == '__main__':
    main()
