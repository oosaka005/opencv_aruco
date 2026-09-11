# Two-stage calibration check (separate from hand_eye_verify.py):
#
#   STAGE 1 - PRECISION: hold the marker completely still and capture repeatedly.
#       Measures pure camera/detection repeatability at 2304 (position mm +
#       orientation deg). If this is ~0.1 mm, it proves the ~1 mm seen across the
#       workspace is robot/calibration, not the camera.
#
#   STAGE 2 - ROTATION ACCURACY: rotate the wrist (yaw) between captures. Compares
#       the camera-measured rotation change against the robot's commanded rotation
#       change. The marker->TCP offset cancels in the relative rotation, so this is
#       a clean, offset-independent check of rotational accuracy.
#
# Reuses transforms/config from hand_eye_calibration.py (one level up). Run from
# Aruco_Pose/Debugging/ in the arm venv, Pi camera service up:  python precision_rotation_test.py

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# This file lives in Aruco_Pose/Debugging/ — add the parent Aruco_Pose/ to the import
# path so aruco_processing and hand_eye_calibration (one level up) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aruco_processing import ArucoProcessor
from hand_eye_calibration import (
    ROBOT_IP, PI_URL, MARKER_ID, MARKER_LENGTH_M, NUM_CAPTURES, OUTPUT_NPY,
    xarm_pose_to_matrix, rtvec_to_matrix, connect_arm, read_pose,
)

N_PRECISION = 10   # number of captures for the fixed-pose precision stage


def capture_marker_base(proc, cam2base):
    """Return (marker_base_xyz (3,), R_marker_base (3x3)) or None."""
    res = proc.capture_marker_pose_camera(MARKER_ID, NUM_CAPTURES, marker_length=MARKER_LENGTH_M)
    if res is None:
        return None
    rvec, tvec = res
    base_T_marker = cam2base @ rtvec_to_matrix(rvec, tvec)
    return base_T_marker[:3, 3], base_T_marker[:3, :3]


def stage1_precision(proc, cam2base):
    print("\n" + "=" * 66)
    print(f"STAGE 1 - PRECISION.  Keep the marker COMPLETELY STILL (do not move the arm).")
    print(f"Press Enter for each of {N_PRECISION} captures.   q = quit")
    print("=" * 66)
    positions, rmats = [], []
    while len(positions) < N_PRECISION:
        cmd = input(f"  [{len(positions)}/{N_PRECISION}] Enter=capture > ").strip().lower()
        if cmd == 'q':
            return False
        c = capture_marker_base(proc, cam2base)
        if c is None:
            print("  marker not detected — reposition / refocus"); continue
        positions.append(c[0]); rmats.append(c[1])
        print(f"    marker base xyz = {np.round(c[0], 2).tolist()} mm")

    positions = np.array(positions)
    pos_std = positions.std(axis=0)
    dev = np.linalg.norm(positions - positions.mean(axis=0), axis=1)
    rots = Rotation.from_matrix(np.array(rmats))
    ang_dev = np.degrees((rots * rots.mean().inv()).magnitude())

    print("\n  --- PRECISION RESULTS (camera repeatability at a fixed pose) ---")
    print(f"  position std (mm):    x={pos_std[0]:.3f}  y={pos_std[1]:.3f}  z={pos_std[2]:.3f}")
    print(f"  position max dev:     {dev.max():.3f} mm (euclidean from mean)")
    print(f"  orientation dev:      mean={ang_dev.mean():.3f} deg   max={ang_dev.max():.3f} deg")
    print("  (position std ~0.1mm => camera is not the bottleneck for the workspace error)")
    return True


def stage2_rotation(arm, proc, cam2base):
    print("\n" + "=" * 66)
    print("STAGE 2 - ROTATION ACCURACY.  Rotate the wrist (yaw) to a NEW angle, then Enter.")
    print("Do several (~5-8) spanning your yaw range.   d = done")
    print("=" * 66)
    R_marker, R_tcp, yaws = [], [], []
    while True:
        cmd = input(f"  [{len(R_marker)} recorded] Enter=capture  d=done > ").strip().lower()
        if cmd == 'd':
            break
        pose = read_pose(arm)
        if pose is None:
            continue
        c = capture_marker_base(proc, cam2base)
        if c is None:
            print("  marker not detected — reposition / refocus"); continue
        G = xarm_pose_to_matrix(pose)
        R_marker.append(c[1]); R_tcp.append(G[:3, :3]); yaws.append(pose[5])

        if len(R_marker) == 1:
            print(f"    reference orientation set (robot yaw = {pose[5]:.1f} deg)")
        else:
            # Rotation each underwent since the first pose. The constant marker->TCP
            # offset cancels, so camera and robot deltas should match.
            dR_m = R_marker[-1] @ R_marker[0].T          # camera-measured rotation change
            dR_t = R_tcp[-1] @ R_tcp[0].T                # robot commanded rotation change
            ang_m = np.degrees(Rotation.from_matrix(dR_m).magnitude())
            ang_t = np.degrees(Rotation.from_matrix(dR_t).magnitude())
            err   = np.degrees(Rotation.from_matrix(dR_m.T @ dR_t).magnitude())
            print(f"    robot rotated {ang_t:6.2f} deg | camera saw {ang_m:6.2f} deg | "
                  f"error {err:5.2f} deg   (yaw={pose[5]:.1f})")

    if len(R_marker) >= 2:
        errs = [np.degrees(Rotation.from_matrix(
                    (R_marker[i] @ R_marker[0].T).T @ (R_tcp[i] @ R_tcp[0].T)).magnitude())
                for i in range(1, len(R_marker))]
        errs = np.array(errs)
        print("\n  --- ROTATION ACCURACY RESULTS (camera vs robot, relative to first pose) ---")
        print(f"  rotation error:   mean={errs.mean():.2f} deg   max={errs.max():.2f} deg")
        print("  (<~1 deg is good; large error = calibration rotation or RPY convention issue)")


def main():
    if not os.path.exists(OUTPUT_NPY):
        print(f"No calibration at {OUTPUT_NPY} — run hand_eye_calibration.py first."); return
    cam2base = np.load(OUTPUT_NPY)
    print("Loaded cam2base.")

    print("Connecting to arm...")
    arm = connect_arm(ROBOT_IP)
    print("Setting up camera...")
    proc = ArucoProcessor(pi_url=PI_URL)

    if stage1_precision(proc, cam2base):
        stage2_rotation(arm, proc, cam2base)

    arm.disconnect()
    print("\nDone.")


if __name__ == '__main__':
    main()
