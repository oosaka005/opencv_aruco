# Verify the eye-to-hand calibration: load cam2base and report the marker's
# position in ROBOT BASE coordinates as you move the arm.
#
# It reuses the exact transforms/config from hand_eye_calibration.py (so the xArm
# RPY convention and marker settings can't drift), and prints three things per pose:
#   1. Marker position in base frame  = cam2base @ (marker in camera)   <- the goal
#   2. Robot TCP position in base frame (from the arm) for comparison
#   3. The implied TCP->marker offset, whose consistency across poses is the health check
#
# It also compares pose-to-pose DELTAS: if you move the arm WITHOUT rotating, the
# camera-derived marker delta must match the robot's TCP delta (same axis, sign,
# magnitude). That is the independent ground-truth test that catches a wrong global
# frame/convention — something the calibration residual alone cannot.
#
# Run from Aruco_Pose/Debugging/ in the arm venv, with the Pi camera service up:
#     python hand_eye_verify.py

import os
import sys
import numpy as np

# This file lives in Aruco_Pose/Debugging/ — add the parent Aruco_Pose/ to the import
# path so aruco_processing and hand_eye_calibration (one level up) resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aruco_processing import ArucoProcessor
from hand_eye_calibration import (
    ROBOT_IP, PI_URL, MARKER_ID, MARKER_LENGTH_M, NUM_CAPTURES, OUTPUT_NPY,
    xarm_pose_to_matrix, rtvec_to_matrix, connect_arm, read_pose,
)


def main():
    if not os.path.exists(OUTPUT_NPY):
        print(f"No calibration found at {OUTPUT_NPY} — run hand_eye_calibration.py first.")
        return
    cam2base = np.load(OUTPUT_NPY)   # base_T_cam, mm
    print("Loaded cam2base (base_T_cam, mm):")
    print(np.array2string(cam2base, precision=3, suppress_small=True))

    print("\nConnecting to arm...")
    arm = connect_arm(ROBOT_IP)
    print("Setting up camera...")
    proc = ArucoProcessor(pi_url=PI_URL)

    print("\n" + "=" * 66)
    print(f"Move the arm, then press Enter to read marker #{MARKER_ID} in base frame.")
    print("For the MOTION TEST: move WITHOUT rotating the wrist and check that")
    print("  'delta marker' matches 'delta TCP'.   q = quit")
    print("=" * 66)

    offsets = []      # implied gripper->marker translations (consistency check)
    prev = None       # (tcp_base_xyz, marker_base_xyz) from the previous pose

    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == 'q':
            break

        pose = read_pose(arm)
        if pose is None:
            continue
        res = proc.capture_marker_pose_camera(MARKER_ID, NUM_CAPTURES, marker_length=MARKER_LENGTH_M)
        if res is None:
            print("  marker not detected — reposition / refocus and try again")
            continue
        rvec, tvec = res

        G = xarm_pose_to_matrix(pose)        # base_T_gripper (from robot)
        M = rtvec_to_matrix(rvec, tvec)      # cam_T_marker  (from camera)
        base_T_marker = cam2base @ M         # marker expressed in base frame
        marker_base = base_T_marker[:3, 3]
        tcp_base    = np.array(pose[:3], dtype=float)
        offset      = (np.linalg.inv(G) @ base_T_marker)[:3, 3]   # gripper->marker
        offsets.append(offset)

        print(f"  Marker  (base, from camera): x={marker_base[0]:8.1f}  y={marker_base[1]:8.1f}  z={marker_base[2]:8.1f}  mm")
        print(f"  Robot TCP (base):            x={tcp_base[0]:8.1f}  y={tcp_base[1]:8.1f}  z={tcp_base[2]:8.1f}  mm")
        print(f"  Implied TCP->marker offset:  {np.round(offset, 1).tolist()}  (|offset| = {np.linalg.norm(offset):.1f} mm)")

        if prev is not None:
            d_tcp    = tcp_base - prev[0]
            d_marker = marker_base - prev[1]
            mismatch = d_marker - d_tcp
            print(f"  delta vs last  TCP: {np.round(d_tcp, 1).tolist()}   "
                  f"marker: {np.round(d_marker, 1).tolist()}")
            print(f"                 mismatch: {np.round(mismatch, 1).tolist()} mm  "
                  f"(|.|={np.linalg.norm(mismatch):.1f}) -> ~0 if you did NOT rotate")
        prev = (tcp_base, marker_base)

        if len(offsets) >= 2:
            std = np.std(np.array(offsets), axis=0)
            print(f"  offset consistency over {len(offsets)} poses: std = "
                  f"{np.round(std, 2).tolist()} mm  (low = good calibration)")

    arm.disconnect()
    print("Disconnected.")


if __name__ == '__main__':
    main()
