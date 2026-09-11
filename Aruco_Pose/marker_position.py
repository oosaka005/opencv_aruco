# Terminal marker-position tool. Type a marker ID; it does a full-resolution
# capture and prints that marker's precise pose in robot BASE coordinates
# (via the hand-eye cam2base). No arm, no video — just the number.
#
# Run from Aruco_Pose in the arm venv, Pi camera service up:
#     python marker_position.py

from aruco_processing import ArucoProcessor
from hand_eye_calibration import PI_URL, MARKER_LENGTH_M, NUM_CAPTURES


def main():
    proc = ArucoProcessor(pi_url=PI_URL)
    if proc._cam2base is None:
        print("cam2base.npy not found in calibration_results — run hand_eye_calibration.py first.")
        return

    print("Type a marker ID for its base-frame pose (full-res capture). 'q' to quit.")
    while True:
        s = input("\nmarker id > ").strip().lower()
        if s == 'q':
            break
        try:
            mid = int(s)
        except ValueError:
            print("  enter an integer marker id, or 'q'"); continue

        res = proc.capture_marker_pose_camera(mid, NUM_CAPTURES, marker_length=MARKER_LENGTH_M)
        if res is None:
            print(f"  marker {mid} not detected — check it's in view / in focus"); continue

        rvec, tvec = res
        x, y, z, yaw = proc._marker_camera_to_base(rvec, tvec)
        cam = tvec.flatten() * 1000.0
        print(f"  BASE frame:    x={x:8.1f}  y={y:8.1f}  z={z:8.1f} mm    yaw={yaw:7.1f} deg")
        print(f"  camera frame:  x={cam[0]:8.1f}  y={cam[1]:8.1f}  z={cam[2]:8.1f} mm")


if __name__ == '__main__':
    main()
