# ArUco Marker Processing for Pose Estimation & Stream Overlay

import cv2
import numpy as np
import time
import os
import json
import urllib.request
from threading import Thread, Lock
from datetime import datetime
import math

# ── Constants ──────────────────────────────────────────────────────────────────

PI_URL = "http://192.168.10.141:8000"

# Must match camera_capture_backend.py — used to scale the camera matrix for preview frames.
# Calibration is now done at CAPTURE_SIZE, so camera_matrix is valid at capture res directly.
PREVIEW_SIZE = (854, 480)
CAPTURE_SIZE = (2304, 1296)

MARKER_LENGTH       = 0.040    # printed side length of the (non-board) pose/TCP marker, metres
BOARD_SQUARES_H     = 4
BOARD_SQUARES_V     = 3
BOARD_SQUARE_LENGTH = 0.0381
BOARD_MARKER_LENGTH = 0.029575
BOARD_NUM_MARKERS   = (BOARD_SQUARES_H * BOARD_SQUARES_V) // 2
BOARD_MARKER_IDS    = set(range(BOARD_NUM_MARKERS))

# ── Rotation utilities ─────────────────────────────────────────────────────────

def _geodesic_distance(R1, R2):
    R_rel = R1.T @ R2
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return float(np.linalg.norm(rvec_rel))

def _slerp_step(R_prev, R_new, alpha):
    R_rel = R_prev.T @ R_new
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    R_delta, _ = cv2.Rodrigues(rvec_rel * alpha)
    return R_prev @ R_delta

def _mean_rotation(rvec_list):
    """SO(3) mean: average rotation matrices then project back via SVD."""
    R_stack = np.array([cv2.Rodrigues(r.reshape(3, 1))[0] for r in rvec_list])
    U, _, Vt = np.linalg.svd(R_stack.mean(axis=0))
    return U @ Vt

def rvec_to_rpy(rvec):
    """Convert a Rodrigues rotation vector to roll/pitch/yaw (degrees, XYZ convention)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return {
        "roll_deg":  np.degrees(roll),
        "pitch_deg": np.degrees(pitch),
        "yaw_deg":   np.degrees(yaw),
    }

def transform_pose_to_board_frame(rvec_marker, tvec_marker, rvec_board, tvec_board):
    """Transform a marker pose from camera frame into board (world) frame."""
    R_board, _ = cv2.Rodrigues(rvec_board)
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R_board
    T_cam_board[:3, 3]  = tvec_board.flatten()

    R_marker, _ = cv2.Rodrigues(rvec_marker)
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = R_marker
    T_cam_marker[:3, 3]  = tvec_marker.flatten()

    T_board_marker = np.linalg.inv(T_cam_board) @ T_cam_marker
    rvec_bf, _ = cv2.Rodrigues(T_board_marker[:3, :3])
    tvec_bf    = T_board_marker[:3, 3:4]
    return rvec_bf, tvec_bf

# ── Filter classes ─────────────────────────────────────────────────────────────

class ExponentialPoseFilter:
    """EMA filter: linear EMA for translation, geodesic SLERP for rotation.
    Resets on 2-second stale timeout or after 3 consecutive outliers."""

    def __init__(self, alpha=0.10, rotation_threshold_deg=10.0):
        self.alpha         = alpha
        self.rot_threshold = np.radians(rotation_threshold_deg)
        self.tvec_prev     = None
        self.R_prev        = None
        self.last_update   = time.time()
        self.outlier_count = 0

    def smooth(self, rvec, tvec):
        now      = time.time()
        R_new, _ = cv2.Rodrigues(rvec)

        if self.tvec_prev is None or (now - self.last_update) > 2.0 or self.outlier_count > 3:
            self.tvec_prev     = tvec.copy()
            self.R_prev        = R_new.copy()
            self.last_update   = now
            self.outlier_count = 0
            return rvec, tvec

        tvec_jump = np.linalg.norm(tvec - self.tvec_prev) > 0.05
        rot_jump  = _geodesic_distance(self.R_prev, R_new) > self.rot_threshold
        if tvec_jump or rot_jump:
            self.outlier_count += 1
            self.last_update    = now
            rvec_prev, _ = cv2.Rodrigues(self.R_prev)
            return rvec_prev, self.tvec_prev

        self.outlier_count = 0
        tvec_smooth = self.alpha * tvec + (1 - self.alpha) * self.tvec_prev
        self.tvec_prev = tvec_smooth.copy()
        self.R_prev    = _slerp_step(self.R_prev, R_new, self.alpha)
        rvec_smooth, _ = cv2.Rodrigues(self.R_prev)
        self.last_update = now
        return rvec_smooth, tvec_smooth


class MovingAverageFilter:
    """SMA filter: arithmetic mean for translation, SVD-projected mean for rotation.
    Resets on 2-second stale timeout or after 3 consecutive outliers."""

    def __init__(self, window_size=20, rotation_threshold_deg=10.0):
        self.window_size   = window_size
        self.rot_threshold = np.radians(rotation_threshold_deg)
        self._tvec_buf     = []
        self._R_buf        = []
        self.last_update   = time.time()
        self.outlier_count = 0

    def _current_means(self):
        tvec_avg = np.mean(self._tvec_buf, axis=0)
        U, _, Vt = np.linalg.svd(np.mean(self._R_buf, axis=0))
        return tvec_avg, U @ Vt

    def smooth(self, rvec, tvec):
        now      = time.time()
        R_new, _ = cv2.Rodrigues(rvec)

        if not self._tvec_buf or (now - self.last_update) > 2.0 or self.outlier_count > 3:
            self._tvec_buf     = [tvec.flatten().copy()]
            self._R_buf        = [R_new.copy()]
            self.last_update   = now
            self.outlier_count = 0
            return rvec, tvec

        tvec_avg, R_avg = self._current_means()
        tvec_jump = np.linalg.norm(tvec.flatten() - tvec_avg) > 0.05
        rot_jump  = _geodesic_distance(R_avg, R_new) > self.rot_threshold
        if tvec_jump or rot_jump:
            self.outlier_count += 1
            self.last_update    = now
            rvec_avg, _ = cv2.Rodrigues(R_avg)
            return rvec_avg, tvec_avg.reshape(tvec.shape)

        self.outlier_count = 0
        self._tvec_buf.append(tvec.flatten().copy())
        self._R_buf.append(R_new.copy())
        if len(self._tvec_buf) > self.window_size:
            self._tvec_buf.pop(0)
            self._R_buf.pop(0)
        self.last_update = now

        tvec_out, R_out = self._current_means()
        rvec_out, _ = cv2.Rodrigues(R_out)
        return rvec_out, tvec_out.reshape(tvec.shape)

# ── Main processor class ───────────────────────────────────────────────────────

class ArucoProcessor:
    """Encapsulates all ArUco detection, pose estimation, filtering, and capture logic.

    Detection loop polls the Pi's /preview endpoint for live annotated frames.
    capture_pose() fetches N high-res images from /capture and averages fresh detections.

    Usage:
        processor = ArucoProcessor()
        processor.start()
        frame  = processor.get_frame()
        result = processor.capture_pose(marker_id)
        processor.stop()
    """

    def __init__(self, pi_url=PI_URL, calibration_dir=None, pose_file=None):
        base = os.path.dirname(__file__)
        if calibration_dir is None:
            calibration_dir = os.path.join(base, 'calibration_results')
        if pose_file is None:
            pose_file = os.path.join(base, 'pose_calibration.json')

        self._pi_url   = pi_url
        self.pose_file = pose_file

        cam_path  = os.path.join(calibration_dir, 'camera_matrix.npy')
        dist_path = os.path.join(calibration_dir, 'dist_coeffs.npy')
        if not os.path.exists(cam_path) or not os.path.exists(dist_path):
            raise FileNotFoundError(
                f"Calibration files not found in '{calibration_dir}'. Run camera_calibration.py first."
            )
        self.camera_matrix = np.load(cam_path)   # valid at CAPTURE_SIZE
        self.dist_coeffs   = np.load(dist_path)

        # Hand-eye transform (base_T_cam, mm). Optional — if present, marker poses
        # are also reported in robot base coordinates (live overlay + capture_pose).
        cam2base_path = os.path.join(calibration_dir, 'cam2base.npy')
        self._cam2base = np.load(cam2base_path) if os.path.exists(cam2base_path) else None

        # Camera matrix scaled to PREVIEW_SIZE — used when processing 854×480 preview frames.
        # Distortion coefficients are resolution-independent so they stay as-is.
        sx = PREVIEW_SIZE[0] / CAPTURE_SIZE[0]
        sy = PREVIEW_SIZE[1] / CAPTURE_SIZE[1]
        self._preview_matrix = self.camera_matrix.copy()
        self._preview_matrix[0, :] *= sx
        self._preview_matrix[1, :] *= sy

        # ArUco detector
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # ChArUco board
        self._charuco_board    = cv2.aruco.CharucoBoard(
            (BOARD_SQUARES_H, BOARD_SQUARES_V),
            BOARD_SQUARE_LENGTH, BOARD_MARKER_LENGTH, aruco_dict
        )
        self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)

        # Shared state
        self.processed_frame     = None
        self.frame_lock          = Lock()
        self.current_poses       = {}
        self.current_poses_board = {}
        self.board_pose          = None
        self.poses_lock          = Lock()
        self.board_pose_lock     = Lock()

        self._pose_filters  = {}
        self._board_filters = {}
        self._running = False

    def _marker_camera_to_base(self, rvec, tvec):
        """Map a marker pose in the camera frame (rvec, tvec in metres) to the robot
        base frame via cam2base. Returns (x, y, z mm, yaw deg) or None if cam2base
        isn't loaded."""
        if self._cam2base is None:
            return None
        R, _ = cv2.Rodrigues(rvec)
        cam_T_marker = np.eye(4)
        cam_T_marker[:3, :3] = R
        cam_T_marker[:3, 3]  = tvec.flatten() * 1000.0   # m -> mm
        base = self._cam2base @ cam_T_marker
        x, y, z = base[:3, 3]
        yaw = rvec_to_rpy(cv2.Rodrigues(base[:3, :3])[0])['yaw_deg']
        return float(x), float(y), float(z), float(yaw)

    # ── Filter management ──────────────────────────────────────────────────────

    def _get_or_create_filter(self, marker_id, marker_type='marker',
                               filter_type='ema', alpha=0.10, window_size=20):
        registry = self._board_filters if marker_type == 'board' else self._pose_filters
        if marker_id not in registry:
            registry[marker_id] = (
                MovingAverageFilter(window_size=window_size)
                if filter_type == 'sma'
                else ExponentialPoseFilter(alpha=alpha)
            )
        return registry[marker_id]

    # ── Board detection ────────────────────────────────────────────────────────

    def _detect_board(self, frame, camera_matrix=None, rvec_init=None, tvec_init=None):
        """Detect ChArUco board; returns (rvec, tvec) in camera frame or (None, None)."""
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        charuco_corners, charuco_ids, _, _ = self._charuco_detector.detectBoard(frame)
        if charuco_ids is None or len(charuco_ids) <= 4:
            return None, None
        obj_points, img_points = self._charuco_board.matchImagePoints(charuco_corners, charuco_ids)
        if len(obj_points) == 0:
            return None, None
        use_guess = rvec_init is not None and tvec_init is not None
        success, rvec_o, tvec_o = cv2.solvePnP(
            obj_points, img_points,
            camera_matrix, self.dist_coeffs,
            rvec=rvec_init.copy() if use_guess else None,
            tvec=tvec_init.copy() if use_guess else None,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec_o, tvec_o) if success else (None, None)

    # ── Single-frame processing (preview) ──────────────────────────────────────

    def _process_frame(self, frame):
        """Run board + marker detection on one preview frame using the scaled camera matrix."""
        cam_mtx = self._preview_matrix

        with self.board_pose_lock:
            prior = self.board_pose
        rvec_init = prior[0] if prior is not None else None
        tvec_init = prior[1] if prior is not None else None

        rvec_board, tvec_board = self._detect_board(frame, cam_mtx, rvec_init, tvec_init)
        board_detected = rvec_board is not None

        if board_detected:
            board_filter = self._get_or_create_filter('board', marker_type='board')
            rvec_board_f, tvec_board_f = board_filter.smooth(rvec_board, tvec_board)
            with self.board_pose_lock:
                self.board_pose = (rvec_board_f.copy(), tvec_board_f.copy())
            cv2.drawFrameAxes(frame, cam_mtx, self.dist_coeffs,
                              rvec_board_f, tvec_board_f, BOARD_SQUARE_LENGTH / 2, thickness=3)

        marker_corners, marker_ids, _ = self._detector.detectMarkers(frame)
        if marker_ids is None:
            with self.frame_lock:
                self.processed_frame = frame.copy()
            return

        for i, marker_id in enumerate(marker_ids):
            mid     = marker_id[0]
            img_pts = marker_corners[i][0].astype(np.float32)
            m_len   = BOARD_MARKER_LENGTH if mid in BOARD_MARKER_IDS else MARKER_LENGTH
            obj_pts = np.array([
                [-m_len/2,  m_len/2, 0], [ m_len/2,  m_len/2, 0],
                [ m_len/2, -m_len/2, 0], [-m_len/2, -m_len/2, 0],
            ], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, cam_mtx, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                continue
            pose_filter = self._get_or_create_filter(mid)
            rvec_s, tvec_s = pose_filter.smooth(rvec, tvec)
            with self.poses_lock:
                self.current_poses[mid] = {
                    'rvec': rvec_s, 'tvec': tvec_s,
                    'timestamp': datetime.now().isoformat()
                }
                if board_detected:
                    rvec_bf, tvec_bf = transform_pose_to_board_frame(
                        rvec_s, tvec_s, rvec_board_f, tvec_board_f
                    )
                    self.current_poses_board[mid] = {
                        'rvec': rvec_bf, 'tvec': tvec_bf,
                        'timestamp': datetime.now().isoformat()
                    }
            cv2.drawFrameAxes(frame, cam_mtx, self.dist_coeffs,
                              rvec_s, tvec_s, MARKER_LENGTH / 2)

        with self.frame_lock:
            self.processed_frame = frame.copy()

    # ── Detection loop ─────────────────────────────────────────────────────────

    def _detection_loop(self):
        while self._running:
            try:
                with urllib.request.urlopen(f"{self._pi_url}/preview", timeout=5) as resp:
                    frame = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    self._process_frame(frame)
            except Exception as e:
                print(f"Preview fetch error: {e}")
                time.sleep(1)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the detection loop in a daemon thread. Returns the thread."""
        self._running = True
        t = Thread(target=self._detection_loop, daemon=True, name="ArucoDetection")
        t.start()
        return t

    def stop(self):
        self._running = False

    def get_frame(self):
        """Return a copy of the latest annotated preview frame, or None if not yet available."""
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None

    def capture_marker_pose_camera(self, marker_id, num_captures=5, marker_length=None):
        """Fetch num_captures high-res images, detect `marker_id`, and return its averaged
        pose in the CAMERA frame as (rvec, tvec): rvec Rodrigues (3,1), tvec (3,1) in metres.
        Returns None if the marker isn't found in at least half the images. Does not save.

        Used by hand_eye_calibration.py — kept separate from capture_pose() so the working
        pose-capture path stays untouched. Pass marker_length (metres) for a marker whose
        size differs from the module defaults.
        """
        if marker_length is None:
            marker_length = BOARD_MARKER_LENGTH if marker_id in BOARD_MARKER_IDS else MARKER_LENGTH
        obj_pts = np.array([
            [-marker_length/2,  marker_length/2, 0], [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0], [-marker_length/2, -marker_length/2, 0],
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for i in range(num_captures):
            try:
                with urllib.request.urlopen(f"{self._pi_url}/capture", timeout=15) as resp:
                    frame = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  Image {i+1} fetch failed: {e}")
                continue
            if frame is None:
                continue
            corners, ids, _ = self._detector.detectMarkers(frame)
            if ids is None:
                continue
            hit = [corners[j] for j in range(len(ids)) if ids[j][0] == marker_id]
            if not hit:
                continue
            img_pts = hit[0][0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                rvecs.append(rvec.flatten())
                tvecs.append(tvec.flatten())

        if len(rvecs) < max(1, num_captures // 2):
            return None
        mean_rvec = cv2.Rodrigues(_mean_rotation(rvecs))[0]
        mean_tvec = np.mean(tvecs, axis=0).reshape(3, 1)
        return mean_rvec, mean_tvec

    def capture_pose(self, marker_id, num_captures=5):
        """Fetch num_captures high-res images from /capture, run fresh detection on each,
        and average the results. Uses the full-res camera matrix for accurate pose.

        Saves: marker pose in camera frame, board pose in camera frame,
        and marker pose in board/world frame.
        """
        board_rvecs, board_tvecs = [], []
        marker_rvecs, marker_tvecs = [], []

        print(f"Capturing pose for marker {marker_id} ({num_captures} images)...")

        for i in range(num_captures):
            try:
                with urllib.request.urlopen(f"{self._pi_url}/capture", timeout=15) as resp:
                    frame = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  Image {i+1} fetch failed: {e}")
                continue
            if frame is None:
                continue

            # Board (raw detection, full-res matrix, no EMA)
            rvec_board, tvec_board = self._detect_board(frame)
            if rvec_board is not None:
                board_rvecs.append(rvec_board.flatten())
                board_tvecs.append(tvec_board.flatten())

            # Target marker
            corners, ids, _ = self._detector.detectMarkers(frame)
            if ids is None:
                print(f"  Image {i+1}: no markers detected")
                continue

            hit = [(corners[j], ids[j][0]) for j in range(len(ids)) if ids[j][0] == marker_id]
            if not hit:
                print(f"  Image {i+1}: marker {marker_id} not found")
                continue

            img_pts = hit[0][0][0].astype(np.float32)
            m_len   = BOARD_MARKER_LENGTH if marker_id in BOARD_MARKER_IDS else MARKER_LENGTH
            obj_pts = np.array([
                [-m_len/2,  m_len/2, 0], [ m_len/2,  m_len/2, 0],
                [ m_len/2, -m_len/2, 0], [-m_len/2, -m_len/2, 0],
            ], dtype=np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                marker_rvecs.append(rvec.flatten())
                marker_tvecs.append(tvec.flatten())
                print(f"  Image {i+1}/{num_captures}: OK")
            else:
                print(f"  Image {i+1}/{num_captures}: solvePnP failed")

        if not marker_tvecs:
            return {'status': 'error', 'message': f'Marker {marker_id} not detected in any capture'}
        if len(marker_tvecs) < num_captures // 2:
            return {
                'status': 'error',
                'message': f'Marker {marker_id} detected in only {len(marker_tvecs)}/{num_captures} images'
            }

        mean_marker_rvec = cv2.Rodrigues(_mean_rotation(marker_rvecs))[0]
        mean_marker_tvec = np.mean(marker_tvecs, axis=0).reshape(3, 1)

        new_capture = {
            'marker_id':           marker_id,
            'timestamp':           datetime.now().isoformat(),
            'type':                'capture',
            'images':              len(marker_tvecs),
            'marker_camera_frame': {
                'rvec':    rvec_to_rpy(mean_marker_rvec),
                'tvec_mm': (mean_marker_tvec.flatten() * 1000).tolist(),
            },
            'board_camera_frame':  None,
            'marker_board_frame':  None,
            'marker_base_frame':   None,
        }

        # Robot base-frame pose via hand-eye (if calibrated)
        b = self._marker_camera_to_base(mean_marker_rvec, mean_marker_tvec)
        if b is not None:
            new_capture['marker_base_frame'] = {
                'x_mm': b[0], 'y_mm': b[1], 'z_mm': b[2], 'yaw_deg': b[3]
            }

        if len(board_tvecs) >= len(marker_tvecs) // 2:
            mean_board_rvec = cv2.Rodrigues(_mean_rotation(board_rvecs))[0]
            mean_board_tvec = np.mean(board_tvecs, axis=0).reshape(3, 1)
            new_capture['board_camera_frame'] = {
                'rvec':    rvec_to_rpy(mean_board_rvec),
                'tvec_mm': (mean_board_tvec.flatten() * 1000).tolist(),
            }
            rvec_bf, tvec_bf = transform_pose_to_board_frame(
                mean_marker_rvec, mean_marker_tvec,
                mean_board_rvec,  mean_board_tvec
            )
            new_capture['marker_board_frame'] = {
                'rvec':    rvec_to_rpy(rvec_bf),
                'tvec_mm': (tvec_bf.flatten() * 1000).tolist(),
            }

        captures = []
        if os.path.exists(self.pose_file):
            try:
                with open(self.pose_file, 'r') as f:
                    captures = json.load(f)
            except Exception:
                captures = []

        captures.append(new_capture)
        try:
            with open(self.pose_file, 'w') as f:
                json.dump(captures, f, indent=2)
            msg = f"Pose captured for marker {marker_id} ({len(marker_tvecs)} images)"
            print(msg)
            return {'status': 'success', 'message': msg, 'capture': new_capture}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to save pose: {e}'}

    def capture_pose_mean_debug(self, marker_id, num_captures=10,
                                 capture_board=True, capture_marker=True):
        """Fetch num_captures high-res images and compute mean, mean deviation, and max deviation
        across all detections — for characterising pose estimation noise and accuracy.

        capture_board:  collect and report board pose stats (camera frame)
        capture_marker: collect and report marker pose stats (camera frame)
        """
        if not capture_board and not capture_marker:
            return {'status': 'error', 'message': 'At least one of capture_board or capture_marker must be True'}

        print(f"Debug capture: {num_captures} images, marker {marker_id}, "
              f"board={'yes' if capture_board else 'no'}, marker={'yes' if capture_marker else 'no'}...")

        board_rvecs, board_tvecs = [], []
        marker_rvecs, marker_tvecs = [], []
        fetch_failures = 0
        board_misses   = 0
        marker_misses  = 0

        for i in range(num_captures):
            try:
                with urllib.request.urlopen(f"{self._pi_url}/capture", timeout=15) as resp:
                    frame = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  Image {i+1} fetch failed: {e}")
                fetch_failures += 1
                continue
            if frame is None:
                fetch_failures += 1
                continue

            print(f"  Image {i+1}/{num_captures} fetched")

            if capture_board:
                rvec_board, tvec_board = self._detect_board(frame)
                if rvec_board is None:
                    board_misses += 1
                else:
                    board_rvecs.append(rvec_board.flatten())
                    board_tvecs.append(tvec_board.flatten())

            if capture_marker:
                corners, ids, _ = self._detector.detectMarkers(frame)
                if ids is None:
                    marker_misses += 1
                    continue
                hit = [(corners[j], ids[j][0]) for j in range(len(ids)) if ids[j][0] == marker_id]
                if not hit:
                    marker_misses += 1
                    continue
                img_pts = hit[0][0][0].astype(np.float32)
                m_len   = BOARD_MARKER_LENGTH if marker_id in BOARD_MARKER_IDS else MARKER_LENGTH
                obj_pts = np.array([
                    [-m_len/2,  m_len/2, 0], [ m_len/2,  m_len/2, 0],
                    [ m_len/2, -m_len/2, 0], [-m_len/2, -m_len/2, 0],
                ], dtype=np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if ok:
                    marker_rvecs.append(rvec.flatten())
                    marker_tvecs.append(tvec.flatten())
                else:
                    marker_misses += 1

        valid = num_captures - fetch_failures

        if capture_board and len(board_tvecs) < valid // 2:
            return {'status': 'error',
                    'message': f'Too many missed board detections ({board_misses}/{valid})'}
        if capture_marker and len(marker_tvecs) < valid // 2:
            return {'status': 'error',
                    'message': f'Too many missed marker detections ({marker_misses}/{valid})'}

        def compute_stats(samples):
            arr      = np.array(samples)
            mean     = arr.mean(axis=0)
            devs     = np.abs(arr - mean)
            euc_devs = np.linalg.norm(arr - mean, axis=1)
            return mean, devs.mean(axis=0), devs.max(axis=0), float(euc_devs.mean()), float(euc_devs.max())

        board_stats = None
        if capture_board and board_tvecs:
            bt_mean, bt_mean_dev, bt_max_dev, bt_euc_mean, bt_euc_max = compute_stats(board_tvecs)
            br_mean, br_mean_dev, br_max_dev, _, _                     = compute_stats(board_rvecs)
            board_stats = {
                'samples':                    len(board_tvecs),
                'tvec_mean_mm':               (bt_mean     * 1000).tolist(),
                'tvec_mean_dev_mm':           (bt_mean_dev * 1000).tolist(),
                'tvec_max_dev_mm':            (bt_max_dev  * 1000).tolist(),
                'tvec_mean_euclidean_dev_mm': bt_euc_mean  * 1000,
                'tvec_max_euclidean_dev_mm':  bt_euc_max   * 1000,
                'rvec_mean_rpy_deg':          rvec_to_rpy(br_mean.reshape(3, 1)),
                'rvec_mean_dev_deg':          np.degrees(br_mean_dev).tolist(),
                'rvec_max_dev_deg':           np.degrees(br_max_dev).tolist(),
            }
            print(f"  [Board]  samples: {len(board_tvecs)}/{valid}  missed: {board_misses}")
            print(f"  [Board]  tvec mean (mm): {(bt_mean * 1000).tolist()}")
            print(f"  [Board]  tvec mean euclidean dev (mm): {bt_euc_mean * 1000:.3f}")
            print(f"  [Board]  tvec max  euclidean dev (mm): {bt_euc_max  * 1000:.3f}")

        marker_stats = None
        if capture_marker and marker_tvecs:
            mt_mean, mt_mean_dev, mt_max_dev, mt_euc_mean, mt_euc_max = compute_stats(marker_tvecs)
            mr_mean, mr_mean_dev, mr_max_dev, _, _                     = compute_stats(marker_rvecs)
            marker_stats = {
                'samples':                    len(marker_tvecs),
                'tvec_mean_mm':               (mt_mean     * 1000).tolist(),
                'tvec_mean_dev_mm':           (mt_mean_dev * 1000).tolist(),
                'tvec_max_dev_mm':            (mt_max_dev  * 1000).tolist(),
                'tvec_mean_euclidean_dev_mm': mt_euc_mean  * 1000,
                'tvec_max_euclidean_dev_mm':  mt_euc_max   * 1000,
                'rvec_mean_rpy_deg':          rvec_to_rpy(mr_mean.reshape(3, 1)),
                'rvec_mean_dev_deg':          np.degrees(mr_mean_dev).tolist(),
                'rvec_max_dev_deg':           np.degrees(mr_max_dev).tolist(),
            }
            print(f"  [Marker] samples: {len(marker_tvecs)}/{valid}  missed: {marker_misses}")
            print(f"  [Marker] tvec mean (mm): {(mt_mean * 1000).tolist()}")
            print(f"  [Marker] tvec mean euclidean dev (mm): {mt_euc_mean * 1000:.3f}")
            print(f"  [Marker] tvec max  euclidean dev (mm): {mt_euc_max  * 1000:.3f}")

        new_capture = {
            'marker_id':                marker_id,
            'timestamp':                datetime.now().isoformat(),
            'type':                     'mean_capture',
            'images_requested':         num_captures,
            'images_fetched':           valid,
            'board_pose_stats':         board_stats,
            'marker_camera_pose_stats': marker_stats,
        }

        captures = []
        if os.path.exists(self.pose_file):
            try:
                with open(self.pose_file, 'r') as f:
                    captures = json.load(f)
            except Exception:
                captures = []

        captures.append(new_capture)
        try:
            with open(self.pose_file, 'w') as f:
                json.dump(captures, f, indent=2)
            msg = f"Mean pose captured for marker {marker_id} ({valid} images)"
            print(msg)
            return {'status': 'success', 'message': msg, 'capture': new_capture}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to save pose: {e}'}
