> [!NOTE]
> The current code in this repository is an unchanged copy of
> [`Renato-D/aruco_pickplace`](https://github.com/Renato-D/aruco_pickplace) at
> commit [`8dc9792`](https://github.com/Renato-D/aruco_pickplace/commit/8dc97928f8fd861ac670f310795e01937ba88ed7),
> shared by Renato. This repository is intended for sharing and reference. We plan to
> develop original code based on this codebase in the future.

# ArUco Vision-Guided Pick-and-Place (Lite6 xArm)

Locate objects with an overhead camera and have a robot arm **pick one up and place
it somewhere else** — using ArUco markers to tell the arm *where* things are.

An overhead Raspberry Pi camera detects ArUco markers. Two calibrations turn a
marker's position in the image into a position in the **robot's base coordinates**:
the *camera calibration* (lens intrinsics) and the *hand-eye calibration*
(camera→robot transform). A UFACTORY **Lite6** arm then picks an object from one
marker's location and places it at another.

**Everything in this branch builds toward one thing: the pick-and-place demo,
[`Aruco_Pose/pick_place_demo.py`](Aruco_Pose/pick_place_demo.py).** Every other
script either produces something the demo needs (a calibration file) or verifies
that it's good enough to trust.

![The workspace: a Lite6 arm with the gripper open above the bench, with ArUco-marked
objects (pick-up and drop-off) laid out in front of it.](system_image.png)

*The physical setup — the Lite6 and its gripper, with ArUco-tagged objects in the
overhead camera's field of view.*

---

## What the demo does

1. Detects the **pickup** marker (ID 37, on the object) and the **drop-off** marker
   (ID 35, on the holder), and converts both to robot base coordinates.
2. Runs a fixed motion cycle:
   `initial → midpoint(s) → above-pick → pick → [close gripper] → above-pick →
   above-drop → drop → [open gripper] → above-drop → midpoint(s) → initial`.
3. Marker X/Y/yaw come from the camera; all Z heights, midpoints, and the home pose
   are hardcoded to your workspace.

The concrete target application is placing a cup into a mixer holder, where both
position **and** orientation matter.

---

## Hardware

- **Raspberry Pi Zero 2 W** + **Camera Module 3** (imx708), mounted overhead.
- **UFACTORY Lite6** 6-axis arm with the Lite6 gripper.
- A printed **ChArUco board** (DICT_4X4_50) for camera calibration.
- Printed **ArUco markers** (DICT_4X4_50, 40 mm): one on the object/pick-up spot, one on the holder,
  and one on the arm's TCP for the hand-eye calibration.

The Pi and the PC talk over your LAN (accessed through SSH); the arm is reached over its own IP.

---

## How it works

The camera never runs vision itself — it's a thin image server. All detection,
pose math, and robot control run on the PC.

```
 Raspberry Pi                             PC (control machine)
 ────────────                             ────────────────────
 camera_capture_backend.py   ── HTTP ─▶   aruco_processing.py  (detect + pose)
   /preview  (fast lores JPEG)               │
   /capture  (2304x1296 still)               ├─ hand_eye_calibration.py  → cam2base
                                             ├─ marker_position.py       (query)
                                             └─ pick_place_demo.py        (the demo)
                                                        │
                                                        ▼
                                                   Lite6 xArm  (xarm-python-sdk)
```

**Coordinate pipeline:** `marker in image → (camera intrinsics) → marker in camera
frame → (cam2base hand-eye transform) → marker in robot base frame → (+ tuned offset)
→ gripper target`.

---

## Repository layout

```
Camera_Cal/                         # STEP 1 — camera lens (intrinsic) calibration
├── charucoGen.py                   # generate the ChArUco board to print
├── camera_capture_backend.py       # (Pi) image server for capturing calibration photos (run on pi)
├── camera_capture_flask.py         # (PC) web UI: trigger captures, save to cal_images/
├── camera_tune.py                  # (Pi) measure focus/exposure to lock into the backends (run on pi and input into backend)
├── camera_calibration.py           # (PC) compute camera_matrix + dist_coeffs from cal_images/
├── templates/capture_index.html    # calibration capture web page
├── cal_images/                     # captured ChArUco photos (input to calibration)
└── calibration_results/            # camera_matrix.npy, dist_coeffs.npy (the intrinsics)

Aruco_Pose/                         # STEPS 2–4 — pose estimation, hand-eye, and the demo
├── camera_capture_backend.py       # (Pi) image server: /preview + /capture (2304x1296)
├── aruco-camera.service            # (Pi) systemd unit — auto-restarts the image server to be resilient against crashes
├── aruco_processing.py             # (PC) ArucoProcessor: detection, pose, filtering, capture
├── flask_stream.py                 # (PC) web UI: live annotated stream + /capture_pose
├── templates/index.html            # pose web page
├── hand_eye_calibration.py         # (PC) eye-to-hand calibration → calibration_results/cam2base.npy
├── marker_position.py              # (PC) terminal tool: type a marker ID → its base-frame pose
├── pick_place_demo.py              # (PC) ★ THE DEMO ★  pick from marker 37, place at marker 35 (or choose your own)
├── Debugging/
│   ├── hand_eye_verify.py          # (PC) prove the hand-eye calibration (motion ground-truth test)
│   └── precision_rotation_test.py  # (PC) camera precision + yaw-accuracy checks
├── calibration_results/            # camera_matrix.npy, dist_coeffs.npy, cam2base.npy/.json
└── pose_calibration.json           # recorded pose captures (accuracy-test data)

requirements.txt                    # PC-side Python deps (see Setup)
```

---

## Setup

### PC (control machine)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```
`requirements.txt` covers everything the PC-side scripts need: `numpy`, `scipy`,
`opencv-contrib-python` (provides `cv2.aruco`), `Flask`, and `xarm-python-sdk`.

### Raspberry Pi (camera server)

`picamera2` + `libcamera` are **not** pip packages — they come with Raspberry Pi OS
(`sudo apt install python3-picamera2`). Only the camera backends run on the Pi.

Run the pose image server as a service so it survives crashes (the Pi Zero is memory-
constrained — see Known limitations). Instructions are in the header of
[`aruco-camera.service`](Aruco_Pose/aruco-camera.service): copy it to
`/etc/systemd/system/`, edit the `User=`/paths, `enable --now`, and (recommended)
enable the hardware watchdog and disable SD-card swap.

---

## The end-to-end pipeline (do this in order)

Everything below leads to the demo.

**1. Calibrate the camera (Camera_Cal).**
   - `charucoGen.py` → print the ChArUco board (rigid/flat backing matters).
   - Mount the camera in its final overhead position and lighting.
   - Run `camera_tune.py` on the Pi to get focus/exposure, and paste those values into
     **both** `camera_capture_backend.py` files. **Focus must be identical for
     calibration and operation** — it changes the intrinsics.
   - Capture ~20+ ChArUco views via `camera_capture_flask.py` (PC) + the Pi backend.
   - Run `camera_calibration.py` → produces `camera_matrix.npy` + `dist_coeffs.npy`.
   - **Copy those two files into `Aruco_Pose/calibration_results/`** — that's what the
     pose code loads.

**2. Start the pose camera server on the Pi** (`aruco-camera.service`, serving 2304×1296).

**3. Hand-eye calibration (`hand_eye_calibration.py`).**
   Mount a marker rigidly on the TCP, jog the arm to ~12–20 varied poses (vary
   orientation about **multiple axes**, not just yaw), and it solves and saves
   `calibration_results/cam2base.npy`. Aim for a residual under ~1 mm.

**4. Verify (`Debugging/`).**
   - `hand_eye_verify.py` — move the arm a known distance and confirm the camera-derived
     base position moves the same way (catches convention errors).
   - `precision_rotation_test.py` — camera repeatability + yaw accuracy.

**5. Tune the demo offsets (`marker_position.py`).**
   Query markers 37 and 35, compare to where the gripper actually needs to be, and set
   the offsets in `pick_place_demo.py`.

**6. Run the demo (`pick_place_demo.py`).**
   It detects both markers, prints the full planned path, and waits for a `y` before it
   moves. Start slow, keep the e-stop in reach.

---

## Configuration you must set

| Where | What |
|---|---|
| `hand_eye_calibration.py` | `ROBOT_IP`, `PI_URL` (shared by the other scripts via import) |
| `hand_eye_calibration.py` / `aruco_processing.py` | `MARKER_LENGTH_M` / `MARKER_LENGTH` = your printed marker size (40 mm) |
| both `camera_capture_backend.py` | `LensPosition`, `ExposureTime`, `AnalogueGain` (from `camera_tune.py`) |
| `pick_place_demo.py` | marker IDs (37 pickup / 35 drop-off), `Z_*` heights, `INITIAL_POSE`, `MIDPOINTS`, offsets |

---

## Known limitations & things still to fix

Read this before trusting the demo — these are the open items.

- **⚠️ Offsets aren't exact when a marker rotates.** In `pick_place_demo.py` the
  pickup/drop-off offsets are only correct at the marker orientation they were tuned
  at (pickup ~0°, drop-off ~91°). A single tuned offset mixes *object-frame* grasp
  geometry (should rotate with the marker) with a *fixed base-frame* bias (should not),
  so the grasp drifts as a marker turns. **Fix:** split each offset into a rotating
  object-frame component + a fixed base-frame component and re-tune at two orientations
  ~90° apart. (A reference-yaw rotation was attempted and reverted.)

- **Accuracy is ~1–2 mm in X/Y and ≤2° in yaw** (full system, including the arm). The
  camera itself is not the bottleneck (repeatability ~0.1 mm); the robot's own accuracy
  and the current camera calibration are. **Depth (Z) is the weakest axis** — fine for
  overhead work where you set height with a fixed approach.

- **Camera calibration is mediocre (~0.4 px RMS at 2304).** Re-doing it with careful
  focus, a flat/rigid board, and more/wider views (~0.2–0.25 px) would tighten both X/Y
  and yaw. This is the single highest-leverage improvement.

- **The 2° yaw figure is pessimistic.** It's inflated by the marker being mounted off
  the wrist axis and swinging across the lens field. Mounting the marker centered on the
  TCP yaw axis (and a better calibration) would improve it — relevant because orientation
  matters for the cup-in-holder task.

- **Pi Zero 2 W memory is tight.** Full 12 MP capture exhausted the camera's CMA buffer
  pool and froze the Pi; the fix was to capture at the **native 2×2-binned 2304×1296**
  mode (full field of view, ~4× less memory). Resilience is *recovery, not prevention*:
  the systemd service auto-restarts on crash and the hardware watchdog reboots on a full
  freeze. Keep SD-card swap disabled.

- **Intrinsics must match the capture resolution (2304×1296).** If you change capture
  resolution, recalibrate (or the poses are silently mis-scaled). Focus must match
  between calibration and operation.

- **Hand-eye needs rotation diversity about multiple axes.** Yaw-only pose sets are
  degenerate and produce a garbage translation — always verify with the `Debugging/`
  scripts and the physical camera-position sanity check.

- **Joint-6 winding** is mitigated (shortest-path yaw + returning to the captured home
  joint angles each cycle), but the mid-cycle yaw handling works in TCP-yaw space, which
  is a close proxy for J6, not exact. If you see winding, the robust fix is joint-space
  commands for the wrist.

- **Angled-cup geometry is deferred.** The demo assumes a tool-down grasp and uses only
  marker X/Y/yaw. Handling a cup that sits at a fixed roll/pitch (e.g. two markers on a
  rotation platform, solving the cup's tilt) is future work.

---

## Notes

- Only one process can own the Pi camera at a time — stop the `aruco-camera` service
  before running `camera_tune.py` (both open the camera).
- `camera_stream_simple.py` (earlier streaming approach) is retired; the system now uses
  on-demand HTTP JPEG capture rather than a continuous video stream.

## Author

Renato
