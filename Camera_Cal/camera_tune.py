# Standalone Pi-side camera tuner.
#
# Aims the camera at your scene, lets autofocus + auto-exposure meter the real
# lighting/distance, then prints a set_controls{} block with the measured fixed
# values to paste into BOTH camera_capture_backend.py files (Camera_Cal + Aruco_Pose).
# Also saves one sample JPEG so you can eyeball sharpness/exposure.
#
# Run on the Pi, with the camera mounted where it will operate and pointed at the
# calibration scene at operating distance:
#     python camera_tune.py
#
# NOTE: only one process can own the camera. If the systemd service is running,
# stop it first:  sudo systemctl stop aruco-camera

from picamera2 import Picamera2
from libcamera import controls
import time

CAPTURE_SIZE = (2304, 1296)   # must match the backend capture resolution
SETTLE_S     = 2.0            # seconds to let auto-exposure/AWB converge
SAMPLE_PATH  = "tune_sample.jpg"


def main():
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": CAPTURE_SIZE}))

    # Everything on AUTO first, so the camera meters the real scene.
    cam.set_controls({
        "AfMode":    controls.AfModeEnum.Auto,
        "AeEnable":  True,
        "AwbEnable": True,
    })
    cam.start()
    time.sleep(1.0)  # warm-up

    print(f"Camera running at {CAPTURE_SIZE}. Aim it at the scene at operating distance.")
    input("Press Enter to measure focus + exposure for the current scene...")

    # ── Focus: one autofocus scan, then read the lens position it settled on ──
    lens_position = None
    try:
        print("Running autofocus scan...")
        if cam.autofocus_cycle():
            lens_position = cam.capture_metadata().get("LensPosition")
    except Exception as e:
        print(f"  autofocus_cycle unavailable ({e}); falling back to a sharpness sweep")

    if lens_position is None:
        best_lp, best_fom = None, -1.0
        for lp in [x / 10.0 for x in range(0, 105, 5)]:   # 0.0 .. 10.0 dioptres
            cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": lp})
            time.sleep(0.4)
            fom = cam.capture_metadata().get("FocusFoM", 0)
            if fom > best_fom:
                best_fom, best_lp = fom, lp
        lens_position = best_lp
        print(f"  sweep picked LensPosition={best_lp} (FocusFoM={best_fom})")

    # ── Exposure / gain / white balance: let auto settle, then read back ──
    print(f"Letting auto-exposure/white-balance settle ({SETTLE_S}s)...")
    time.sleep(SETTLE_S)
    md = cam.capture_metadata()
    exposure = md.get("ExposureTime")
    gain     = md.get("AnalogueGain")
    colour   = md.get("ColourGains")

    lens_val = round(float(lens_position), 3) if lens_position is not None else 0.0
    exp_val  = int(exposure) if exposure else 8000
    gain_val = round(float(gain), 3) if gain else 1.0

    # ── Lock the measured values and grab one sample image to inspect ──
    cam.set_controls({
        "AfMode":       controls.AfModeEnum.Manual,
        "LensPosition": lens_val,
        "AeEnable":     False,
        "ExposureTime": exp_val,
        "AnalogueGain": gain_val,
    })
    time.sleep(0.5)
    cam.capture_file(SAMPLE_PATH)
    cam.stop()
    cam.close()

    # ── Report ──
    print("\n" + "=" * 62)
    print("Paste this into set_controls({...}) in BOTH backends:")
    print("=" * 62)
    print('    _cam.set_controls({')
    print('        "AeEnable":     False,')
    print(f'        "ExposureTime": {exp_val},')
    print(f'        "AnalogueGain": {gain_val},')
    print('        "AfMode":       controls.AfModeEnum.Manual,')
    print(f'        "LensPosition": {lens_val},')
    print('    })')
    print("=" * 62)
    if colour:
        print(f"(AWB ColourGains were {tuple(round(float(c), 3) for c in colour)} — "
              "irrelevant for grayscale ArUco, shown for reference)")
    print(f"Sample image saved to ./{SAMPLE_PATH} — scp it over and check sharpness/exposure.")
    if gain_val > 2.0:
        print(f"[WARN] AnalogueGain is {gain_val} (>2) — add light and re-run. "
              "Gain amplifies noise, which blurs sub-pixel corner detection.")


if __name__ == "__main__":
    main()
