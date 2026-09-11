# Pi-side HTTP server for on-demand JPEG capture.
# Exposes /preview (lores, fast) and /capture (high-res, for calibration).
# Run on Raspberry Pi: python camera_capture_backend.py

from flask import Flask, send_file
from picamera2 import Picamera2
from libcamera import controls
import io
import threading
import time

PREVIEW_SIZE = (854, 480)
CAPTURE_SIZE = (2304, 1296)   # native 2x2-binned imx708 mode (full FoV); MUST match Aruco_Pose capture res

app   = Flask(__name__)
_lock = threading.Lock()
_cam  = None


def _init_camera():
    global _cam
    _cam = Picamera2()
    cfg = _cam.create_still_configuration(
        main  = {"size": CAPTURE_SIZE},
        lores = {"size": PREVIEW_SIZE, "format": "YUV420"},
    )
    _cam.configure(cfg)
    _cam.set_controls({
        "AeEnable":     False,
        "ExposureTime": 10000,
        "AnalogueGain": 1.123,
        "AfMode":       controls.AfModeEnum.Manual,
        "LensPosition": 0.4,
    })
    _cam.start()
    time.sleep(1)  # warm-up

    # # Auto-find sharpest lens position across a coarse sweep
    # best_lp, best_fom = 0.4, -1
    # for lp in [0.3, 0.35, 0.4, 0.45, 0.5]:
    #     _cam.set_controls({"LensPosition": lp})
    #     time.sleep(0.4)
    #     fom = _cam.capture_metadata().get("FocusFoM", 0)
    #     if fom > best_fom:
    #         best_fom, best_lp = fom, lp

    # _cam.set_controls({"LensPosition": best_lp})
    # print(f"Best lens position: {best_lp}  FoM: {best_fom}")

    #I think one used in calibration was 0.3...


@app.route('/preview')
def preview():
    """Lores JPEG for live preview polling (~854x480, fast)."""
    buf = io.BytesIO()
    with _lock:
        _cam.capture_file(buf, format='jpeg', name='lores')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


@app.route('/capture')
def capture():
    """High-res JPEG for calibration (2304x1296, binned full FoV)."""
    buf = io.BytesIO()
    with _lock:
        _cam.capture_file(buf, format='jpeg')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


if __name__ == '__main__':
    _init_camera()
    app.run(host='0.0.0.0', port=8000, threaded=True, use_reloader=False)
