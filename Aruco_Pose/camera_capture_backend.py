# Pi-side HTTP server for on-demand JPEG capture — Aruco Pose system.
# Exposes /preview (854x480, fast) and /capture (2304x1296, native 2x2-binned full FoV).
# Run on Raspberry Pi: python camera_capture_backend.py

from flask import Flask, send_file
from picamera2 import Picamera2
from libcamera import controls
import io
import threading
import time

PREVIEW_SIZE = (854, 480)
CAPTURE_SIZE = (2304, 1296)   # native 2x2-binned imx708 mode: full FoV, ~4x less memory than 12MP

app   = Flask(__name__)
_lock = threading.Lock()
_cam  = None
_capture_count = 0


def _log_mem(tag):
    """Print CMA-pool free + MemAvailable from /proc/meminfo.

    CMA is the camera/ISP DMA buffer pool — separate from normal RAM. Exhausting
    CMA (not RAM) is what stalls the driver, so CmaFree is the number to watch.
    """
    try:
        wanted = {'MemAvailable:', 'CmaFree:'}
        found = {}
        with open('/proc/meminfo') as f:
            for line in f:
                key = line.split()[0]
                if key in wanted:
                    found[key] = line.split()[1]
        print(f"[mem] {tag}: " + ", ".join(f"{k.rstrip(':')}={v}kB" for k, v in found.items()))
    except Exception as e:
        print(f"[mem] {tag}: read failed ({e})")


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
    _log_mem("camera ready")
    print("Camera ready.")


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
    """High-res JPEG for pose capture (2304x1296, binned full FoV)."""
    global _capture_count
    _capture_count += 1
    _log_mem(f"before capture #{_capture_count}")
    buf = io.BytesIO()
    with _lock:
        _cam.capture_file(buf, format='jpeg')
    buf.seek(0)
    _log_mem(f"after capture #{_capture_count}")
    return send_file(buf, mimetype='image/jpeg')


if __name__ == '__main__':
    _init_camera()
    app.run(host='0.0.0.0', port=8000, threaded=True, use_reloader=False)
