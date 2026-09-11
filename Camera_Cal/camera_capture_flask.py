# PC-side Flask server for the photo-capture calibration workflow.
# Triggers high-res captures on the Pi backend, saves JPEGs to cal_images/, and serves the UI.
# Requires camera_capture_backend.py running on the Pi at PI_URL.

from flask import Flask, render_template, jsonify, request, send_file
import os
import shutil
import tempfile
import urllib.request
from datetime import datetime

PI_URL   = "http://192.168.10.141:8000"
_cal_dir = os.path.join(os.path.dirname(__file__), 'cal_images')

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


@app.route('/')
def index():
    return render_template('capture_index.html')


@app.route('/cal_image', methods=['POST'])
def cal_image():
    try:
        with urllib.request.urlopen(f"{PI_URL}/capture", timeout=10) as resp:
            image_data = resp.read()
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Pi capture failed: {e}'}), 500

    os.makedirs(_cal_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename  = f"calibration_{timestamp}.jpg"

    with open(os.path.join(_cal_dir, filename), 'wb') as f:
        f.write(image_data)

    return jsonify({'status': 'success', 'filename': filename})


@app.route('/image_count')
def image_count():
    if not os.path.exists(_cal_dir):
        return jsonify({'count': 0})
    count = sum(1 for f in os.listdir(_cal_dir) if f.lower().endswith('.jpg'))
    return jsonify({'count': count})


@app.route('/download_all')
def download_all():
    if not os.path.exists(_cal_dir) or not any(f.lower().endswith('.jpg') for f in os.listdir(_cal_dir)):
        return jsonify({'status': 'error', 'message': 'No images to download'}), 404

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_images_{timestamp}"
    zip_path  = os.path.join(tempfile.gettempdir(), base_name + '.zip')
    shutil.make_archive(os.path.join(tempfile.gettempdir(), base_name), 'zip', _cal_dir)
    return send_file(zip_path, as_attachment=True, download_name=base_name + '.zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
