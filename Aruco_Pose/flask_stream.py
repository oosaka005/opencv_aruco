# Flask server for ArUco pose estimation — streams video and exposes pose capture API.
# Runs the detection loop in a background thread via ArucoProcessor.

from aruco_processing import ArucoProcessor
import flask
import os
import cv2

app = flask.Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

processor = ArucoProcessor()

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def frame_to_bytes():
        while True:
            frame = processor.get_frame()
            if frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--FRAME\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')

    return flask.Response(frame_to_bytes(), mimetype='multipart/x-mixed-replace; boundary=FRAME')

@app.route('/capture_pose', methods=['POST'])
def capture_pose():
    data      = flask.request.json
    marker_id = data.get('marker_id')
    debug     = data.get('debug', False)

    if marker_id is None:
        return flask.jsonify({'status': 'error', 'message': 'marker_id is required'}), 400

    try:
        if debug:
            result = processor.capture_pose_mean_debug(int(marker_id))
        else:
            result = processor.capture_pose(int(marker_id))
        return flask.jsonify(result)
    except ValueError:
        return flask.jsonify({'status': 'error', 'message': 'marker_id must be an integer'}), 400
    except Exception as e:
        return flask.jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    processor.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
