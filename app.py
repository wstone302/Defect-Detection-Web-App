from flask import Flask, render_template, request, send_file, jsonify
import torch
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 載入模型
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model.conf = 0.25  # 信心閾值

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({'error': '未上傳任何檔案'}), 400

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)
    results.render()

    names = model.names
    detections = results.pred[0].cpu().numpy()
    class_counts = {}
    for det in detections:
        cls_id = int(det[5])
        name = names[cls_id]
        class_counts[name] = class_counts.get(name, 0) + 1

    img = Image.fromarray(results.ims[0])
    result_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
    img.save(result_path)

    return jsonify({
        'image_path': result_path,
        'class_counts': class_counts
    })

@app.route('/video_detect', methods=['POST'])
def video_detect():
    file = request.files['file']
    if not file or not file.filename.endswith(('.mp4', '.avi', '.mov', '.MOV')):
        return jsonify({'error': '請上傳影片檔（.mp4/.avi/.mov）'}), 400

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_classes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()

        for det in results.pred[0].cpu().numpy():
            cls_id = int(det[5])
            name = model.names[cls_id]
            all_classes[name] = all_classes.get(name, 0) + 1

        writer.write(results.ims[0][:, :, ::-1])  # BGR

    cap.release()
    writer.release()

    return jsonify({
        'video_path': output_path,
        'class_counts': all_classes
    })

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """接收單一幀畫面，進行 YOLO 推論，回傳渲染後的圖片"""
    file = request.files['file']
    if not file:
        return 'No frame received', 400

    # 轉為 OpenCV 圖片格式
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)
    results.render()

    # 回傳渲染後圖片（BGR to JPEG）
    _, jpeg = cv2.imencode('.jpg', results.ims[0])
    return jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/get_image')
def get_image():
    path = request.args.get("path")
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)  # 指定 host 和 port

from flask import Flask, render_template, request, send_file, jsonify
import torch
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 載入模型
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model.conf = 0.25  # 信心閾值

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if not file:
        return jsonify({'error': '未上傳任何檔案'}), 400

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)
    results.render()

    names = model.names
    detections = results.pred[0].cpu().numpy()
    class_counts = {}
    for det in detections:
        cls_id = int(det[5])
        name = names[cls_id]
        class_counts[name] = class_counts.get(name, 0) + 1

    img = Image.fromarray(results.ims[0])
    result_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
    img.save(result_path)

    return jsonify({
        'image_path': result_path,
        'class_counts': class_counts
    })

@app.route('/video_detect', methods=['POST'])
def video_detect():
    file = request.files['file']
    if not file or not file.filename.endswith(('.mp4', '.avi', '.mov', '.MOV')):
        return jsonify({'error': '請上傳影片檔（.mp4/.avi/.mov）'}), 400

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(UPLOAD_FOLDER, 'result_' + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_classes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()

        for det in results.pred[0].cpu().numpy():
            cls_id = int(det[5])
            name = model.names[cls_id]
            all_classes[name] = all_classes.get(name, 0) + 1

        writer.write(results.ims[0][:, :, ::-1])  # BGR

    cap.release()
    writer.release()

    return jsonify({
        'video_path': output_path,
        'class_counts': all_classes
    })

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """接收單一幀畫面，進行 YOLO 推論，回傳渲染後的圖片"""
    file = request.files['file']
    if not file:
        return 'No frame received', 400

    # 轉為 OpenCV 圖片格式
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame)
    results.render()

    # 回傳渲染後圖片（BGR to JPEG）
    _, jpeg = cv2.imencode('.jpg', results.ims[0])
    return jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/get_image')
def get_image():
    path = request.args.get("path")
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)  # 指定 host 和 port
