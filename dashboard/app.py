"""Glaucoma analysis dashboard (ANDG-Net). Run:  python app.py   then open http://127.0.0.1:5000"""
import os, json, datetime
import numpy as np, cv2
from flask import Flask, render_template, request, jsonify
from inference import Engine

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, 'config.json')))
for k, env in (('tau', 'ANDG_TAU'), ('stage1_threshold', 'ANDG_THRESHOLD'), ('port', 'PORT')):
    if os.environ.get(env): cfg[k] = type(cfg[k])(os.environ[env])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
engine = Engine(cfg, BASE)
ALLOWED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/status')
def status():
    return jsonify(engine.status())


@app.post('/api/analyze')
def analyze():
    f = request.files.get('image')
    if f is None or not f.filename:
        return jsonify(error='No image uploaded.'), 400
    if os.path.splitext(f.filename.lower())[1] not in ALLOWED:
        return jsonify(error='Unsupported file type. Use PNG, JPG, BMP, TIFF or WEBP.'), 400
    arr = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error='Could not read the file as an image.'), 400
    res = engine.analyze(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res['filename'] = os.path.basename(f.filename)
    res['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    return jsonify(res)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='File too large (limit 25 MB).'), 413


if __name__ == '__main__':
    print('Mode:', engine.status()['mode'], '|', engine.error or 'checkpoints loaded')
    app.run(host=cfg['host'], port=cfg['port'], debug=False)
