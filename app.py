from flask import Flask, render_template, send_file
from threading import Lock
from flask_socketio import SocketIO
from model import KuroModel
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './models/upload'

socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*', async_mode=None)
thread_train = None
thread_attribute = None
thread_lock = Lock()
kuro_ns = '/kuro'

adjs = np.load('./large_data/adjs.npy')

MODELS = {}

@app.route('/')
def index():
    return "Hello."

@app.route('/testtrain')
def test_train():
    return render_template('train.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file(request):
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'file uploaded successfully'
    return "file uploaded Fail"

@app.route("/download")
def download_file(request):
    model_id = request.args.get('id')
    dir = f'./models/{model_id}'
    if os.path.exists(dir):
        return send_file(dir, as_attachment=True)
    else:
        return "The downloaded file does not exist"

@socketio.on('connect', namespace=kuro_ns)
def connected_msg():
    socketio.emit('response_connect', {'msg': 'Connected.'}, namespace=kuro_ns)
    print('client connected.')

@socketio.on('response_disconnect', namespace=kuro_ns)
def disconnect_msg():
    socketio.emit('disconnect', {'msg': 'Disconnected.'}, namespace=kuro_ns)
    print('client disconnected.')

@socketio.on('adjs', namespace=kuro_ns)
def get_adj(index):
    socketio.emit('adj', {'in': adjs[:, :, index].tolist(), 'out': adjs[:, index].tolist()}, namespace=kuro_ns)

@socketio.on('train', namespace=kuro_ns)
def kuro_train(train_set, params, uuid):
    global thread_train
    with thread_lock:
        if thread_train is None or not thread_train.is_alive():
            thread = socketio.start_background_task(target=train_model, train_set=train_set, params=params, uuid=uuid)

@socketio.on('attribute', namespace=kuro_ns)
def kuro_attribute(rid, models):
    global thread_attribute
    with thread_lock:
        if thread_attribute is None or not thread_attribute.is_alive():
            thread = socketio.start_background_task(target=attribute, rid=rid, models=models)

def attribute(rid, models):
    result = []
    i = 0
    for model in models:
        socketio.emit('attribute_info', {'type': 'progress', 'content': {'count': i + 1}}, namespace=kuro_ns)
        result.append(MODELS[model].get_ig(rid))
        i += 1
    socketio.emit('attribute_result', {'type': 'result', 'content': result}, namespace=kuro_ns)

def train_model(train_set, params, uuid):
    def onEpochEnd(epoch, logs):
        logs['lr'] = logs['lr'].item()
        logs['weight_decay'] = logs['weight_decay'].item()
        socketio.emit('train_info', {'type': 'epoch', 'content': {'count': epoch, 'logs': logs}}, namespace=kuro_ns)

    model = KuroModel(uuid, params['embeddingSize'], [params['gcnSize1'],params['gcnSize2']], params['dropout'], params['lr'], params['wd'])
    socketio.emit('train_info', {'type': 'info', 'content': 'Loading Data.'}, namespace=kuro_ns)
    model.load_data()
    model.create_model()
    model.train(train_set, onEpochEnd)
    socketio.emit('train_result', {'type': 'info', 'content': 'Train Finished.'}, namespace=kuro_ns)
    model.prediction(model.default_train_set)
    MODELS[model.id] = model
    socketio.emit('train_result', {'type': 'result', 'content': model.result}, namespace=kuro_ns)

if __name__ == '__main__':
    socketio.run(app)