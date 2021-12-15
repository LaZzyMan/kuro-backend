from flask import Flask, render_template
from threading import Lock
from flask_socketio import SocketIO
from model import KuroModel

app = Flask(__name__)

socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*', async_mode=None)
thread = None
thread_lock = Lock()
kuro_ns = '/kuro'


@app.route('/')
def index():
    return "Hello."

@app.route('/testtrain')
def test_train():
    return render_template('train.html')


@socketio.on('connect', namespace=kuro_ns)
def connected_msg():
    socketio.emit('response_connect', {'msg': 'Connected.'}, namespace=kuro_ns)
    print('client connected.')

@socketio.on('response_disconnect', namespace=kuro_ns)
def disconnect_msg():
    socketio.emit('disconnect', {'msg': 'Disconnected.'}, namespace=kuro_ns)
    print('client disconnected.')

@socketio.on('train', namespace=kuro_ns)
def kuro_train(train_set):
    global thread
    with thread_lock:
        if thread is None or not thread.is_alive():
            thread = socketio.start_background_task(target=train_model, train_set=train_set)

def train_model(train_set):
    def onEpochEnd(epoch, logs):
        logs['lr'] = logs['lr'].item()
        logs['weight_decay'] = logs['weight_decay'].item()
        socketio.emit('train_info', {'type': 'epoch', 'content': {'count': epoch, 'logs': logs}}, namespace=kuro_ns)

    model = KuroModel(50, [64,64], 0.5, 0.012, 0.009)
    socketio.emit('train_info', {'type': 'info', 'content': 'Loading Data.'}, namespace=kuro_ns)
    model.load_data()
    model.create_model()
    model.train(model.default_train_set, onEpochEnd)
    socketio.emit('train_result', {'type': 'info', 'content': 'Train Finished.'}, namespace=kuro_ns)
    results = model.prediction(model.default_train_set)
    socketio.emit('train_result', {'type': 'result', 'content': results}, namespace=kuro_ns)

if __name__ == '__main__':
    socketio.run(app)