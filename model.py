import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, LambdaCallback
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow_addons.optimizers import AdamW
from layers import GraphConvolution, AggregationLayer, SqueezedSparseConversion, Loss
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import os
import json
import zipfile
import shutil

def zip(folder, file):
  z = zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED) 
  for dirpath, dirnames, filenames in os.walk(folder):
    fpath = dirpath.replace(folder,'') 
    fpath = fpath and fpath + os.sep or ''
    for filename in filenames:
        z.write(os.path.join(dirpath, filename),fpath+filename)
  z.close()

class KuroModel:
    def __init__(self, id, embedding_size, layer_sizes, dropout, lr, wd, e, weights) -> None:
        self.id = id
        self.classes = ['C', 'G', 'M', 'P', 'R', 'U']
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.e = e
        self.weights = weights
        self.model = None
        self.result = None
        self.train_set = None
        self.int_grad = None

    def info(self):
      return {
        'params': {
          'embeddingSize': self.embedding_size,
          'gcnSize1': self.layer_sizes[0],
          'gcnSize2': self.layer_sizes[1],
          'dropout': self.dropout,
          'lr': self.lr,
          'wd': self.wd,
        },
        'id': self.id,
        'trainSet': list([np.int(i) for i in self.train_set]),
        'result': self.result
      }

    def save(self):
        if(os.path.exists(f'./models/{self.id}.zip')):
          return self.id
        dir = f'./models/{self.id}'
        os.mkdir(dir)
        self.model.save(f'{dir}/model')
        with open(f'{dir}/params.json', 'w+') as f:
            json.dump({
                'embeddingSize': self.embedding_size,
                'layerSizes': self.layer_sizes,
                'dropout': self.dropout,
                'lr': self.lr,
                'wd': self.wd,
                'trainSet': list([np.int(i) for i in self.train_set]),
                'result': self.result,
                'id': self.id
                }, f)
            f.close()
        zip(dir, f'./models/{self.id}.zip')
        shutil.rmtree(dir,ignore_errors=True)
        return self.id

    def load(self):
        dir = f'./models/{self.id}'
        if os.path.exists(dir):
            with open(f'{dir}/params.json', 'w+') as f:
                params = json.load(f)
                f.close()
            self.embedding_size = params['embeddingSize']
            self.layer_sizes = params['layerSizes']
            self.dropout = params['dropout']
            self.lr = params['lr']
            self.wd = params['wd']
            self.result = params['result']
            self.train_set = params['trainSet']
            self.model = tf.keras.models.load_model(f'{dir}/model')

    def encode(self, codes):
        out = []
        for code in codes:
            r = np.asanyarray([0, 0, 0, 0, 0, 0])
            r[self.classes.index(code)] = 1
            out.append(r)
        return np.asanyarray(out)
    
    def decode(self, codes):
        return np.asanyarray([self.classes[np.argmax(code)] for code in codes])

    def create_model(self):
        features=['LC', 'POI', 'Building', 'Mobility', 'Rhythm']
        n_features = [19, 17, 4, 1514, 48]
        # 网络输入[[features], indices, adj_indices, adj_values]
        feature_inputs = [layers.Input(batch_shape=(1, 1514, n_features[i]), name='Input_%s' % f) for i, f in enumerate(features)]
        out_indices = layers.Input(batch_shape=(1, None), dtype='int32', name='out_indices')
        adj_indices = layers.Input(batch_shape=(1, None, 2), dtype='int64', name='Adj_indices')
        adj_values = layers.Input(batch_shape=(1, None), name='Adj_values')
        target = layers.Input(batch_shape=(1, None, 6), name='target')
        weight = layers.Input(batch_shape=(1, 1602, 6), name='weight')
        x_input = [feature_inputs, out_indices, adj_indices, adj_values, target, weight]
        adj_inputs = SqueezedSparseConversion(shape=(1514, 1514), dtype=adj_values.dtype)([adj_indices, adj_values])
        x_outs = [layers.Dense(units=self.embedding_size, activation='relu', name=f'Hidden_%s' % features[i])(f) for i, f in enumerate(feature_inputs)]
        # x_outs = feature_inputs
        # 多通道图卷积
        x_outs = [layers.Dropout(self.dropout)(f) for f in x_outs]
        x_outs = [GraphConvolution(self.layer_sizes[0], activation='relu', name=f'GC_%s' % features[i])([f, adj_inputs]) for i, f in enumerate(x_outs)]
        x_out = AggregationLayer(activation='relu', method='pool', name='graph_avg')(x_outs)
        x_out = layers.Dropout(self.dropout)(x_out)
        x_out = GraphConvolution(self.layer_sizes[1], activation='relu', name='GC_2')([x_out, adj_inputs])

        # softmax多分类预测
        x_out = Activation('softmax')(layers.Dense(units=6, name='pred_score')(x_out)) 
        feature_concat = Concatenate()(feature_inputs)
        x_out = Loss(e=self.e)([x_out, target, out_indices, weight, feature_concat])
        self.model = Model(inputs=x_input, outputs=x_out)

        def loss_function(y_true, y_pred):
            return y_pred[0]
        
        self.model.compile(
            optimizer=AdamW(weight_decay=self.wd, learning_rate=self.lr),
            loss=loss_function,
        )

    def load_data(self):
        adj_indices = np.load('./data/adjIndices.npy')
        adj_values = np.load('./data/adjValues.npy')
        feature_building = np.load('./data/featureBuilding.npy')
        feature_lc = np.load('./data/featureLC.npy')
        feature_poi = np.load('./data/featurePOI.npy')
        feature_mobility = np.load('./data/featureMobility.npy')
        feature_rhythm = np.load('./data/featureRhythm.npy')

        # self.all_out_indices = np.load('./data/allOutIndices.npy')[0]
        self.all_out_indices = np.array([i for i in range(1514)])
        self.all_output = np.load('./data/allOutput.npy')

        self.default_train_set = np.load('./data/trainSet.npy')[0]

        self.input_seq_generator = InputSequenceGenerator(adj_indices, adj_values, [feature_lc, feature_poi, feature_building, feature_mobility, feature_rhythm], self.weights)

    def train(self, train_set, callback):
        # 生成训练集输入Sequence
        train_seq = self.input_seq_generator.seq(train_set, self.all_output[:, train_set])
        
        # 验证数据集Sequence用于计算损失函数
        val_set = np.setdiff1d(self.all_out_indices, train_set)
        val_seq = self.input_seq_generator.seq(val_set, self.all_output[:, val_set])

        def lr_scheduler(epoch):
            if epoch == 200:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr * 0.1)
            if epoch == 300:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr * 0.1)
            if epoch == 400:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr * 0.1)
            return K.get_value(self.model.optimizer.lr)

        def wd_schedule(epoch):
            if epoch == 200:
                wd = K.get_value(self.model.optimizer.weight_decay)
                K.set_value(self.model.optimizer.weight_decay, wd * 0.1)
            if epoch == 300:
                wd = K.get_value(self.model.optimizer.weight_decay)
                K.set_value(self.model.optimizer.weight_decay, wd * 0.1)
            if epoch == 400:
                wd = K.get_value(self.model.optimizer.weight_decay)
                K.set_value(self.model.optimizer.weight_decay, wd * 0.1)
            return K.get_value(self.model.optimizer.weight_decay)

        reduce_lr = LearningRateScheduler(lr_scheduler)
        reduce_wd = WeightDecayScheduler(wd_schedule)
        es_callback = EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)

        self.model.fit(
            train_seq,
            epochs=500,
            validation_data=val_seq,
            verbose=2,
            shuffle=True,
            callbacks=[
            reduce_lr,
            reduce_wd,
            es_callback,
            LambdaCallback(on_epoch_end=lambda epoch, logs: callback(epoch, logs))
            ],
        )

        self.int_grad = IntegratedGradients(self.model, train_seq)

    def prediction(self, train_set):
        self.train_set = train_set
        test_set = np.setdiff1d(self.all_out_indices, train_set)
        test_seq = self.input_seq_generator.seq(test_set, self.all_output[:, test_set])

        # 计算整个分类的结果
        all_seq = self.input_seq_generator.seq(self.all_out_indices, self.all_output[:, self.all_out_indices])

        all_scores = self.model.predict(all_seq)[1].squeeze()
        test_scores = self.model.predict(test_seq)[1].squeeze()

        all_preds = self.decode(all_scores)
        all_true = self.decode(self.all_output.squeeze()[self.all_out_indices])
        test_preds = self.decode(test_scores)
        test_true = self.decode(self.all_output.squeeze()[test_set])

        Y = self.all_output.squeeze()[self.all_out_indices]
        Y_ = self.encode(all_preds)
        overall_single_result = classification_report(all_true, all_preds, target_names=self.classes, output_dict=True)
        overall_kappa = cohen_kappa_score([i.argmax() for i in Y], [i.argmax() for i in Y_])


        test_single_result = classification_report(test_true, test_preds, target_names=self.classes, output_dict=True)
        test_kappa = cohen_kappa_score([i.argmax() for i in self.all_output.squeeze()[test_set]], [i.argmax() for i in self.encode(test_preds)])

        self.result = {'id': self.id, 'score': list([list([np.float(j) for j in i]) for i in all_scores]), 'pred': list(all_preds), 'all_result': overall_single_result, 'test_result': test_single_result, 'all_kappa': overall_kappa, 'test_kappa': test_kappa}

    def get_ig(self, rid):
        return [[np.float(j) for j in self.int_grad.get_self_feature_igs(rid, i, steps=50, features_baseline='mean')] for i in range(6)]

class WeightDecayScheduler(Callback):
    def __init__(self, schedule, verbose=0):
        super(WeightDecayScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'weight_decay'):
            raise ValueError('Optimizer must have a "weight_decay" attribute.')
        try:  # new API
            weight_decay = float(K.get_value(self.model.optimizer.weight_decay))
            weight_decay = self.schedule(epoch, weight_decay)
        except TypeError:  # Support for old API for backward compatibility
            weight_decay = self.schedule(epoch)
        if not isinstance(weight_decay, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.weight_decay, weight_decay)
        if self.verbose > 0:
            print('\nEpoch %05d: WeightDecayScheduler reducing weight '
                  'decay to %s.' % (epoch + 1, weight_decay))
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['weight_decay'] = K.get_value(self.model.optimizer.weight_decay)

class InputSequence(Sequence):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.A_indices = inputs[2]
        self.A_values = inputs[3]
        self.features = inputs[0]
        self.weight_matrix = inputs[5]
        self.n_features = sum([i.shape[-1] for i in self.features])
        self.n_nodes = self.features[0].shape[1]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.inputs, self.targets

class InputSequenceGenerator:
    def __init__(self, adj_indices, adj_values, features, weights):
        # weights: [f, c, w]
        self.weights = weights
        self.adj_indices = adj_indices
        self.adj_values = adj_values
        self.features = features
        self.targets = None
        self.inputs = None
        self.weight_matrix = None

    def seq(self, node_ids, targets=None):
        target_indices = np.asanyarray(node_ids)
        target_indices = np.reshape(target_indices, (1,) + target_indices.shape)

        if targets is not None:
            self.targets = targets
        self.weight_matrix = np.zeros([1, 1602, 6])
        for w in self.weights:
            self.weight_matrix[0][w[0]][w[1]] = w[2] / len(self.weights)
        self.inputs = [self.features, target_indices, self.adj_indices, self.adj_values, self.targets, self.weight_matrix]
        return InputSequence(self.inputs, self.targets)
class IntegratedGradients:
  def __init__(self, model, seq):
    self._adj_value = seq.A_values
    self._adj_inds = seq.A_indices
    self._features = seq.features
    self._weight_matrix = seq.weight_matrix
    self._targets = seq.targets
    self._model = model
    self._num_node = seq.n_nodes
    self._node_idx = None

  def get_integrated_gradients(self, node_idx, class_of_interest, features_baseline=None, steps=20):
    if features_baseline is None:
      features_baseline = np.zeros(self._features.shape)
    if features_baseline == 'mean':
      features_baseline = K.expand_dims(self._features.mean(1).repeat(self._num_node, axis=0), 0)
    features_diff = self._features - features_baseline
    total_gradients = np.zeros(self._features.shape)
    for alpha in np.linspace(0, 1, steps):
      features_step = features_baseline + alpha * features_diff
      model_input = [tf.convert_to_tensor(features_step), tf.convert_to_tensor(np.array([[node_idx]])), tf.convert_to_tensor(self._adj_inds), tf.convert_to_tensor(self._adj_value)]
      grads = self._compute_gradients(model_input, class_of_interest, wrt=model_input[0])
      total_gradients += grads
    return np.squeeze(total_gradients * features_diff / steps, 0)

  def get_feature_igs(self, node_idx, class_of_interest, features_baseline=None, steps=20):
    cat_features = np.concatenate(self._features, axis=-1)
    if features_baseline is None:
      features_baseline = np.zeros(cat_features.shape)
    if features_baseline == 'mean':
      features_baseline = K.expand_dims(cat_features.mean(1).repeat(self._num_node, axis=0), 0)
    features_diff = cat_features - features_baseline
    total_gradients = np.zeros(cat_features.shape)
    for alpha in np.linspace(0, 1, steps):
      features_step = features_baseline + alpha * features_diff
      features_input = [features_step[:,:,:19], features_step[:,:,19:36], features_step[:,:,36:40], features_step[:,:,40:1554], features_step[:,:,1554:]]
      model_input = [[tf.convert_to_tensor(f) for f in features_input], tf.convert_to_tensor(np.array([[node_idx]])), tf.convert_to_tensor(self._adj_inds), tf.convert_to_tensor(self._adj_value)]
      grads = self._compute_gradients(model_input, class_of_interest, wrt=model_input[0])
      total_gradients += tf.concat(grads, -1)
    return np.squeeze(total_gradients * features_diff / steps, 0)

  def get_self_feature_igs(self, node_idx, class_of_interest, features_baseline=None, steps=20):
    self._node_idx = node_idx
    cat_features = np.concatenate(self._features, axis=-1)
    num_features = cat_features.shape[2]
    if features_baseline is None:
      features_baseline = np.zeros(num_features)
    if features_baseline == 'mean':
      features_baseline = cat_features.mean(1)[0]
    features_diff = cat_features[0][node_idx] - features_baseline
    total_gradients = np.zeros(num_features)
    for alpha in np.linspace(0, 1, steps):
      features_step = features_baseline + alpha * features_diff
      features_input = [features_step[:19], features_step[19:36], features_step[36:40], features_step[40:1554], features_step[1554:]]
      wrt = [tf.convert_to_tensor(f, dtype='float32') for f in features_input]
      model_input = [[tf.convert_to_tensor(f) for f in self._features], tf.convert_to_tensor(np.array([[node_idx]])), tf.convert_to_tensor(self._adj_inds), tf.convert_to_tensor(self._adj_value), tf.convert_to_tensor(self._targets[:, 0:1, :]), tf.convert_to_tensor(self._weight_matrix)]
      grads = self._compute_self_gradients(model_input, class_of_interest, wrt=wrt)
      total_gradients += tf.concat(grads, -1)
    return np.array(total_gradients * features_diff / steps)

  def _compute_self_gradients(self, model_input, class_of_interest, wrt):
    sup_1 = [np.zeros((1, self._num_node, i.shape[0])) for i in wrt]
    sup_2 = [np.ones((1, self._num_node, i.shape[0])) for i in wrt]
    for i in range(len(self._features)):
      sup_1[i][0][self._node_idx] = np.ones(wrt[i].shape)
      sup_2[i][0][self._node_idx] = np.zeros(wrt[i].shape)
    sup_1 = [tf.convert_to_tensor(f, dtype='float32') for f in sup_1]
    sup_2 = [tf.convert_to_tensor(f, dtype='float32') for f in sup_2]
    class_of_interest = tf.convert_to_tensor(class_of_interest)
    with tf.GradientTape() as tape:
      tape.watch(wrt)
      for i in range(len(self._features)):
        model_input[0][i] = tf.multiply(sup_1[i], K.expand_dims(tf.tile(input=K.expand_dims(wrt[i], 0), multiples=[self._num_node, 1]), 0)) + tf.multiply(model_input[0][i], sup_2[i])
      output = self._model(model_input)[1]
      cost_value = K.gather(output[0, 0], class_of_interest)
    gradients = tape.gradient(cost_value, wrt)
    return gradients
  
  def _compute_gradients(self, model_input, class_of_interest, wrt):
    class_of_interest = tf.convert_to_tensor(class_of_interest)
    with tf.GradientTape() as tape:
      tape.watch(wrt)
      output = self._model(model_input)[1]
      cost_value = K.gather(output[0, 0], class_of_interest)
    gradients = tape.gradient(cost_value, wrt)
    return gradients