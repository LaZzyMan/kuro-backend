import numpy as np
from tensorflow.keras import layers, losses, Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, LambdaCallback
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Activation
from tensorflow_addons.optimizers import AdamW
from layers import GatherIndices, GraphConvolution, AggregationLayer, SqueezedSparseConversion
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import ParameterGrid

class KuroModel:
    def __init__(self, embedding_size, layer_sizes, dropout, lr, wd) -> None:
        self.classes = ['C', 'G', 'M', 'P', 'R', 'U']
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.model = None

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
        x_input = [feature_inputs, out_indices, adj_indices, adj_values]
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
        x_out = GatherIndices(batch_dims=1)([x_out, out_indices])
        predictions = Activation('softmax')(layers.Dense(units=6, name='pred_score')(x_out))
        self.model = Model(inputs=x_input, outputs=predictions)

        self.model.compile(
            optimizer=AdamW(weight_decay=self.wd, learning_rate=self.lr),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

    def load_data(self):
        adj_indices = np.load('./data/adjIndices.npy')
        adj_values = np.load('./data/adjValues.npy')
        feature_building = np.load('./data/featureBuilding.npy')
        feature_lc = np.load('./data/featureLC.npy')
        feature_poi = np.load('./data/featurePOI.npy')
        feature_mobility = np.load('./data/featureMobility.npy')
        feature_rhythm = np.load('./data/featureRhythm.npy')

        self.all_out_indices = np.load('./data/allOutIndices.npy')[0]
        self.all_output = np.load('./data/allOutput.npy')

        self.default_train_set = np.load('./data/trainSet.npy')[0]

        self.input_seq_generator = InputSequenceGenerator(adj_indices, adj_values, [feature_lc, feature_poi, feature_building, feature_mobility, feature_rhythm])

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
        es_callback = EarlyStopping(monitor="val_acc", patience=200, restore_best_weights=True)

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

    def prediction(self, train_set):
        test_set = np.setdiff1d(self.all_out_indices, train_set)
        test_seq = self.input_seq_generator.seq(test_set, self.all_output[:, test_set])

        # 计算整个分类的结果
        all_seq = self.input_seq_generator.seq(self.all_out_indices)

        all_scores = self.model.predict(all_seq).squeeze()
        test_scores = self.model.predict(test_seq).squeeze()

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

        return {'all_result': overall_single_result, 'test_result': test_single_result, 'all_kappa': overall_kappa, 'test_kappa': test_kappa}

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
        self.n_features = sum([i.shape[-1] for i in self.features])
        self.n_nodes = self.features[0].shape[1]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.inputs, self.targets

class InputSequenceGenerator:
    def __init__(self, adj_indices, adj_values, features):
        self.adj_indices = adj_indices
        self.adj_values = adj_values
        self.features = features
        self.targets = None
        self.inputs = None

    def seq(self, node_ids, targets=None):
        target_indices = np.asanyarray(node_ids)
        target_indices = np.reshape(target_indices, (1,) + target_indices.shape)
        self.inputs = [self.features, target_indices, self.adj_indices, self.adj_values]

        if targets is not None:
            self.targets = targets
        return InputSequence(self.inputs, self.targets)