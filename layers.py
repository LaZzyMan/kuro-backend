import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers

from tensorflow.keras.layers import Layer,Average, Concatenate
from tensorflow.keras import activations, initializers, constraints, regularizers

class SqueezedSparseConversion(Layer):
    def __init__(self, shape, axis=0, dtype=None):
        super().__init__(dtype=dtype)

        self.trainable = False
        self.supports_masking = True
        self.matrix_shape = shape
        self.axis = axis

        if K.backend() != "tensorflow":
            raise RuntimeError(
                "SqueezedSparseConversion only supports the TensorFlow backend"
            )

    def get_config(self):
        config = {"shape": self.matrix_shape, "dtype": self.dtype}
        return config

    def compute_output_shape(self, input_shapes):
        return tuple(self.matrix_shape)

    def call(self, inputs):
        if self.axis is not None:
            indices = K.squeeze(inputs[0], self.axis)
            values = K.squeeze(inputs[1], self.axis)
        else:
            indices = inputs[0]
            values = inputs[1]

        if self.dtype is not None:
            values = K.cast(values, self.dtype)

        import tensorflow as tf

        output = tf.SparseTensor(
            indices=indices, values=values, dense_shape=self.matrix_shape
        )
        return output

class GatherIndices(Layer):
    def __init__(self, axis=None, batch_dims=0, **kwargs):
        super().__init__(**kwargs)
        self._axis = axis
        self._batch_dims = batch_dims

    def get_config(self):
        config = super().get_config()
        config.update(axis=self._axis, batch_dims=self._batch_dims)
        return config

    def compute_output_shape(self, input_shapes):
        data_shape, indices_shape = input_shapes
        axis = self._batch_dims if self._axis is None else self._axis
        return (
            data_shape[:axis]
            + indices_shape[self._batch_dims :]
            + data_shape[axis + 1 :]
        )

    def call(self, inputs):
        data, indices = inputs
        return tf.gather(data, indices, axis=self._axis, batch_dims=self._batch_dims)

class GraphConvolution(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        feature_shape, *As_shapes = input_shapes
        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]
        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        feat_shape = input_shapes[0]
        adj_shape = input_shapes[1]
        input_dim = int(feat_shape[-1])
        num_node = int(adj_shape[0])
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name=self.name + "_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.self_loop_weight = self.add_weight(
            shape=1,
            trainable=True,
            name=self.name+'_self_loop_weight',
            initializer=initializers.Ones(),
            constraint=None,
            regularizer=None,
        )
        self.self_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name=self.name + "_self_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name=self.name + "_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features, A = inputs
        num_nodes = features.shape[1]
        assert K.is_sparse(A)
        features = K.squeeze(features, axis=0)
        # ?????????????????????
        h_features = K.dot(features, self.kernel)
        # ????????????????????????
        self_h_features = K.dot(features, self.self_kernel)
        # ?????????????????????
        output = K.dot(A, h_features)
        # ??????????????????????????????????????????
        output = tf.add(tf.multiply(self.self_loop_weight, self_h_features), output)
        output = K.expand_dims(output, axis=0)
        # ???????????????
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)
        return output

class AggregationLayer(Layer):
    def __init__(self, activation, method='avg', kernel_regularizer=None, **kwargs):
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel = None
        self.method = method
        super(AggregationLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {"activation": activations.serialize(self.activation), "kernel_regularizer": regularizers.serialize(self.kernel_regularizer)}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        feature_shape = input_shapes[0]
        return feature_shape

    def build(self, input_shapes):
        num_graph = len(input_shapes)
        num_node = input_shapes[0][1]
        num_feature = input_shapes[0][2]
        if self.method == 'pool':
          self.kernel = self.add_weight(name=self.name+'_pool_kernel', shape=(num_graph * num_feature, num_feature), initializer=initializers.GlorotUniform(), regularizer=self.kernel_regularizer)
        else:
          self.kernel = [self.add_weight(name=self.name+f'_graph_weight_{i}', shape=1, initializer=initializers.Ones(), trainable=True) for i in range(num_graph)]
        self.built = True

    def call(self, inputs):
        if self.method == 'pool':
          out = K.dot(Concatenate()(inputs), self.kernel)
          return self.activation(out)
        else:
          outs = [x * weight for x, weight in zip(inputs, self.kernel)]
          if self.method == 'avg':
            return self.activation(Average()(outs))
          else:
            return self.activation(Concatenate()(outs))

class Loss(Layer):
    def __init__(self, e=0.1, **kwargs):
        self.e = e
        super(Loss, self).__init__(**kwargs)
    
    def get_config(self):
        config = {}
        base_config = super.get_config()
        return {**base_config, **config}
    
    def compute_output_shape(self, input_shapes):
        return [1]
    
    def build(self, input_shapes):
        self.built = True
        
    def call(self, inputs):
        y_pred_all = inputs[0]
        y_true = inputs[1]
        out_indices = inputs[2]
        weight = inputs[3]
        feature = inputs[4]
        
        weight_pos = tf.clip_by_value(weight, 0, 1)
        weight_neg = tf.clip_by_value(weight, -1, 0)
        
        wp_neg = tf.multiply(tf.tile(tf.expand_dims(y_pred_all, 2), [1, 1, 1602, 1]), tf.tile(tf.expand_dims(weight_neg, 1), [1, 1514, 1, 1]))
        wp_pos = tf.multiply(tf.tile(tf.expand_dims((tf.ones_like(y_pred_all) - y_pred_all), 2), [1, 1, 1602, 1]), tf.tile(tf.expand_dims(weight_pos, 1), [1, 1514, 1, 1]))
        
        y_pred = GatherIndices(batch_dims=1)([y_pred_all, out_indices])
        
        # feature = (tf.exp(1 - feature) - 1) / (np.e - 1)
        
        expand_feature = tf.tile(tf.expand_dims(feature, 3), [1, 1, 1, 6])
        loss1 = losses.categorical_crossentropy(y_true, y_pred)
        loss2 = tf.reduce_sum(tf.multiply(wp_pos, expand_feature)) - tf.reduce_sum(tf.multiply(wp_neg, expand_feature))
        return [(1-self.e) * loss1 + self.e * loss2, y_pred]