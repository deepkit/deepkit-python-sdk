import inspect
import sys

import numpy as np

if 'keras' in sys.modules:
    from keras import Model
else:
    from tensorflow.keras import Model


def count_params(weights):
    return int(sum(np.prod(p.shape.as_list()) for p in weights))


def layer_visitor(model, callback):
    def walk(entry):
        for layer in entry.layers:
            callback(layer)
            if isinstance(layer, Model):
                walk(layer)
    walk(model)


def extract_model_graph(model):
    dk_graph = {'nodes': []}

    def extract_attributes(layer):
        attrs = [
            # InputLayer
            'input_shape', 'batch_size', 'dtype', 'sparse', 'ragged',

            # conv, LocallyConnected1D
            'rank', 'filters', 'kernel_size', 'strides', 'padding', 'data_format', 'dilation_rate',
            'use_bias',
            'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
            'activity_regularizer', 'kernel_constraint', 'bias_constraint',

            # pooling
            'pool_size', 'strides', 'padding', 'data_format',
            'pool_function',

            # RNN
            'cell', 'return_sequences', 'return_state', 'go_backwards', 'stateful', 'unroll', 'time_major',
            # RNNCell
            'units', 'recurrent_activation', 'use_bias', 'kernel_initializer', 'recurrent_initializer',
            'bias_initializer', 'unit_forget_bias', 'kernel_regularizer', 'recurrent_regularizer',
            'bias_regularizer', 'kernel_constraint', 'recurrent_constraint', 'bias_constraint',
            'dropout', 'recurrent_dropout', 'implementation',

            # Embedding
            'input_dim', 'output_dim',
            'embeddings_initializer', 'embeddings_regularizer', 'activity_regularizer',
            'embeddings_constraint', 'mask_zero', 'input_length', 'fused',

            # Merge
            'axes', 'normalize',

            # Noise
            'stddev', 'rate', 'noise_shape',

            # BatchNormalization
            'momentum', 'epsilon', 'center', 'scale',
            'beta_initializer', 'gamma_initializer', 'moving_mean_initializer', 'moving_variance_initializer',
            'beta_regularizer', '', 'gamma_regularizer', 'beta_constraint', 'gamma_constraint', 'renorm',
            'virtual_batch_size', 'adjustment'

                                  'rate', 'noise_shape',  # Dropout
            'data_format',  # Flatten
            'target_shape',  # Reshape
            'dims',  # Permute
            'n',  # RepeatVector
            'function',  # Lambda
            'l1', 'l2',  # ActivityRegularization
            'mask_value',  # Masking
        ]
        res = {}

        def normalize_value(name, v):
            if inspect.isfunction(v):
                return v.__name__

            if isinstance(v, (str, int, float, bool)):
                return v

            if isinstance(v, (list, tuple)):
                return str(v)

            if type(v).__name__ != 'type':
                # todo, if `cell` for RNN we probably want to extract those information as well
                return type(v).__name__

            return str(v)

        for attr in attrs:
            if hasattr(layer, attr):
                res[attr] = getattr(layer, attr)

                res[attr] = normalize_value(attr, res[attr])

        if hasattr(layer, 'activation'):
            if layer.activation:
                res['activation'] = layer.activation.__name__
                # todo get action parameters. `alpha`, etc

        if hasattr(layer, 'trainable_weights'):
            res['trainable_weights'] = count_params(layer.trainable_weights)
        if hasattr(layer, 'non_trainable_weights'):
            res['non_trainable_weights'] = count_params(layer.non_trainable_weights)

        return res

    def extract_layers(model, container):
        for layer in model.layers:
            n = {
                'id': layer.name,
                'label': layer.name,
                'type': type(layer).__name__,
                'input': set(),
                'attributes': extract_attributes(layer),
                'children': []
            }

            for node in layer._inbound_nodes:
                if isinstance(node.inbound_layers, list):
                    for inbound_layer in node.inbound_layers:
                        n['input'].add(inbound_layer.name)
                else:
                    n['input'].add(node.inbound_layers.name)

            n['shape'] = layer.output.shape.as_list()
            n['input'] = list(n['input'])

            if isinstance(layer, Model):
                extract_layers(layer, n['children'])

            container.append(n)

    extract_layers(model, dk_graph['nodes'])
    dk_graph['inputs'] = []
    dk_graph['outputs'] = []
    return dk_graph
