import inspect
import sys
from os.path import dirname
from struct import pack
from typing import Dict, Optional, List

import PIL.Image
import numpy as np

if 'keras' in sys.modules:
    import keras
else:
    import tensorflow.keras as keras

import tensorflow as tf

import deepkit.debugger
from deepkit.utils.image import get_layer_vis_square, get_image_tales, make_image_from_dense

if 'keras' in sys.modules:
    from keras import Model
else:
    from tensorflow.keras import Model


def count_params(weights):
    return int(sum(np.prod(p.shape.as_list()) for p in weights))


def get_tf_shape_as_list(tf_shape_dim):
    return list(map(lambda x: x.size, list(tf_shape_dim)))


def extract_model_graph(model):
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

    def tensor_name_to_node_name(name: str) -> str:
        return name[0:name.rindex(':')]

    def get_parent(name, go_up=1) -> str:
        return '/'.join(name.split('/')[:go_up * -1])

    def get_scope_id(name: str):
        """
        Takes a name like 'dense_2/MatMul' and converts it to its scope `dense_2`.
        Examples
            1. 'dense_1/MatMul/ReadVariableOp/resource'
            -> dense_1/MatMul/ReadVariableOp
            2. 'dense_1/MatMul/ReadVariableOp'
            -> dense_1/MatMul
        """
        return dirname(name)

    edges = dict()
    nodes = dict()
    names_to_scope = dict()
    scope_nodes = dict()
    input_names = []
    output_names = []
    record_map = dict()

    output_tensor = model.outputs[0] if hasattr(model, 'outputs') else model.output
    if not hasattr(output_tensor, 'graph'):
        # only tensorflow has `graph` defined.
        graph = {'nodes': [], 'scopes': []}
        return graph, record_map, input_names

    g = output_tensor.graph
    tf_nodes = list(g.as_graph_def(add_shapes=True).node)
    blacklist = {'Placeholder', 'PlaceholderWithDefault', 'Const'}

    model_scoped_layer_names = set()
    model_unique_layer_names = set()

    def extract_layers(model, scope_id=''):
        scope_prefix = ((scope_id + '/') if scope_id else '')
        for layer in model.layers:
            model_scoped_layer_names.add(scope_prefix + layer.name)
            model_unique_layer_names.add(layer.name)

            if isinstance(layer, Model):
                extract_layers(layer, scope_prefix + layer.name)

    extract_layers(model, '')

    def get_scoped_name(full_name: str):
        """
        1. 'sequential_1/conv2d_1/convolution/ReadVariableOp'
        => 'sequential_1', 'conv2d_1'
        2. 'conv2d_1/convolution/ReadVariableOp'
        => '', 'conv2d_1'
        3. 'dense_1/MatMul'
        => '', 'dense_1'
        """
        names = full_name.split('/')
        scope = ''
        name = ''

        for part in names:
            next_scope = scope + ('/' if scope else '') + name
            next_name = part
            next_full_name = next_scope + ('/' if next_scope else '') + next_name
            if next_full_name not in model_scoped_layer_names:
                break
            scope = next_scope
            name = next_name

        if not scope and not name and names[0] in model_unique_layer_names:
            return '', names[0]

        return scope, name

    for tensor in model.inputs:
        input_names.append(tensor_name_to_node_name(tensor.name))

    for tensor in model.outputs:
        output_names.append(tensor_name_to_node_name(tensor.name))

    for node in tf_nodes:
        is_input = node.name in input_names
        if node.op in blacklist and not is_input: continue

        nodes[node.name] = node
        scope_id = get_scope_id(node.name)
        names_to_scope[node.name] = scope_id

        if scope_id not in scope_nodes:
            scope_nodes[names_to_scope[node.name]] = []

        scope_nodes[scope_id].append(node.name)

    for node in nodes.values():
        edges[node.name] = set()
        node_scope, node_name = get_scoped_name(node.name)

        for input in node.input:
            # filter unknown nodes
            if input not in nodes: continue

            input_scope, input_name = get_scoped_name(input)
            if input_name == node_name and node_scope != input_scope:
                # 'sequential_1/conv2d_1/convolution/ReadVariableOp' points to
                # its internals at 'conv2d_1/kernel', which are both conv2d_1, but
                # on different scopes, which mean we don't display `conv2d_1/kernel`, since
                # its only internals.
                continue

            edges[node.name].add(input)

    nodes_names_to_display = set()

    def collect_nodes_to_display(inputs):
        for input in inputs:
            if input not in nodes_names_to_display:
                nodes_names_to_display.add(input)
                if input in edges:
                    collect_nodes_to_display(edges[input])

    dk_nodes = []
    dk_scopes = []

    primitive = {'Identity'}

    # shows those layers activation nodes.
    activations = {'elu', 'softmax', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                   'exponential', 'linear', 'leakyrelu'}

    op_names_normalization = {'AddV2': 'add'}

    # show as type 'layer' when no `activation` or linear activation has been set. This
    # hides internals of those layers in the graph.
    layers = {'Embedding', 'Flatten', 'Dense', 'Dropout', 'Reshape', 'BatchNormalization', 'UpSampling2D', 'Conv2D'}

    for i, output in enumerate(model.outputs):
        name = tensor_name_to_node_name(output.name)
        nodes_names_to_display.add(name)
        collect_nodes_to_display(edges[name])
        record_map[output.name] = name + ':0'

        shape = []
        if '_output_shapes' in nodes[name].attr:
            shape = list(map(lambda x: x.size, list(nodes[name].attr['_output_shapes'].list.shape[0].dim)))

        dk_nodes.append({
            'id': output.name,
            'label': name,
            'type': 'output',
            'subType': '',
            'input': list(edges[name]),
            'attributes': {},
            'recordable': True,
            'scope': '',
            'shape': shape,
        })

    for name in nodes_names_to_display:
        scope_id = names_to_scope[name]
        node_label = name
        node_type = 'op'
        node_sub_type: str = nodes[name].op
        inputs = edges[name] if name in edges else []
        recordable = True
        shape = []
        if '_output_shapes' in nodes[name].attr:
            shape = list(map(lambda x: x.size, list(nodes[name].attr['_output_shapes'].list.shape[0].dim)))

        if name in input_names:
            recordable = True
            node_type = 'input'

        if name in output_names:
            recordable = True
            node_type = 'output'

        if node_sub_type in primitive:
            node_type = 'primitive'

        if node_sub_type.lower() in activations:
            node_type = 'activation'

        if node_sub_type in op_names_normalization:
            node_sub_type = op_names_normalization[node_sub_type]

        if recordable:
            # map names to tensor, which can be later used to fetch the output
            try:
                g.get_tensor_by_name(name + ':0')
                record_map[name] = name + ':0'
            except:
                recordable = False

        is_collapsible = node_sub_type != 'Sequential'
        if scope_id and is_collapsible and scope_id in scope_nodes and len(scope_nodes[scope_id]) == 1:
            # the scope has only one item, so collapse it.
            parent_scope_id = get_parent(scope_id)
            if scope_id not in nodes:
                scope_id = parent_scope_id

        # is_collapsible = node_sub_type != 'Sequential'
        # while scope_id and is_collapsible and scope_id in scope_nodes and len(scope_nodes[scope_id]) == 1:
        #     # the scope has only one item, so collapse it.
        #     scope_id = get_parent(scope_id)
        #     if scope_id in nodes:
        #         is_collapsible = nodes[scope_id].op != 'Sequential'
        #         inputs = edges[scope_id]
        #     else:
        #         is_collapsible = False

        node = {
            'id': name,
            'label': node_label,
            'type': node_type,
            'subType': node_sub_type,
            'input': list(inputs),
            'attributes': {},
            'recordable': recordable,
            'scope': scope_id,
            'shape': shape,
        }
        dk_nodes.append(node)

    def extract_layers(model, scope_id=''):
        if scope_id:
            dk_scopes.append({
                'id': scope_id,
                'label': scope_id,
                'subType': type(model).__name__,
                'recordable': True,
            })
        scope_prefix = ((scope_id + '/') if scope_id else '')

        for layer in model.layers:
            recordable = True
            if hasattr(layer, 'outputs'):
                recordable = len(layer.outputs) == 1
            else:
                recordable = layer.output is not None

            if recordable:
                # we track here the actual layer, because it contains the weights/biases correctly
                tensor = layer.outputs[0] if hasattr(layer, 'outputs') else layer.output
                # sub tensors must have the layer name as prefix. If this is not the case
                # it references the wrong tensor. We make here sure the correct sub tensor
                # is chosen and not one from a shadow/sibling graph.
                # 1. scope_prefix='', layer=Dense1 and tensor is like 'dense_1/Relu' which is correct
                # 2. scope_prefix='', layer=Sequential1 and tensor is like 'activation/tanh', but we need 'sequential_1/activation/tanh'
                # 3. scope_prefix='sequential_1', layer=Dense2, tensor is like 'dense_1/Relu', but we need 'sequential_1/dense_2/Relu'
                if '/' in tensor.name and not tensor.name.startswith(scope_prefix + layer.name + '/'):
                    if tensor.name.startswith(layer.name + '/'):
                        tensor = g.get_tensor_by_name(scope_prefix + tensor.name)
                    else:
                        tensor = g.get_tensor_by_name(scope_prefix + layer.name + '/' + tensor.name)

                record_map[scope_prefix + layer.name] = {
                    'layer': layer,
                    'tensor': tensor
                }

            node_sub_type = type(layer).__name__
            node_type = 'scope'

            if node_sub_type in layers:
                if not hasattr(layer, 'activation') or layer.activation is None or layer.activation.__name__ == 'linear':
                    # once the layer has a custom activation function, we don't collapse it to a layer type
                    # since that wouldn't be visible in the graph anymore.
                    node_type = 'layer'

            dk_scopes.append({
                'id': scope_prefix + layer.name,
                'label': layer.name,
                'type': node_type,
                'subType': node_sub_type,
                'attributes': extract_attributes(layer),
                'recordable': recordable,
                'shape': layer.output_shape,
            })

            if isinstance(layer, Model):
                extract_layers(layer, scope_prefix + layer.name)

    extract_layers(model, '')

    graph = {'nodes': dk_nodes, 'scopes': dk_scopes}
    return graph, record_map, input_names


class TFDebugger:
    def __init__(self, debugger: deepkit.debugger.DebuggerManager, model, model_input, graph_name: str,
                 record_map: dict, is_batch: bool, input_names: List[str]):
        self.debugger = debugger
        self.model = model
        self.model_input = model_input
        self.graph_name = graph_name
        self.input_names = input_names
        self.is_batch = is_batch

        # contains a map of recording map, names from nodes of the full graph to actual modules
        # this is necessary since we map certain internal nodes to a scope/layer/module.
        self.record_map = record_map

        self.fetch_result: Dict[str, deepkit.debugger.DebuggerFetchItem] = dict()
        self.fetch_config: Optional[deepkit.debugger.DebuggerFetchConfig] = None

    def set_input(self, x):
        # resize batches to size 1 if is_batch=True
        if isinstance(x, tf.data.Dataset):
            x = next(iter(x))[0]

        if len(self.input_names) == 1:
            self.model_input = np.array([x[0]] if self.is_batch else x)
        else:
            self.model_input = [np.array([v[0]]) if self.is_batch else v for v in x]

    def fetch(self, fetch_config: deepkit.debugger.DebuggerFetchConfig) -> Dict[
        str, deepkit.debugger.DebuggerFetchItem]:
        self.fetch_config = fetch_config
        self.fetch_result = dict()

        node_names = []
        for name in self.record_map:
            # if name is an input, we need to fetch it directly from the self.model_input
            # otherwise TF crashes with `input_1:0 is both fed and fetched`
            if name in self.input_names:
                continue

            node_id = self.graph_name + ':' + name
            if self.fetch_config.needs_fetch(node_id):
                node_names.append(name)

        if self.model_input is not None:
            if len(self.input_names) > 1:
                for i, name in enumerate(self.input_names):
                    self._set_item_from_input(i, self.model_input[i])
            elif len(self.input_names) == 1:
                self._set_item_from_input(0, self.model_input)

        if not len(node_names):
            return self.fetch_result

        if self.model_input is None:
            return self.fetch_result

        data = self.get_image_and_histogram_from_layers(self.fetch_config.x, node_names)

        for i, name in enumerate(node_names):
            jpeg, ahistogram = data[i]
            whistogram = None
            bhistogram = None
            tensor_or_layer_dict = self.record_map[name]
            if isinstance(tensor_or_layer_dict, dict):
                layer = tensor_or_layer_dict['layer']
                whistogram, bhistogram = self.get_weight_histogram_from_layer(self.fetch_config.x, layer)

            node_id = self.graph_name + ':' + name
            self.fetch_result[node_id] = deepkit.debugger.DebuggerFetchItem(
                name=node_id,
                output=jpeg,
                ahistogram=ahistogram,
                whistogram=whistogram,
                bhistogram=bhistogram,
            )

        return self.fetch_result

    def _set_item_from_input(self, index, data):
        name = self.input_names[index]
        node_id = self.graph_name + ':' + name

        if not self.fetch_config.needs_fetch(node_id):
            return

        jpeg, ahistogram = self._image_and_histogram(self.fetch_config.x, data)
        self.fetch_result[node_id] = deepkit.debugger.DebuggerFetchItem(
            name=node_id,
            output=jpeg,
            ahistogram=ahistogram,
            whistogram=None,
            bhistogram=None,
        )

    def get_image_and_histogram_from_layers(self, x, names):
        outputs = []
        output_tensor = self.model.outputs[0] if hasattr(self.model, 'outputs') else self.model.output
        g = output_tensor.graph
        for name in names:
            tensor_name_or_layer_dict = self.record_map[name]
            if isinstance(tensor_name_or_layer_dict, str):
                tensor = g.get_tensor_by_name(tensor_name_or_layer_dict)
                outputs.append(tensor)
            else:
                layer_dict = tensor_name_or_layer_dict
                outputs.append(layer_dict['tensor'])

        inputs = self.model.inputs if hasattr(self.model, 'inputs') else self.model.input

        fn = keras.backend.function(inputs, outputs)
        y = fn(self.model_input)

        result = []

        for i, _ in enumerate(names):
            result.append(self._image_and_histogram(x, y[i]))

        return result

    def _image_and_histogram(self, x, output):
        image = None
        histogram = None
        if hasattr(output, 'shape'):
            # tf is not batch per default
            sample = np.copy(output)
            shape = output.shape

            if self.is_batch:
                # display only first item in batch
                sample = np.copy(output[0])
                shape = output.shape[1:] # first is batch shizzle

            if len(shape) == 3:
                if keras.backend.image_data_format() == 'channels_last':
                    sample = np.transpose(sample, (2, 0, 1))

                if sample.shape[0] == 3:
                    image = PIL.Image.fromarray(get_layer_vis_square(sample))
                else:
                    image = PIL.Image.fromarray(get_image_tales(sample))
            elif len(shape) > 1:
                image = PIL.Image.fromarray(get_layer_vis_square(sample))
            elif len(shape) == 1:
                if shape[0] == 1:
                    # we got a single number
                    output = sample[0]
                else:
                    image = make_image_from_dense(sample)

            h = np.histogram(sample, bins=20)
            histogram = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        output_rep = None
        if isinstance(image, PIL.Image.Image):
            output_rep = image
        elif isinstance(output, (float, np.floating)):
            output_rep = float(output)
        elif isinstance(output, (int, np.integer)):
            output_rep = int(output)

        return output_rep, histogram

    def get_weight_histogram_from_layer(self, x, layer):
        layer_weights = layer.get_weights()
        weights = None

        if len(layer_weights) > 0:
            h = np.histogram(layer_weights[0], bins=20)
            # <version><x><bins><...x><...y>, little endian
            # uint8|Uint32|Uint16|...Float32|...Uint32
            # B|L|H|...f|...L
            weights = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        biases = None
        if len(layer_weights) > 1:
            h = np.histogram(layer_weights[1], bins=20)
            biases = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        return weights, biases
