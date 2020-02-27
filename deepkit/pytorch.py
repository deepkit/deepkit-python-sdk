import math
import re
from struct import pack
from typing import Dict, Optional

import PIL.Image
import numpy as np

import deepkit.context
import deepkit.debugger
from deepkit.pytorch_graph import build_graph
from deepkit.utils import array_to_img
from deepkit.utils.image import get_layer_vis_square, get_image_tales

blacklist_attributes = {'weight', 'dump_patches'}


def extract_attributes(module):
    res = {}
    for attr in dir(module):
        if attr in blacklist_attributes: continue
        if attr.startswith('_'): continue
        val = getattr(module, attr)
        if not isinstance(val, (str, bool, int, float, list, tuple)):
            continue
        res[attr] = val

    return res


scope_name_prog = re.compile(r'^([a-zA-Z0-9_\-]+)/')
short_name_prog = re.compile(r'\[([a-zA-Z0-9_]+)\]')
is_variable = re.compile(r'/([a-zA-Z0-9_]+(?:\.[0-9]+)?)$')


def get_layer_id(name: str):
    """
    Takes a name like 'ResNet/Conv2d[conv1]/1504' and converts it to a shorter version

    Examples
        1. 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/1658'
        -> layer1.1.conv2/1657
        2. 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]/1714'
        -> layer2.0.bn1/1714
        3. 'ResNet/Sequential[layer1]/BasicBlock[0]/input.4'
        -> layer1.0/input.4
        4. 'input/input.1'
        -> input-1
        5. 'output/output.1'
        -> output-1
    """
    res = short_name_prog.findall(name)
    var = is_variable.search(name)
    if not res:
        return name
    if var:
        return '.'.join(res) + '/' + var.group(1)
    return '.'.join(res)


def get_scope_id(name: str):
    """
    Takes a name like 'ResNet/Conv2d[conv1]/1504' and converts it to
    its scope variant, which could be later used for `named_modules` method.
    Examples
        1. 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/1658'
        -> Resnet.layer1.1.conv2
        2. 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]/1714'
        -> Resnet.layer2.0.bn1
        2. 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]/input.2'
        -> Resnet.layer2.0.bn1
        3. 'ResNet/Sequential[layer1]/BasicBlock[0]/input.4'
        -> Resnet.layer1.0
        3. 'ResNet/x.1'
        -> Resnet.x
    """
    res = short_name_prog.findall(name)
    if not res:
        # no groups mean its something like Resnet/x.2, which we normalize to Resnet
        return name.split('/')[0]

    scope = scope_name_prog.findall(name)

    return scope[0] + '.' + ('.'.join(res))


def get_pytorch_graph(net, inputs):
    names_from_id = dict()
    nodes_from_id = dict()
    names_from_debug = dict()
    scopes_from_debug = dict()
    names_to_scope = dict()
    scope_nodes = dict()
    # names_to_scope = dict()

    container_names = dict()
    known_modules_map = dict()
    known_modules_name_map = dict()

    torch_graph, torch_nodes = build_graph(net, inputs)

    for name, module in net.named_modules(prefix=type(net).__name__):
        known_modules_map[module] = name
        known_modules_name_map[name] = module

    def get_parent_names(name):
        t = ''
        for i in name.split('.')[:-1]:
            if t:
                t += '.'
            t += i
            yield t

    def get_parent(name, go_up=1) -> str:
        return '.'.join(name.split('.')[:go_up * -1])

    for node in torch_nodes.values():
        if node.kind == 'prim::Constant': continue
        if node.kind == 'prim::GetAttr': continue
        layer_id = get_layer_id(node.debugName)
        scope_id = get_scope_id(node.debugName)

        if node.kind == 'prim::ListConstruct':
            # if that list constrcutor has only inputs of the same scope, ignore it
            all_scope = True
            for input in node.inputs:
                if get_scope_id(input) != scope_id:
                    all_scope = False
                    break
            if all_scope:
                continue

        # if node.kind == 'aten::t': continue

        names_from_id[layer_id] = node.debugName
        nodes_from_id[layer_id] = node
        names_from_debug[node.debugName] = layer_id
        scopes_from_debug[node.debugName] = scope_id
        names_to_scope[layer_id] = scopes_from_debug[node.debugName]
        if scope_id not in scope_nodes:
            scope_nodes[scope_id] = [layer_id]
        else:
            scope_nodes[scope_id].append(layer_id)

    edges = dict()
    edges_internal = dict()

    for node in torch_nodes.values():
        if node.debugName not in names_from_debug: continue
        layer_id = names_from_debug[node.debugName]
        short_layer_id = scopes_from_debug[node.debugName]

        # print(node.debugName, '=>', layer_id, short_layer_id, node.kind, node.tensor_size)
        for parent in get_parent_names(layer_id):
            container_names[parent] = True

        for input in node.inputs:
            if layer_id not in edges_internal: edges_internal[layer_id] = []
            edges_internal[layer_id].append(input)

            if input in names_from_debug and layer_id != names_from_debug[input] \
                    and short_layer_id != names_from_debug[input]:
                # print('   outgoing', names_from_debug[input], scopes_from_debug[input], input,
                #       nodes_from_id[names_from_debug[input]].kind)
                # this node points out of itself, so create an edge
                edge_to = names_from_debug[input]

                if layer_id in edges:
                    edges[layer_id].add(edge_to)
                else:
                    edges[layer_id] = set([edge_to])

    # def resolve_edges_to_known_layer(from_layer: str, inputs: Set[str]) -> List[str]:
    #     new_inputs = set()
    #     short_name = names_to_scope[from_layer] if from_layer in names_to_scope else None
    #     parent_name = get_parent(short_name) if short_name else None
    #
    #     # parent_layer = get_parent(from_layer)
    #     for input in inputs:
    #         input_short_name = names_to_scope[input] if input in names_to_scope else None
    #
    #         # we skip connection where even the 2. parent is not the same or a child of from_layer.
    #         # we could make this configurable.
    #         second_parent = get_parent(input_short_name, 2)
    #         if second_parent and short_name and not short_name.startswith(second_parent):
    #             continue
    #
    #         if input_short_name and short_name and short_name != input_short_name and input_short_name in known_modules_name_map:
    #             if not parent_name or (parent_name != input_short_name):
    #                 new_inputs.add(input_short_name)
    #                 continue
    #
    #         if input in edges:
    #             for i in resolve_edges_to_known_layer(from_layer, edges[input]):
    #                 new_inputs.add(i)
    #         else:
    #             # we let it as is
    #             new_inputs.add(input)
    #
    #     return list(new_inputs)

    deepkit_nodes = []

    nodes_names_to_display = set()
    scopes = dict()

    def collect_inputs(inputs):
        for input in inputs:
            if input not in nodes_names_to_display:
                nodes_names_to_display.add(input)
                if input in edges:
                    collect_inputs(edges[input])

    def find_outputs(name: str, outputs: set):
        kind = nodes_from_id[name].kind

        if kind == 'IO Node' and len(edges[name]) != 1:
            # an IO node with multiple inputs is probably correct already
            outputs.add(name)
            return

        if kind == 'IO Node' or kind == 'prim::TupleConstruct':
            # resolve inputs
            for input in edges[name]:
                find_outputs(input, outputs)
        else:
            outputs.add(name)

    for name in edges.copy().keys():
        if name.startswith('output/'):
            collect_inputs(edges[name])

            # resolve first to first nodes with available shape, and then use those as output
            # this is necessary since tuple outputs come via prim::TupleConstruct and no shape.
            found_outputs = set()
            find_outputs(name, found_outputs)
            i = 0
            # print('found new outputs', name, found_outputs)

            for output in found_outputs:
                i += 1
                new_name = 'output/output.' + str(i)
                edges[new_name] = edges[name]
                nodes_from_id[new_name] = nodes_from_id[output]
                names_to_scope[new_name] = ''

                nodes_names_to_display.add(new_name)

    activation_functions = {
        'ReLU6'.lower(),
        'LogSigmoid'.lower(),
        'LeakyReLU'.lower(),
        'MultiheadAttention'.lower(),
        'elu', 'hardshrink', 'hardtanh', 'leaky_relu', 'logsigmoid', 'prelu',
        'rrelu', 'relu',
        'sigmoid', 'elu', 'celu', 'selu', 'glu', 'gelu', 'softplus', 'softshrink', 'softsign',
        'tanh', 'tanhshrink',
        'softmin', 'softmax', 'softmax2d', 'log_softmax', 'LogSoftmax'.lower(),
        'AdaptiveLogSoftmaxWithLoss'.lower()
    }

    input_names = []
    output_names = []

    record_map = dict()
    for name in nodes_names_to_display:
        inputs = edges[name] if name in edges else []
        # for [name, inputs] in edges.items():
        torch_node = nodes_from_id[name]
        scope_name = names_to_scope[name]
        if not name:
            raise Exception('No name given')

        node_type = 'layer'
        scope_id = scope_name
        recordable = False

        # filterer_inputs = []
        if name.startswith('input/'):
            recordable = True
            node_type = 'input'
            input_names.append(name)

        if name.startswith('output/'):
            recordable = True
            node_type = 'output'
            output_names.append(name)

        # for input in inputs:
        #     # second_parent = get_parent(names_to_scope[input], 2)
        #     # if second_parent and not scope_name.startswith(second_parent):
        #     #     continue
        #     if input.startswith('input/input'):
        #         filterer_inputs.append(input)
        #         continue
        #     if input in edges: filterer_inputs.append(input)

        attributes = {}
        node_sub_type = ''
        node_label = name

        if node_type != 'output':
            if scope_name and scope_name in scope_nodes and len(
                    scope_nodes[scope_name]) == 1 and scope_name in known_modules_name_map:
                # this node is at the same time a module(and thus scope), since it only has one node.
                recordable = True
                record_map[scope_name] = name
                node_label = scope_name
                module = known_modules_name_map[scope_name]
                node_sub_type = type(module).__name__
                scope_id = get_parent(scope_name)
                attributes = extract_attributes(module)
            else:
                if str(torch_node.kind).startswith('aten::'):
                    node_type = 'op'
                    node_sub_type = torch_node.kind.replace('aten::', '').strip('_')

                if str(torch_node.kind).startswith('prim::'):
                    node_type = 'primitive'
                    node_sub_type = torch_node.kind.replace('prim::', '').strip('_')

            if node_sub_type.lower() in activation_functions:
                node_type = 'activation'
                node_sub_type = node_sub_type

        # attributes['torch.debugName'] = torch_node.debugName
        # attributes['torch.kind'] = torch_node.kind
        # attributes['torch.inputs'] = ', '.join(torch_node.inputs)

        # source = str(torch_node.node.debugName).split(' # ')[1].strip() \
        #     if hasattr(torch_node.node, 'debugName') and ' # ' in str(torch_node.node.debugName) else None

        node = {
            'id': name,
            'label': node_label,
            'type': node_type,
            'subType': node_sub_type,
            # 'source': source,
            'input': list(inputs),
            'attributes': attributes,
            'recordable': recordable,
            'scope': scope_id.replace('.', '/'),
            'shape': torch_node.tensor_size,
        }
        deepkit_nodes.append(node)

    for name, module in known_modules_name_map.items():

        # skip modules that are already added as nodes
        if name in scope_nodes and len(scope_nodes[name]) == 1:
            continue

        scope_id = name.replace('.', '/')
        record_map[name] = scope_id

        # the root scope is not recordable. For that we have global input and outputs
        recordable = '/' in scope_id

        scope = {
            'id': scope_id,
            'label': scope_id,
            'subType': type(module).__name__,
            'recordable': recordable,
            'attributes': extract_attributes(module)
        }
        scopes[scope_id] = scope

    graph = {
        'nodes': deepkit_nodes,
        'scopes': scopes,
    }

    return graph, record_map, input_names, output_names


class TorchDebugger:
    def __init__(self, debugger: deepkit.debugger.DebuggerManager, net, graph_name: str, resolve_map):
        self.known_modules_map = dict()
        self.known_modules_name_map = dict()
        self.debugger = debugger

        for name, module in net.named_modules(prefix=type(net).__name__):
            self.known_modules_map[module] = name
            self.known_modules_name_map[name] = module

        self.net = net
        self.graph_name = graph_name
        self.resolve_map = resolve_map

        # contains a map of recording map, names from nodes of the full graph to actual modules
        # this is necessary since we map certain internal nodes to a scope/layer/module.
        self.record_map = dict()
        self.model_input_names = []
        self.model_output_names = []
        self.model_input = None
        self.extract_graph = False

        self.fetch_result: Dict[str, deepkit.debugger.DebuggerFetchItem] = dict()
        self.fetch_config: Optional[deepkit.debugger.DebuggerFetchConfig] = None

        def root_hook(module, input):
            if self.extract_graph: return
            if self.debugger.active_debug_data_for_this_run: return

            if self.model_input is None:
                self.model_input = input
                self.extract_graph = True
                self.record_map, self.model_input_names, self.model_output_names = self.resolve_map(input)
                self.extract_graph = False
            else:
                self.debugger.tick()

        net.register_forward_pre_hook(root_hook)

        self.net.apply(self.register_hook)

    def fetch(self, fetch_config: deepkit.debugger.DebuggerFetchConfig) -> Dict[str, deepkit.debugger.DebuggerFetchItem]:
        self.fetch_config = fetch_config
        self.fetch_result = dict()

        if not self.model_input:
            return self.fetch_result

        if len(self.model_input_names) > 1:
            for i, name in enumerate(self.model_input_names):
                self.send_debug(name, self.net, self.model_input[i])
        elif len(self.model_input_names) == 1:
            self.send_debug(self.model_input_names[0], self.net, self.model_input)

        self.net(*self.model_input)

        return self.fetch_result

    def register_hook(self, module):
        def hook(module, input, output):
            if self.extract_graph: return
            if not self.debugger.active_debug_data_for_this_run:
                # we don't care about hook calls outside of our debug tracking
                return

            module_id = self.known_modules_map[module]
            node_id = module_id
            if '.' not in module_id:
                # we are in the root module, so we use that for global output tracking
                if len(self.model_output_names) > 1:
                    for i, name in enumerate(self.model_output_names):
                        self.send_debug(name, module, output[i])
                elif len(self.model_output_names) == 1:
                    self.send_debug(self.model_output_names[0], module, output)
            else:
                # sub node
                self.send_debug(node_id, module, output)

        module.register_forward_hook(hook)

    def make_image_from_dense(self, neurons):
        cols = int(math.ceil(math.sqrt(len(neurons))))

        even_length = cols * cols
        diff = even_length - len(neurons)
        if diff > 0:
            neurons = np.append(neurons, np.zeros(diff, dtype=neurons.dtype))

        img = array_to_img(neurons.reshape((1, cols, cols)))
        img = img.resize((cols * 8, cols * 8))

        return img

    def get_histogram(self, x, tensor):
        h = np.histogram(tensor.detach().numpy(), bins=20)
        # <version><x><bins><...x><...y>, little endian
        # uint8|Uint32|Uint16|...Float32|...Uint32
        # B|L|H|...f|...L
        return pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

    def get_debug_data(self, x, module, output):
        image = None
        activations = None
        if isinstance(output, tuple) and len(output) > 0:
            output = output[0]

        if hasattr(output, 'shape'):
            activations = self.get_histogram(x, output)

            if len(output.shape) > 0:
                # outputs come in batch usually, so pick first
                sample = output[0].detach().numpy()
                if len(sample.shape) == 3:
                    if sample.shape[0] == 3:
                        image = PIL.Image.fromarray(get_layer_vis_square(sample))
                    else:
                        image = PIL.Image.fromarray(get_image_tales(sample))
                elif len(sample.shape) > 1:
                    image = PIL.Image.fromarray(get_layer_vis_square(sample))
                else:
                    image = self.make_image_from_dense(sample)
        # elif isinstance(output[0], (float, str, int)):
        #     image = output

        whistogram = None
        bhistogram = None

        if hasattr(module, 'weight') and module.weight is not None:
            whistogram = self.get_histogram(x, module.weight)

        if hasattr(module, 'bias') and module.bias is not None:
            bhistogram = self.get_histogram(x, module.bias)

        output_rep = None
        if isinstance(image, PIL.Image.Image):
            output_rep = image
        elif isinstance(output, (float, int)):
            output_rep = output

        return output_rep, activations, whistogram, bhistogram

    def send_debug(self, node_id, module, output):
        if node_id in self.record_map:
            node_id = self.record_map[node_id]
        node_id = self.graph_name + ':' + node_id

        if self.fetch_config.needs_fetch(node_id):
            output_rep, ahistogram, whistogram, bhistogram = self.get_debug_data(
                self.fetch_config.x, module, output
            )

            self.fetch_result[node_id] = deepkit.debugger.DebuggerFetchItem(
                name=node_id,
                output=output_rep,
                ahistogram=ahistogram,
                whistogram=whistogram,
                bhistogram=bhistogram,
            )