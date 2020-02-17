import re
from typing import Set, List

from deepkit.pytorch_graph import build_graph

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
short_name_prog = re.compile(r'\[([a-zA-Z0-9]+)\]')
is_variable = re.compile(r'/([a-zA-Z_0-9]+(?:\.[0-9]+)?)$')


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


def get_pytorch_graph(net, x):
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

    torch_graph, torch_nodes = build_graph(net, x)

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
        if node.kind == 'prim::ListConstruct': continue
        if node.kind == 'aten::t': continue

        layer_id = get_layer_id(node.debugName)
        scope_id = get_scope_id(node.debugName)
        names_from_id[layer_id] = node.debugName
        nodes_from_id[layer_id] = node
        names_from_debug[node.debugName] = layer_id
        scopes_from_debug[node.debugName] = scope_id
        names_to_scope[layer_id] = scopes_from_debug[node.debugName]
        if scope_id not in scope_nodes:
            scope_nodes[scope_id] = [layer_id]
        else:
            scope_nodes[scope_id].append(layer_id)
        # names_to_scope[layer_id] = get_scope_name(node.debugName)

    edges = dict()
    edges_internal = dict()

    for node in torch_nodes.values():
        if node.debugName not in names_from_debug: continue
        layer_id = names_from_debug[node.debugName]
        short_layer_id = scopes_from_debug[node.debugName]

        print(node.debugName, '=>', layer_id, short_layer_id, node.kind)
        for parent in get_parent_names(layer_id):
            container_names[parent] = True

        for input in node.inputs:
            if layer_id not in edges_internal: edges_internal[layer_id] = []
            edges_internal[layer_id].append(input)

            if input in names_from_debug and layer_id != names_from_debug[input] \
                    and short_layer_id != names_from_debug[input]:
                print('   outgoing', names_from_debug[input], scopes_from_debug[input], input)
                # this node points out of itself, so create an edge
                edge_to = names_from_debug[input]

                if layer_id in edges:
                    edges[layer_id].add(edge_to)
                else:
                    edges[layer_id] = set([edge_to])

    def resolve_edges_to_known_layer(from_layer: str, inputs: Set[str]) -> List[str]:
        new_inputs = set()
        short_name = names_to_scope[from_layer] if from_layer in names_to_scope else None
        parent_name = get_parent(short_name) if short_name else None

        # parent_layer = get_parent(from_layer)
        for input in inputs:
            input_short_name = names_to_scope[input] if input in names_to_scope else None

            # we skip connection where even the 2. parent is not the same or a child of from_layer.
            # we could make this configurable.
            second_parent = get_parent(input_short_name, 2)
            if second_parent and short_name and not short_name.startswith(second_parent):
                continue

            if input_short_name and short_name and short_name != input_short_name and input_short_name in known_modules_name_map:
                if not parent_name or (parent_name != input_short_name):
                    new_inputs.add(input_short_name)
                    continue

            if input in edges:
                for i in resolve_edges_to_known_layer(from_layer, edges[input]):
                    new_inputs.add(i)
            else:
                # we let it as is
                new_inputs.add(input)

        return list(new_inputs)

    deepkit_nodes = []

    nodes_names_to_display = set()
    scopes = dict()

    def collect_inputs(inputs):
        for input in inputs:
            nodes_names_to_display.add(input)
            if input in edges:
                collect_inputs(edges[input])

    for [name, inputs] in edges.items():
        if name.startswith('output/output.'):
            nodes_names_to_display.add(name)
            collect_inputs(inputs)


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

    for name in nodes_names_to_display:
        inputs = edges[name] if name in edges else []
        # for [name, inputs] in edges.items():
        torch_node = nodes_from_id[name]
        scope_name = names_to_scope[name]

        node_type = 'layer'

        filterer_inputs = []
        if name.startswith('input/input'):
            node_type = 'input'
        if name.startswith('output/output'):
            node_type = 'output'

        for input in inputs:
            # second_parent = get_parent(names_to_scope[input], 2)
            # if second_parent and not scope_name.startswith(second_parent):
            #     continue
            if input.startswith('input/input'):
                filterer_inputs.append(input)
                continue
            if input in edges: filterer_inputs.append(input)

        attributes = {}
        node_sub_type = ''
        node_label = name

        scope_id = scope_name
        recordable = False

        if len(scope_nodes[scope_name]) == 1 and scope_name in known_modules_name_map:
            # this node is at the same time a module(and thus scope), since it only has one node.
            recordable = True
            node_label = scope_name
            module = known_modules_name_map[scope_name]
            node_sub_type = type(module).__name__
            scope_id = get_parent(scope_name)
            attributes = extract_attributes(module)
        else:
            if str(torch_node.kind).startswith('aten::'):
                node_type = 'op'
                node_sub_type = torch_node.kind.replace('aten::', '').strip('_')

        if node_sub_type.lower() in activation_functions:
            node_type = 'activation'
            node_sub_type = node_sub_type

        attributes['torch.debugName'] = torch_node.debugName
        attributes['torch.kind'] = torch_node.kind
        attributes['torch.inputs'] = torch_node.inputs

        # source = str(torch_node.node.debugName).split(' # ')[1].strip() \
        #     if hasattr(torch_node.node, 'debugName') and ' # ' in str(torch_node.node.debugName) else None

        node = {
            'id': name,
            'label': node_label,
            'type': node_type,
            'subType': node_sub_type,
            # 'source': source,
            'input': filterer_inputs,
            'attributes': attributes,
            'recordable': recordable,
            'scope': scope_id.replace('.', '/'),
            'shape': torch_node.tensor_size,
        }
        deepkit_nodes.append(node)

    for name, module in known_modules_name_map.items():

        # skip modules that are already added as nodes
        if name in scope_nodes and len(scope_nodes[name]) == 1: continue

        scope_id = name.replace('.', '/')
        scope = {
            'id': scope_id,
            'label': scope_id,
            'subType': type(module).__name__,
            'recordable': True,
            'attributes': extract_attributes(module)
        }
        scopes[scope_id] = scope

    graph = {
        'nodes': deepkit_nodes,
        'scopes': scopes,
    }

    return graph
