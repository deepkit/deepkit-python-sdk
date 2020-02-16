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


short_name_prog = re.compile(r'\[([a-zA-Z0-9]+)\]')
is_variable = re.compile(r'/([a-zA-Z_0-9]+(?:\.[0-9]+)?)$')


def get_layer_id(name: str):
    """
    Takes a name like 'ResNet/Conv2d[conv1]/1504' and converts it back to
    the name from named_modules method, e.g. conv1
    Examples
        1. 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/1658'
        -> layer1.1.conv2
        2. 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]/1714'
        -> layer2.0.bn1
        3. 'ResNet/Sequential[layer1]/BasicBlock[0]/input.4'
        -> layer1.0/input.4
    """
    res = short_name_prog.findall(name)
    var = is_variable.search(name)
    if not res:
        return name, False
    if var:
        return '.'.join(res) + '/' + var.group(1), True
    return '.'.join(res), False


def get_short_layer_id(name: str):
    """
    Takes a name like 'ResNet/Conv2d[conv1]/1504' and converts it back to
    the name from named_modules method, e.g. conv1
    Examples
        1. 'ResNet/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/1658'
        -> layer1.1.conv2
        2. 'ResNet/Sequential[layer2]/BasicBlock[0]/BatchNorm2d[bn1]/1714'
        -> layer2.0.bn1
        3. 'ResNet/Sequential[layer1]/BasicBlock[0]/input.4'
        -> layer1.0
    """
    res = short_name_prog.findall(name)
    if not res:
        return name
    return '.'.join(res)


def get_pytorch_graph(net, x):
    names_from_id = dict()
    names_is_variable = dict()
    nodes_from_id = dict()
    names_from_debug = dict()
    names_short_from_debug = dict()
    names_to_short = dict()

    container_names = dict()
    known_modules_map = dict()
    known_modules_name_map = dict()

    torch_graph, torch_nodes = build_graph(net, x)

    for name, module in net.named_modules():
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

        layer_id, is_variable = get_layer_id(node.debugName)
        if is_variable:
            names_is_variable[layer_id] = node
        names_from_id[layer_id] = node.debugName
        nodes_from_id[layer_id] = node
        names_from_debug[node.debugName] = layer_id
        names_short_from_debug[node.debugName] = get_short_layer_id(node.debugName)
        names_to_short[layer_id] = names_short_from_debug[node.debugName]

    edges = dict()
    edges_internal = dict()

    for node in torch_nodes.values():
        if node.debugName not in names_from_debug: continue
        layer_id = names_from_debug[node.debugName]
        short_layer_id = names_short_from_debug[node.debugName]

        print(node.debugName, '=>', layer_id, short_layer_id, node.kind)
        for parent in get_parent_names(layer_id):
            container_names[parent] = True

        for input in node.inputs:
            if layer_id not in edges_internal: edges_internal[layer_id] = []
            edges_internal[layer_id].append(input)

            if input in names_from_debug and layer_id != names_from_debug[input] \
                    and short_layer_id != names_from_debug[input]:
                print('   outgoing', names_from_debug[input], names_short_from_debug[input], input)
                # this node points out of itself, so create an edge
                edge_to = names_from_debug[input]

                if layer_id in edges:
                    edges[layer_id].add(edge_to)
                else:
                    edges[layer_id] = set([edge_to])

    def resolve_edges_to_known_layer(from_layer: str, inputs: Set[str]) -> List[str]:
        new_inputs = set()
        short_name = names_to_short[from_layer] if from_layer in names_to_short else None
        parent_name = get_parent(short_name) if short_name else None

        # parent_layer = get_parent(from_layer)
        for input in inputs:
            input_short_name = names_to_short[input] if input in names_to_short else None

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

    # edges_resolved = dict()
    # shapes = dict()
    # short_name_to_id = dict()

    # # we resolve the edges only from known layers
    # for [name, inputs] in edges.items():
    #     # first name=layer2.0/input.1 => layer2.0
    #     short_name = name
    #     if name in names_to_short:
    #         short_name = names_to_short[name]
    #
    #     if short_name not in known_modules_name_map and name not in names_is_variable: continue
    #     # if short_name in edges_resolved: continue
    #
    #     shapes[short_name] = nodes_from_id[name].tensor_size
    #     short_name_to_id[short_name] = name
    #     edges_resolved[short_name] = resolve_edges_to_known_layer(name, inputs)
    #
    deepkit_nodes = []

    # for [name, inputs] in edges_resolved.items():
    #     module = known_modules_name_map[name]
    #     node = {
    #         'id': name,
    #         'label': name,
    #         'type': type(module).__name__,
    #         'input': inputs,
    #         'attributes': extract_attributes(module),
    #         'internalInputs': list(edges[short_name_to_id[name]]),
    #         'shape': shapes[name]
    #     }
    #     deepkit_nodes.append(node)
    #
    # allowed_torch_kinds = {
    #     'add', 'div', 'mul', 'add_', 'div_', 'mul_'
    #     'relu', 'sigmoid',
    #     'threshold'
    # }

    # for torch_node in torch_graph.outputs():
    #     print(torch_node.debugName, torch_node.kind)
    #     pass

    # we start from all outputs and go backwards, all visited input nodes are
    # collected. We are interested in only nodes that are in the actual graph
    # of outputs.
    nodes_names_to_display = set()

    def collect_inputs(inputs):
        for input in inputs:
            nodes_names_to_display.add(input)
            if input in edges:
                collect_inputs(edges[input])

    for [name, inputs] in edges.items():
        if name.startswith('output/output.'):
            nodes_names_to_display.add(name)
            collect_inputs(inputs)

    for name in nodes_names_to_display:
        inputs = edges[name] if name in edges else []
    # for [name, inputs] in edges.items():
        torch_node = nodes_from_id[name]
        short_name = names_to_short[name]

        filterer_inputs = []

        for input in inputs:
            second_parent = get_parent(names_to_short[input], 2)
            if second_parent and not short_name.startswith(second_parent):
                continue
            if input.startswith('input/input'):
                filterer_inputs.append(input)
                continue
            if input in edges: filterer_inputs.append(input)

        attributes = {}
        node_type = torch_node.kind
        # that only works when we detected that short_name is unique
        # means that there is only one name that points to short_name. or should we always pick the latest?
        # dunno
        # if short_name in known_modules_name_map:
        #     extract_attributes(known_modules_name_map[short_name])
        #     node_type = type(known_modules_name_map[short_name]).__name__

        attributes['torch.debugName'] = torch_node.debugName
        attributes['torch.kind'] = torch_node.kind
        attributes['torch.inputs'] = torch_node.inputs

        node = {
            'id': name,
            'label': name,
            'type': node_type,
            'input': filterer_inputs,
            'attributes': attributes,
            # 'internalInputs': list(edges[short_name_to_id[name]]),
            'shapes': torch_node.tensor_size,
            # 'shape': shapes[name]
        }
        deepkit_nodes.append(node)


    graph = {
        'nodes': deepkit_nodes
    }

    return graph
