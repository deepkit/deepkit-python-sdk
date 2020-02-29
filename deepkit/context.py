import asyncio
import atexit
import base64
import json
import os
import signal
import struct
import time
from threading import Lock
from typing import Optional, Callable, NamedTuple, Dict, List

import numpy as np
import psutil
import typedload
from rx import interval

import deepkit.debugger
import deepkit.client
import deepkit.globals
import deepkit.utils
from deepkit.model import ContextOptions


def get_job_config():
    if deepkit.globals.loaded_job_config is None:
        if 'DEEPKIT_TASK_CONFIG' in os.environ:
            print('DEEPKIT_JOB_CONFIG', os.environ['DEEPKIT_JOB_CONFIG'])
            deepkit.globals.loaded_job_config = json.loads(os.environ['DEEPKIT_TASK_CONFIG'])
        else:
            deepkit.globals.loaded_job_config = {
            }

    return deepkit.globals.loaded_job_config


class JobController:
    def stop(self):
        """
        Raising the SIGINT signal in the current process and all sub-processes.
        os.kill() only issues a signal in the current process (without subprocesses).
        CTRL+C on the console sends the signal to the process group (which we need).
        """
        if hasattr(signal, 'CTRL_C_EVENT'):
            # windows. Need CTRL_C_EVENT to raise the signal in the whole process group
            os.kill(os.getpid(), signal.CTRL_C_EVENT)
        else:
            # unix.
            pgid = os.getpgid(os.getpid())
            if pgid == 1:
                os.kill(os.getpid(), signal.SIGINT)
            else:
                os.killpg(os.getpgid(os.getpid()), signal.SIGINT)


class JobDebuggingState(NamedTuple):
    watchingLayers: Dict[str, bool]
    live: bool
    recording: bool

    # 'epoch' | 'second'
    recordingMode: str

    # 'watched' | 'all'
    recordingLayers: str

    recordingSecond: int


class JobDebuggerController:
    def __init__(self, client: deepkit.client.Client):
        self.state: Optional[JobDebuggingState] = None
        self.client = client

    async def connected(self):
        await self._update_watching_layers()

    # registered RPC function
    async def updateWatchingLayer(self):
        await self._update_watching_layers()

    async def _update_watching_layers(self):
        self.state = typedload.load(await self.client.job_action('getDebuggingState'), JobDebuggingState)


class Context:
    def __init__(self, options: ContextOptions = None):
        if options is None:
            options = ContextOptions()

        self.metric_buffer = []
        self.speed_buffer = []
        self.logs_buffer = []
        self.last_throttle_call = dict()

        self.client = deepkit.client.Client(options)
        deepkit.globals.last_context = self
        self.log_lock = Lock()
        self.defined_metrics = {}
        self.shutting_down = False

        self.last_iteration_time = 0
        self.last_batch_time = 0
        self.job_iteration = 0
        self.job_iterations = 0
        self.job_step = 0
        self.job_steps = 0

        self.model_watching = dict()

        self.auto_x_of_metrix = dict()

        self.seconds_per_iteration = 0
        self.seconds_per_iterations = []
        self.debugger = deepkit.debugger.DebuggerManager(self)

        if deepkit.utils.in_self_execution():
            self.job_controller = JobController()

        self.debugger_controller: JobDebuggerController = JobDebuggerController(self.client)

        # runs in the client Thread
        def on_connect(connected):
            if connected:
                if deepkit.utils.in_self_execution():
                    self.client.register_controller('job/' + self.client.job_id, self.job_controller)

                self.client.register_controller('job/' + self.client.job_id + '/debugger', self.debugger_controller)

                asyncio.run_coroutine_threadsafe(self.debugger_controller.connected(), loop=self.client.loop)
            else:
                self.debugger.on_disconnect()

        self.client.connected.subscribe(on_connect)

        atexit.register(self.shutdown)
        self.client.connect()
        self.wait_for_connect()

        if deepkit.utils.in_self_execution():
            # the CLI handles output logging otherwise
            if len(deepkit.globals.last_logs.getvalue()) > 0:
                self.logs_buffer.append(deepkit.globals.last_logs.getvalue())

        if deepkit.utils.in_self_execution():
            # the CLI handles hardware monitoring otherwise
            p = psutil.Process()

            def on_hardware_metrics(dummy):
                net = psutil.net_io_counters()
                disk = psutil.disk_io_counters()
                data = struct.pack(
                    '<BHdHHffff',
                    1,
                    0,
                    time.time(),
                    # stretch to max precision of uint16
                    min(65535, int(((p.cpu_percent(interval=None) / 100) / psutil.cpu_count(False)) * 65535)),
                    # stretch to max precision of uint16
                    min(65535, int((p.memory_percent() / 100) * 65535)),
                    float(net.bytes_recv),
                    float(net.bytes_sent),
                    float(disk.write_bytes),
                    float(disk.read_bytes),
                )

                self.client.job_action_threadsafe('streamInternalFile',
                                                  ['.deepkit/hardware/main_0.hardware',
                                                   base64.b64encode(data).decode('utf8')])

            self.hardware_subscription = interval(1).subscribe(on_hardware_metrics)

    def throttle_call(self, fn: Callable, delay: int = 1000):
        last_time = self.last_throttle_call.get(fn)
        if not last_time or (time.time() - (delay / 1000)) > last_time:
            self.last_throttle_call[fn] = time.time()
            fn()

    def drain_speed_report(self):
        # only save latest value, each second
        if len(self.speed_buffer) == 0: return
        item = self.speed_buffer[-1]
        self.speed_buffer = []
        self.client.job_action_threadsafe(
            'streamInternalFile',
            ['.deepkit/speed.metric', base64.b64encode(item).decode('utf8')]
        )

    def drain_logs(self):
        if len(self.logs_buffer) == 0: return
        packed = ''
        buffer = self.logs_buffer.copy()
        self.logs_buffer = []
        for d in buffer:
            packed += d

        self.client.job_action_threadsafe('log', ['main_0', packed])

    def drain_metric_buffer(self):
        if len(self.metric_buffer) == 0:
            return
        buffer = self.metric_buffer.copy()
        self.metric_buffer = []
        try:
            packed = {}
            items = {}
            for d in buffer:
                if d['id'] not in packed:
                    packed[d['id']] = b''
                    items[d['id']] = 0

                items[d['id']] += 1
                packed[d['id']] += d['row']

            for i, v in packed.items():
                # print('channelData', items[i], len(v) / 27)

                self.client.job_action_threadsafe('channelData', [i, base64.b64encode(v).decode('utf8')])
        except Exception as e:
            print('on_metric failed', e)

    def wait_for_connect(self):
        async def wait():
            await self.client.connecting

        asyncio.run_coroutine_threadsafe(wait(), self.client.loop).result()

    def shutdown(self):
        if self.shutting_down: return
        self.shutting_down = True
        self.drain_metric_buffer()
        self.drain_speed_report()
        self.drain_logs()
        self.client.shutdown()

    def epoch(self, current: int, total: Optional[int]):
        self.iteration(current, total)
        self.debugger.tick()

    def iteration(self, current: int, total: Optional[int]):
        self.job_iteration = current
        if total:
            self.job_iterations = total

        now = time.time()
        if self.last_iteration_time:
            self.seconds_per_iterations.append({
                'diff': now - self.last_iteration_time,
                'when': now,
            })

        self.last_iteration_time = now
        self.last_batch_time = now

        # remove all older than twenty seconds
        self.seconds_per_iterations = [x for x in self.seconds_per_iterations if (now - x['when']) < 20]
        self.seconds_per_iterations = self.seconds_per_iterations[-30:]

        if len(self.seconds_per_iterations) > 0:
            diffs = [x['diff'] for x in self.seconds_per_iterations]
            self.seconds_per_iteration = sum(diffs) / len(diffs)

        if self.seconds_per_iteration:
            self.client.patch('secondsPerIteration', self.seconds_per_iteration)

        self.client.patch('iteration', self.job_iteration)
        if total:
            self.client.patch('iterations', self.job_iterations)

        iterations_left = self.job_iterations - self.job_iteration
        if iterations_left > 0:
            self.client.patch('eta', self.seconds_per_iteration * iterations_left)
        else:
            self.client.patch('eta', 0)

    def batch(self, current: int, total: int = None, size: int = None):
        self.step(current, total, size)

    def step(self, current: int, total: int = None, size: int = None):
        self.job_step = current
        if total is not None:
            self.job_steps = total

        self.client.patch('step', current)
        now = time.time()

        x = self.job_iteration + (current / total)
        speed_per_second = size / (now - self.last_batch_time) if self.last_batch_time else size

        if self.last_batch_time:
            self.seconds_per_iterations.append({
                'diff': (now - self.last_batch_time) * total,
                'when': now
            })

        # remove all older than twenty seconds
        self.seconds_per_iterations = [x for x in self.seconds_per_iterations if (now - x['when']) < 20]
        self.seconds_per_iterations = self.seconds_per_iterations[-30:]

        if len(self.seconds_per_iterations) > 0:
            diffs = [x['diff'] for x in self.seconds_per_iterations]
            self.seconds_per_iteration = sum(diffs) / len(diffs)

            iterations_left = self.job_iterations - self.job_iteration
            self.client.patch('eta', self.seconds_per_iteration * iterations_left)

        self.last_batch_time = now

        if self.seconds_per_iteration:
            self.client.patch('secondsPerIteration', self.seconds_per_iteration)

        self.client.patch('speed', speed_per_second)

        speed = struct.pack('<Bddd', 1, float(x), now, float(speed_per_second))
        self.speed_buffer.append(speed)
        self.throttle_call(self.drain_logs)

        if total:
            self.client.patch('steps', total)

        self.debugger.tick()

    def set_title(self, s: str):
        self.client.patch('title', s)

    def set_info(self, name: str, value: any):
        self.client.patch('infos.' + str(name), value)

    def set_description(self, description: any):
        self.client.patch('description', description)

    def add_label(self, label_name: str):
        self.client.job_action_threadsafe('addLabel', [label_name])

    def remove_label(self, label_name: str):
        self.client.job_action_threadsafe('removeLabel', [label_name])

    def set_config(self, name: str, value: any):
        self.client.patch('config.config.' + name, value)

    def define_metric(self, name: str, traces: List[str] = None):
        name = name.replace('.', '/')
        if not traces:
            traces = ['0']
        self.defined_metrics[name] = {'traces': traces}
        self.client.job_action_threadsafe('defineMetric', [name, self.defined_metrics[name]])

        context = self

        class Controller:
            def send(self, *y, x=None):
                context.metric(name, *y, x=x)

        return Controller()

    def add_file(self, path: str):
        self.client.job_action_threadsafe('uploadFile',
                                          [path, base64.b64encode(open(path, 'rb').read()).decode('utf8')])

    def add_file_content(self, path: str, content: bytes):
        self.client.job_action_threadsafe('uploadFile', [path, base64.b64encode(content).decode('utf8')])

    def set_list(self, name: str):
        self.client.job_action_threadsafe('setList', [name])

    def get_config(self, path, default=None):
        res = deepkit.utils.get_parameter_by_path(get_job_config(), path)
        if res is None:
            self.set_config(path, default)
            return default

        return res

    def intconfig(self, path, default=None):
        v = self.get_config(path, None)
        return int(v) if v is not None else default

    def floatconfig(self, path, default=None):
        v = self.get_config(path, None)
        return float(v) if v is not None else default

    def boolconfig(self, path, default=None):
        v = self.get_config(path, None)
        return bool(v) if v is not None else default

    def config(self, path, default=None):
        v = self.get_config(path, None)
        return v if v is not None else default

    def watch_keras_model(self, model, model_input=None, name=None, is_batch=True):
        if model in self.model_watching: return
        self.model_watching[model] = True

        from deepkit.tf import TFDebugger, extract_model_graph
        name = name if name else model.name

        graph, record_map, input_names = extract_model_graph(model)
        debugger = TFDebugger(self.debugger, model, model_input, name, record_map, is_batch, input_names)

        if not model_input:
            # we monkey patch entry methods to keras so we automatically fetch the model_input
            ori_fit_generator = model.fit_generator
            ori_fit = model.fit
            ori_train_on_batch = model.train_on_batch
            ori_predict = model.predict

            def get_x(x):
                # resize batches to size 1 if is_batch=True
                if len(input_names) == 1:
                    return np.array([x[0]] if is_batch else x)
                else:
                    return [np.array([v[0]]) if is_batch else v for v in x]

            def fit_generator(generator, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.model_input = get_x(next(generator))
                return ori_fit_generator(generator, *args, **kwargs)

            model.fit_generator = fit_generator

            def fit(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.model_input = get_x(x)
                return ori_fit(x, *args, **kwargs)

            model.fit = fit

            def train_on_batch(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    # we want only one item of the batch
                    if len(input_names) == 1:
                        debugger.model_input = np.array([x[0]])
                    else:
                        debugger.model_input = [np.array([v[0]]) for v in x]
                return ori_train_on_batch(x, *args, **kwargs)

            model.train_on_batch = train_on_batch

            def predict(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.model_input = get_x(x)
                return ori_predict(x, *args, **kwargs)

            model.predict = predict

        self.debugger.register_debugger(debugger)
        self.client.job_action_threadsafe('setModelGraph', [graph, name])
        return debugger

    def watch_torch_model(self, model, name='main'):
        if model in self.model_watching: return
        self.model_watching[model] = True
        from deepkit.pytorch import TorchDebugger

        def resolve_map(inputs):
            graph, record_map, input_names, output_names = self._set_torch_model(model, name=name, inputs=inputs)
            return record_map, input_names, output_names

        debugger = TorchDebugger(self.debugger, model, name, resolve_map)
        self.debugger.register_debugger(debugger)
        return debugger

    def _set_torch_model(self, model, input_shape=None, input_sample=None, inputs=None, name='main'):
        """
        Extracts the computation graph using either the given input_shape with random data
        or the given (real) input_sample. If you have multiple models per training, use the name
        argument to differentiate.
        :param model: your pytorch model instance
        :param input_shape: shape like (1, 32, 32) or a list of input shapes for multi input ((1, 3, 64, 64), (10,)).
                            Don't forget to specify the batch dimension.
        :param input_sample: a simple (not in a batch)
        :param inputs: full inputs list with real examples as if `model(*inputs)` is  called
        :param name: optional name if you have multiple models
        :return:
        """
        from torch import from_numpy
        from deepkit.pytorch import get_pytorch_graph

        if not inputs and not input_shape and input_sample is None:
            raise Exception('No inputs, input_shape and no input_sample given. Specify either of those.')
        xs = inputs

        if xs is None:
            if input_sample is not None:
                if isinstance(input_sample, (tuple, list)):
                    xs = input_sample
                else:
                    # we got single input sample
                    xs = [input_sample]

            if xs is None:
                # we need a batch size of 1
                if isinstance(input_shape[0], (tuple, list)):
                    # we got multi inputs
                    xs = []
                    for i in range(0, len(input_shape)):
                        # convert to float32 per default
                        x = (from_numpy(np.random.random_sample(input_shape[i]).astype(np.single)),)
                        xs.append(x)
                else:
                    # convert to float32 per default
                    xs = (from_numpy(np.random.random_sample(input_shape).astype(np.single)),)

        graph, record_map, input_names, output_names = get_pytorch_graph(model, xs)

        self.client.job_action_threadsafe('setModelGraph', [graph, name])
        return graph, record_map, input_names, output_names

    def metric(self, name: str, *y, x=None):
        if y is None:
            y = 0

        if not isinstance(y, (list, tuple)):
            y = [y]

        y = [float(v) for v in y]

        if x is None:
            if self.job_steps > 0:
                x = self.job_iteration + (self.job_step / self.job_steps)
            else:
                if name not in self.auto_x_of_metrix:
                    self.auto_x_of_metrix[name] = 0
                self.auto_x_of_metrix[name] += 1
                x = self.auto_x_of_metrix[name]

        name = name.replace('.', '/')

        if name not in self.defined_metrics:
            traces = [str(i) for i, _ in enumerate(y)]
            self.define_metric(name, traces=traces)
        else:
            if 'traces' in self.defined_metrics[name] and len(self.defined_metrics[name]['traces']) != len(y):
                traces = self.defined_metrics[name]['traces']
                raise Exception(f'Metric {name} has {len(traces)} traces defined, but you provided {len(y)}')

        row_binary = struct.pack('<BHdd', 1, len(y), float(x), time.time())
        for y1 in y:
            row_binary += struct.pack('<d', float(y1) if y1 is not None else 0.0)

        self.client.patch('channelLastValues.' + name, y)
        self.metric_buffer.append({'id': name, 'row': row_binary})
        self.throttle_call(self.drain_metric_buffer)

    def create_keras_callback(self, debug_x=None):
        from .deepkit_keras import KerasCallback
        callback = KerasCallback(debug_x)

        return callback

    def log(self, s: str):
        self.logs_buffer.append(s)
        self.throttle_call(self.drain_logs)
