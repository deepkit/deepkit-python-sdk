import asyncio
import atexit
import base64
import json
import os
import signal
import struct
import sys
import time
from datetime import datetime
from threading import Lock
from typing import Optional, Callable, NamedTuple, Dict, List

import PIL.Image
import numpy as np
import psutil
import typedload
from rx import interval

import deepkit.client
import deepkit.debugger
import deepkit.globals
import deepkit.utils
from deepkit.model import ExperimentOptions
from deepkit.utils.image import pil_image_to_jpeg, get_layer_vis_square, get_image_tales, make_image_from_dense
from deepkit.utils import numpy_to_binary


def get_job_config():
    if deepkit.globals.loaded_job_config is None:
        if 'DEEPKIT_JOB_CONFIG' in os.environ:
            deepkit.globals.loaded_job_config = json.loads(os.environ['DEEPKIT_JOB_CONFIG'])
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


class Experiment:
    def __init__(self, options: ExperimentOptions = None):
        if options is None:
            options = ExperimentOptions()

        self.metric_buffer = []
        self.speed_buffer = []
        self.logs_buffer = []
        self.last_throttle_call = dict()

        self.client = deepkit.client.Client(options)
        deepkit.globals.last_experiment = self
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
        self.auto_x_of_insight = dict()
        self.created_insights = dict()

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
        if current and self.job_iteration == current:
            # nothing to do
            return

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

    def batch(self, current: int, total: int = None, size: int = 1):
        self.step(current, total, size)

    def step(self, current: int, total: int = None, size: int = 1):
        if current and self.job_steps == current:
            # nothing to do
            return

        self.job_step = current
        if total is not None:
            self.job_steps = total
        if total is None:
            total = self.job_steps

        self.client.patch('step', current)
        now = time.time()

        x = self.job_iteration + (current / total)
        speed_per_second = 0
        if size:
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
        self.drain_logs()

        if total:
            self.client.patch('steps', total)

        self.debugger.tick()

    def set_title(self, s: str):
        self.client.patch('title', s)

    def set_info(self, name: str, value: any):
        self.client.patch('infos.' + str(name.replace('.', '/')), value)

    def set_description(self, description: any):
        self.client.patch('description', description)

    def add_label(self, *label_name: str):
        for name in label_name:
            self.client.job_action_threadsafe('addLabel', [name])

    def remove_label(self, label_name: str):
        self.client.job_action_threadsafe('removeLabel', [label_name])

    def set_config(self, name: str, value: any):
        self.client.patch('config.config.' + name.replace('.', '/'), value)

    def define_metric(self, name: str, traces: List[str] = None):
        name = name.replace('.', '/')
        if not traces:
            traces = ['0']
        self.defined_metrics[name] = {'traces': traces}
        self.client.job_action_threadsafe('defineMetric', [name, self.defined_metrics[name]])

        that = self

        class Controller:
            def send(self, *y, x=None):
                that.log_metric(name, *y, x=x)

        return Controller()

    def add_output_file(self, path: str):
        self.add_file(path, as_output=True)

    def add_output_content(self, path: str, content):
        self.add_file_content(path, content, as_output=True)

    def add_file(self, path: str, as_output=False):
        relative_path = os.path.relpath(path, os.getcwd())
        if 'DEEPKIT_ROOT_DIR' in os.environ:
            relative_path = os.path.relpath(path, os.environ['DEEPKIT_ROOT_DIR'])

        if '..' in relative_path:
            relative_path = '__parent/' + relative_path.replace('..', '__')

        self.add_file_content(relative_path, open(path, 'rb').read(), as_output=as_output)

    def add_file_content(self, path: str, content, as_output=False):
        if isinstance(content, (dict, list, tuple)):
            content = json.dumps(content)

        if isinstance(content, str):
            content = bytes(content, encoding='utf-8')

        if not isinstance(content, bytes):
            raise Exception('Data type is not supported. Please provide str, bytes, or dict/list/tuple.')

        method = 'uploadOutputFile' if as_output else 'uploadFile'
        self.client.job_action_threadsafe(method, [path, base64.b64encode(content).decode('utf8')])

    def set_list(self, name: str):
        self.client.job_action_threadsafe('setList', [name])

    def full_config(self):
        return get_job_config()

    def get_config(self, path, default=None):
        res = deepkit.utils.get_parameter_by_path(get_job_config(), path)
        if res is None:
            self.set_config(path, default)
            return default

        return res

    def intconfig(self, path, default=None):
        v = self.get_config(path, default)
        return int(v) if v is not None else default

    def floatconfig(self, path, default=None):
        v = self.get_config(path, default)
        return float(v) if v is not None else default

    def boolconfig(self, path, default=None):
        v = self.get_config(path, default)
        if v is None:
            return default
        if not v or v is 'false' or v is 0 or v is '0':
            return False
        return True

    def config(self, path, default=None):
        v = self.get_config(path, default)
        return v if v is not None else default

    def watch_keras_model(self, model, model_input=None, name=None, is_batch=True):
        if model in self.model_watching: return
        self.model_watching[model] = True

        from deepkit.keras_tf import TFDebugger, extract_model_graph
        name = name if name else model.name

        graph, record_map, input_names = extract_model_graph(model)
        debugger = TFDebugger(self.debugger, model, model_input, name, record_map, is_batch, input_names)

        if not model_input:
            # we monkey patch entry methods to keras so we automatically fetch the model_input
            ori_fit_generator = model.fit_generator
            ori_fit = model.fit
            ori_train_on_batch = model.train_on_batch
            ori_predict = model.predict

            def fit_generator(generator, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.set_input(next(iter(generator)))
                return ori_fit_generator(generator, *args, **kwargs)

            model.fit_generator = fit_generator

            def fit(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.set_input(x)
                return ori_fit(x, *args, **kwargs)

            model.fit = fit

            def train_on_batch(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.set_input(x)
                return ori_train_on_batch(x, *args, **kwargs)

            model.train_on_batch = train_on_batch

            def predict(x=None, *args, **kwargs):
                if debugger.model_input is None:
                    debugger.set_input(x)
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

    def log_insight(self, *data, name: str, x=None, image_convertion=True, meta=None):
        if x is None:
            if self.job_steps > 0:
                x = self.job_iteration + (self.job_step / self.job_steps)
            elif self.job_iteration > 0:
                x = self.job_iteration
            else:
                if name not in self.auto_x_of_insight:
                    self.auto_x_of_insight[name] = 0
                self.auto_x_of_insight[name] += 1
                x = self.auto_x_of_insight[name]

        if not isinstance(x, (int, float)):
            raise Exception('x needs to be integer or float')

        if x not in self.created_insights:
            self.created_insights[x] = True
            self.client.job_action_threadsafe('addInsight', [
                x,
                time.time(),
                self.job_iteration,
                self.job_step,
            ])

        for i, d in enumerate(data):
            file_type = ''
            if isinstance(d, PIL.Image.Image):
                file_type = 'png'
                d = pil_image_to_jpeg(d)
            elif isinstance(d, np.ndarray):
                # tf is not batch per default

                if image_convertion:
                    sample = np.copy(d)
                    shape = d.shape
                    image = False
                    if len(shape) == 3:
                        try:
                            if 'keras' in sys.modules:
                                import keras
                                if keras.backend.image_data_format() == 'channels_last':
                                    sample = np.transpose(sample, (2, 0, 1))
                            elif 'tensorflow.keras' in sys.modules:
                                import tensorflow.keras as keras
                                if keras.backend.image_data_format() == 'channels_last':
                                    sample = np.transpose(sample, (2, 0, 1))
                        except:
                            pass

                        if sample.shape[0] == 3:
                            d = PIL.Image.fromarray(get_layer_vis_square(sample))
                            image = True
                        else:
                            d = PIL.Image.fromarray(get_image_tales(sample))
                            image = True
                    elif len(shape) > 1:
                        d = PIL.Image.fromarray(get_layer_vis_square(sample))
                        image = True
                    elif len(shape) == 1:
                        if shape[0] != 1:
                            # we got a single number
                            d = sample[0]
                        else:
                            d = make_image_from_dense(sample)
                            image = True
                    if image:
                        file_type = 'png'
                        d = pil_image_to_jpeg(d)
                    else:
                        file_type = 'npy'
                        d = numpy_to_binary(d)
                else:
                    file_type = 'npy'
                    d = numpy_to_binary(d)
            else:
                file_type = 'json'
                d = bytes(json.dumps(d), encoding='utf-8')

            if len(data) > 1:
                file_name = name + '_' + str(i) + '.' + file_type
            else:
                file_name = name + '.' + file_type

            self.client.job_action_threadsafe('addInsightEntry', [
                x,
                file_name,
                datetime.now().isoformat(),
                {
                    'type': file_type,
                    'meta': meta
                },
                base64.b64encode(d).decode(),
            ])

    def log_metric(self, name: str, *y, x=None):
        if y is None:
            y = 0

        if not isinstance(y, (list, tuple)):
            y = [y]

        y = [float(v) if v is not None else 0 for v in y]

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
        self.drain_metric_buffer()

    def create_keras_callback(self, model=None, debug_model_input=None):
        from .deepkit_keras import KerasCallback
        callback = KerasCallback(debug_model_input)
        if model:
            self.watch_keras_model(model)

        return callback

    def log(self, s: str):
        self.logs_buffer.append(s)
        self.drain_logs()
