import asyncio
import atexit
import base64
import os
import signal
import struct
import time
from threading import Lock
from typing import Optional, Callable, NamedTuple, Dict

import psutil
import typedload
from rx import interval
from rx.subject import Subject

import deepkit.client
import deepkit.globals
import deepkit.utils
from deepkit.model import ContextOptions


def pytorch_graph():
    # see https://discuss.pytorch.org/t/print-autograd-graph/692/18
    # https://github.com/szagoruyko/pytorchviz
    pass


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
        self.state: JobDebuggingState
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
        self.seconds_per_iteration = 0
        self.seconds_per_iterations = []
        self.debugger_controller = None

        if deepkit.utils.in_self_execution():
            self.job_controller = JobController()

        self.debugger_controller = JobDebuggerController(self.client)

        # runs in the client Thread
        def on_connect(connected):
            if connected:
                if deepkit.utils.in_self_execution():
                    self.client.register_controller('job/' + self.client.job_id, self.job_controller)

                self.client.register_controller('job/' + self.client.job_id + '/debugger', self.debugger_controller)

                asyncio.run_coroutine_threadsafe(self.debugger_controller.connected(), loop=self.client.loop)

        self.client.connected.subscribe(on_connect)

        atexit.register(self.shutdown)
        self.client.connect()
        self.wait_for_connect()

        if deepkit.utils.in_self_execution:
            # the CLI handles output logging
            if len(deepkit.globals.last_logs.getvalue()) > 0:
                self.logs_buffer.append(deepkit.globals.last_logs.getvalue())

        if deepkit.utils.in_self_execution:
            # the CLI handles hardware monitoring
            p = psutil.Process()

            def on_hardware_metrics(dummy):
                net = psutil.net_io_counters()
                disk = psutil.disk_io_counters()
                data = struct.pack(
                    '<BHdHHffff',
                    1,
                    0,
                    time.time(),
                    int(((p.cpu_percent(interval=None) / 100) / psutil.cpu_count()) * 65535),
                    # stretch to max precision of uint16
                    int((p.memory_percent() / 100) * 65535),  # stretch to max precision of uint16
                    float(net.bytes_recv),
                    float(net.bytes_sent),
                    float(disk.write_bytes),
                    float(disk.read_bytes),
                )

                self.client.job_action_threadsafe('streamFile',
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
            'streamFile',
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

    def step(self, current: int, total: int = None, size: int = None):
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

    def set_parameter(self, name: str, value: any):
        self.client.patch('config.parameters.' + name, value)

    def define_metric(self, name: str, options: dict):
        self.defined_metrics[name] = {}
        self.client.job_action_threadsafe('defineMetric', [name, options])


    # def debug_snapshot(self, graph: dict):
    #     self.client.job_action_threadsafe('debugSnapshot', [graph])

    def add_file(self, path: str):
        self.client.job_action_threadsafe('uploadFile',
                                          [path, base64.b64encode(open(path, 'rb').read()).decode('utf8')])

    def add_file_content(self, path: str, content: bytes):
        self.client.job_action_threadsafe('uploadFile', [path, base64.b64encode(content).decode('utf8')])

    def set_model_graph(self, graph: dict):
        self.client.job_action_threadsafe('setModelGraph', [graph])

    def metric(self, name: str, x, y):
        if name not in self.defined_metrics:
            self.define_metric(name, {})

        if not isinstance(y, list):
            y = [y]

        row_binary = struct.pack('<BHdd', 1, len(y), float(x), time.time())
        for y1 in y:
            row_binary += struct.pack('<d', float(y1) if y1 is not None else 0.0)

        self.client.patch('channelLastValues.' + name, y)
        self.metric_buffer.append({'id': name, 'row': row_binary})
        self.throttle_call(self.drain_metric_buffer)

    def log(self, s: str):
        self.logs_buffer.append(s)
        self.throttle_call(self.drain_logs);
