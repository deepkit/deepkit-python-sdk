import asyncio
import atexit
import base64
import json
import os
import signal
import time
from threading import Lock
from typing import Optional, List

import psutil
from rx import interval
from rx.operators import buffer
from rx.subject import Subject

import deepkit.client
import deepkit.globals
import deepkit.utils


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


class JobDebuggerController:
    def __init__(self):
        self.watching_layers = {}

    def debugStopWatchLayer(self, id: str):
        if id in self.watching_layers:
            del self.watching_layers[id]

    def debugStartWatchLayer(self, id: str):
        self.watching_layers[id] = True


class Context:
    def __init__(self, config_path: str):
        deepkit.globals.last_context = self
        self.log_lock = Lock()
        self.defined_metrics = {}
        self.log_subject = Subject()
        self.metric_subject = Subject()
        self.speed_report_subject = Subject()

        self.client = deepkit.client.Client(config_path)
        self.wait_for_connect()
        atexit.register(self.shutdown)

        self.last_iteration_time = 0
        self.last_batch_time = 0
        self.job_iteration = 0
        self.job_iterations = 0
        self.seconds_per_iteration = 0
        self.seconds_per_iterations = []
        self.debugger_controller = None

        def on_connect(connected):
            if connected:
                if deepkit.utils.in_self_execution():
                    self.job_controller = JobController()
                    asyncio.run_coroutine_threadsafe(
                        self.client.register_controller('job/' + self.client.job_id, self.job_controller),
                        self.client.loop
                    )

                self.debugger_controller = JobDebuggerController()
                asyncio.run_coroutine_threadsafe(
                    self.client.register_controller('job/' + self.client.job_id + '/debugger',
                                                    self.debugger_controller),
                    self.client.loop
                )

        self.client.connected.subscribe(on_connect)

        def on_log(data: List):
            if len(data) == 0: return
            packed = ''
            for d in data:
                packed += d

            self.client.job_action('log', ['main_0', packed])

        self.log_subject.pipe(buffer(interval(1))).subscribe(on_log)

        if len(deepkit.globals.last_logs.getvalue()) > 0:
            self.log_subject.on_next(deepkit.globals.last_logs.getvalue())

        def on_metric(data: List):
            if len(data) == 0: return
            packed = {}

            for d in data:
                if d['id'] not in packed:
                    packed[d['id']] = []

                packed[d['id']].append(d['row'])

            for i, v in packed.items():
                self.client.job_action('channelData', [i, v])

        self.metric_subject.pipe(buffer(interval(1))).subscribe(on_metric)

        def on_speed_report(rows):
            # only save latest value, each second
            if len(rows) == 0: return
            self.client.job_action('streamJsonFile', ['.deepkit/speed.csv', [rows[-1]]])

        self.speed_report_subject.pipe(buffer(interval(1))).subscribe(on_speed_report)

        p = psutil.Process()
        self.client.job_action(
            'streamJsonFile',
            [
                '.deepkit/hardware/main_0.csv',
                [
                    [
                        'time', 'cpu', 'memory', 'network_rx', 'network_tx',
                        'block_write',
                        'block_read'
                    ]
                ]
            ]
        )

        def on_hardware_metrics(dummy):
            net = psutil.net_io_counters()
            disk = psutil.disk_io_counters()
            data = [
                time.time(),
                (p.cpu_percent(interval=None) / 100) / psutil.cpu_count(),
                p.memory_percent() / 100,
                net.bytes_recv,
                net.bytes_sent,
                disk.write_bytes,
                disk.read_bytes,
            ]
            self.client.job_action('streamJsonFile', ['.deepkit/hardware/main_0.csv', [data]])

        interval(1).subscribe(on_hardware_metrics)

    def wait_for_connect(self):
        async def wait():
            await self.client.connecting

        asyncio.run_coroutine_threadsafe(wait(), self.client.loop).result()

    def shutdown(self):
        self.metric_subject.on_completed()
        self.log_subject.on_completed()

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
        self.speed_report_subject.on_next([x, now, speed_per_second])

        if total:
            self.client.patch('steps', total)

    def set_title(self, s: str):
        self.client.patch('title', s)

    def set_info(self, name: str, value: any):
        self.client.patch('infos.' + name, value)

    def set_parameter(self, name: str, value: any):
        self.client.patch('config.parameters.' + name, value)

    def define_metric(self, name: str, options: dict):
        self.defined_metrics[name] = {}
        self.client.job_action('defineMetric', [name, options])

    def debug_snapshot(self, graph: dict):
        self.client.job_action('debugSnapshot', [graph])

    def add_file(self, path: str):
        self.client.job_action('uploadFile', [path, base64.b64encode(open(path, 'rb').read()).decode('utf8')])

    def add_file_content(self, path: str, content: bytes):
        self.client.job_action('uploadFile', [path, base64.b64encode(content).decode('utf8')])

    def set_model_graph(self, graph: dict):
        self.add_file_content('.deepkit/graph.json', bytes(json.dumps(graph), 'utf8'))
        self.client.patch('hasModelGraph', True)

    def metric(self, name: str, x, y):
        if name not in self.defined_metrics:
            self.define_metric(name)

        if not isinstance(y, list):
            y = [y]

        self.metric_subject.on_next({'id': name, 'row': [x, time.time()] + y})
        self.client.patch('channels.' + name + '.lastValue', y)

    def log(self, s):
        self.log_subject.on_next(s)
