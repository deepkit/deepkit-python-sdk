from __future__ import annotations

import asyncio
import atexit
import time
from threading import Lock
from typing import Optional, List

from rx import interval
from rx.operators import buffer
from rx.subject import Subject

import deepkit.client
import deepkit.globals


class Context:
    def __init__(self, config_path: str):
        deepkit.globals.last_context = self
        self.config_path = config_path
        self.log_lock = Lock()
        self.defined_metrics = {}
        self.log_subject = Subject()
        self.metric_subject = Subject()

        self.client = deepkit.client.Client(config_path)
        atexit.register(self.shutdown)

        def on_log(data: List):
            packed = ''
            for d in data:
                packed += d

            self.client.job_action('log', ['main_0', packed])

        self.log_subject.pipe(buffer(interval(1))).subscribe(on_log)

        if len(deepkit.globals.last_logs.getvalue()) > 0:
            self.log_subject.on_next(deepkit.globals.last_logs.getvalue())

        def on_metric(data: List):
            packed = {}

            for d in data:
                if d['id'] not in packed:
                    packed[d['id']] = []

                packed[d['id']].append(d['row'])

            for i, v in packed.items():
                self.client.job_action('channelData', [i, v])

        self.metric_subject.pipe(buffer(interval(1))).subscribe(on_metric)

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
        self.client.patch('iteration', current)
        if total:
            self.client.patch('iterations', total)

    def step(self, current: int, total: int = None, size: int = None):
        self.client.patch('step', current)
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

    def metric(self, name: str, x, y):
        if name not in self.defined_metrics:
            self.define_metric(name)

        if not isinstance(y, list):
            y = [y]

        self.metric_subject.on_next({'id': name, 'row': [x, time.time()] + y})
        self.client.patch('channels.' + name + '.lastValue', y)

    def log(self, s):
        self.log_subject.on_next(s)
