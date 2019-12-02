from __future__ import annotations

import json
import os
import sys

import six

import deepkit.globals
from deepkit.client import Client
from deepkit.context import Context
from deepkit.utils import get_parameter_by_path


def log(s):
    if deepkit.globals.last_context:
        deepkit.globals.last_context.log(s)
    else:
        deepkit.globals.last_logs.write(s)


def context(config_path: str = None) -> Context:
    if deepkit.globals.last_context:
        return deepkit.globals.last_context

    context = Context(config_path)

    return context


if 'DEEPKIT_JOB_ACCESSTOKEN' not in os.environ:
    class StdHook:
        def __init__(self, s):
            self.s = s

        def fileno(self):
            return self.s.fileno()

        def isatty(self):
            return self.s.isatty()

        def flush(self):
            self.s.flush()

        def write(self, s):
            self.s.write(s)
            log(s)


    sys.stdout = StdHook(sys.__stdout__)
    sys.stderr = StdHook(sys.__stderr__)


def batch(current, total=None, size=None):
    context().step(current, total, size)


def step(current, total=None):
    context().step(current, total)


def get_job():
    if deepkit.globals.loaded_job is None:
        cwd = os.getcwd()
        job_file = os.path.join(cwd, '.deepkit', 'job.json')
        if os.path.exists(job_file):
            with open(job_file) as file:
                deepkit.globals.loaded_job = json.load(file)

        else:
            deepkit.globals.loaded_job = {
                'config': {
                    'parameters': {}
                }
            }

    return deepkit.globals.loaded_job


def parameter(path, value):
    context().set_parameter(path, value)


def get_parameter(path, default=None):
    res = get_parameter_by_path(get_job()['config']['parameters'], path)
    if res is None:
        parameter(path, default)
        return default

    return res


def intparam(path, default=None):
    v = get_parameter(path, None)
    return int(v) if v is not None else default


def floatparam(path, default=None):
    v = get_parameter(path, None)
    return float(v) if v is not None else default


def param(path, default=None):
    v = get_parameter(path, None)
    return v if v is not None else default


def epoch(epoch, total=None):
    context().epoch(epoch, total)


def set_title(s):
    context().set_title(s)


def set_info(name, value):
    context().set_info(name, value)


def set_parameter(name, value):
    context().set_parameter(name, value)


class JobMetric:
    """
    :type job_backend: JobBackend
    """

    def __init__(self, name, traces=None,
                 main=False, xaxis=None, yaxis=None, layout=None):
        """
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this metric is visible in the job view per default.
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        self.name = name

        if not (isinstance(traces, list) or traces is None):
            raise Exception(
                'traces can only be None or a list of strings: [name1, name2]')

        if not traces:
            traces = [name]

        options = {
            'traces': traces,
            'main': main,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
        }

        self.traces = traces
        context().define_metric(name, options)

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception(
                'You tried to set more y values (%d items) then traces available in metric %s (%d traces).' % (
                    len(y), self.name, len(self.traces)))

        for v in y:
            if not isinstance(v, (int, float)) and v is not None and not isinstance(v, six.string_types):
                raise Exception('Could not send metric value for ' + self.name + ' since type ' + type(
                    y).__name__ + ' is not supported. Use int, float or string values.')

        context().metric(self.name, x, y)


class JobLossMetric:
    """
    :type job_backend : JobBackend
    """

    def __init__(self, name, xaxis=None, yaxis=None, layout=None):
        self.name = name
        options = {
            'traces': ['training', 'validation'],
            'main': True,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
            'lossChannel': True
        }

        context().define_metric(name, options)

    def send(self, x, training, validation):
        context().metric(self.name, x, [training, validation])


def create_loss_metric(name='loss', xaxis=None, yaxis=None, layout=None):
    """
    :param name: string
    :return: JobLossGraph
    """

    return JobLossMetric(name, xaxis, yaxis, layout)


def create_metric(name, traces=None,
                  main=False, xaxis=None, yaxis=None, layout=None):
    """
    :param name: str
    :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this metric is visible in the job view per default.
    :param xaxis: dict
    :param yaxis: dict
    :param layout: dict
    """
    return JobMetric(name, traces, main, xaxis, yaxis, layout)


def create_keras_callback(model,
                          insights=False, insights_x=None,
                          additional_insights_layer=[]):
    """
    :type validation_data: int|None: (x, y) or generator
    :type validation_data_size: int|None: Defines the size of validation_data, if validation_data is a generator
    """

    if insights and (insights_x is None or insights_x is False):
        raise Exception('Can not build Keras callback with active insights but with invalid `insights_x` as input.')

    from .deepkit_keras import KerasCallback
    callback = KerasCallback(model)
    callback.insights_x = insights_x
    callback.insight_layer = additional_insights_layer

    return callback
