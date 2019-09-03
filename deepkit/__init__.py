import simplejson as simplejson
import six
import sys
import time
import os
import json
from collections import OrderedDict
from deepkit.utils import get_parameter_by_path

start_time = time.time()

loaded_job = None


def stdout_api_call(command, **kwargs):
    action = OrderedDict()
    action['deepkit'] = command;
    action.update(kwargs)
    sys.stdout.flush()
    sys.stdout.write(simplejson.dumps(action) + '\n')
    sys.stdout.flush()


def batch(current, total=None, size=None):
    stdout_api_call('batch', current=current, total=total, size=size)


def step(current, total=None):
    stdout_api_call('batch', current=current, total=total)


def get_job():
    global loaded_job

    if loaded_job is None:
        cwd = os.getcwd()
        job_file = os.path.join(cwd, '.deepkit', 'job.json')
        if os.path.exists(job_file):
            with open(job_file) as file:
                loaded_job = json.load(file)

        else:
            loaded_job = {
                'config': {
                    'parameters': {}
                }
            }

    return loaded_job


def parameter(path, value):
    stdout_api_call('parameter', path=path, value=value)


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


def get_run_time(precision=3):
    global start_time
    return round(time.time() - start_time, precision)


def epoch(epoch, total=None):
    stdout_api_call('epoch', epoch=epoch, total=total)


def set_status(status):
    stdout_api_call('status', status=status)


def set_info(name, value):
    stdout_api_call('info', name=name, value=value)


class JobChannel:
    NUMBER = 'number'
    TEXT = 'text'

    """
    :type job_backend: JobBackend
    """

    def __init__(self, name, traces=None,
                 main=False, kpi=False, kpiTrace=0, max_optimization=True,
                 type=None, xaxis=None, yaxis=None, layout=None):
        """
        :param name: str
        :param traces: None|list : per default create a trace based on "name".
        :param main: bool : whether this channel is visible in the job list as column for better comparison.
        :param kpi: bool : whether this channel is the KPI (key performance indicator).
                           Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.
        :param kpiTrace: bool : if you have multiple traces, define which is the KPI. 0 based index.
        :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi . Use max_optimization=False to
                                        tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.
        :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT, JobChannel.IMAGE
        :param xaxis: dict
        :param yaxis: dict
        :param layout: dict
        """
        self.name = name
        self.kpi = kpi
        self.kpiTrace = kpiTrace

        if not (isinstance(traces, list) or traces is None):
            raise Exception(
                'traces can only be None or a list of strings: [name1, name2]')

        if not traces:
            traces = [name]

        message = {
            'name': name,
            'traces': traces,
            'main': main,
            'kpi': kpi,
            'kpiTrace': kpiTrace,
            'maxOptimization': max_optimization,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
        }

        self.traces = traces
        stdout_api_call('create-channel', **message)

    def send(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if len(y) != len(self.traces):
            raise Exception(
                'You tried to set more y values (%d items) then traces available in channel %s (%d traces).' % (
                    len(y), self.name, len(self.traces)))

        for v in y:
            if not isinstance(v, (int, float)) and v is not None and not isinstance(v, six.string_types):
                raise Exception('Could not send channel value for ' + self.name + ' since type ' + type(
                    y).__name__ + ' is not supported. Use int, float or string values.')

        stdout_api_call('channel', **{
            'name': self.name,
            'time': get_run_time(),
            'x': x,
            'y': y
        })


class JobLossChannel:
    """
    :type job_backend : JobBackend
    """

    def __init__(self, name, xaxis=None, yaxis=None, layout=None):
        self.name = name
        message = {
            'name': self.name,
            'traces': ['training', 'validation'],
            'main': True,
            'xaxis': xaxis,
            'yaxis': yaxis,
            'layout': layout,
            'lossChannel': True
        }

        stdout_api_call('create-channel', **message)

    def send(self, x, training, validation):
        stdout_api_call('channel', **{
            'name': self.name,
            'time': get_run_time(),
            'x': x,
            'y': [training, validation]
        })


def create_loss_channel(name='loss', xaxis=None, yaxis=None, layout=None):
    """
    :param name: string
    :return: JobLossGraph
    """

    return JobLossChannel(name, xaxis, yaxis, layout)


def create_channel(name, traces=None,
                   main=False, kpi=False, kpiTrace=0, max_optimization=True,
                   type=JobChannel.NUMBER,
                   xaxis=None, yaxis=None, layout=None):
    """
    :param name: str
    :param traces: None|list : per default create a trace based on "name".
    :param main: bool : whether this channel is visible in the job list as column for better comparison.
    :param kpi: bool : whether this channel is the KPI (key performance indicator).
                       Used for hyperparameter optimization. Only one channel can be a kpi. Only first trace used.
    :param kpiTrace: bool : if you have multiple traces, define which is the KPI. 0 based index.
    :param max_optimization: bool : whether the optimization maximizes or minmizes the kpi. Use max_optimization=False to
                                    tell the optimization algorithm that his channel minimizes a kpi, for instance the loss of a model.
    :param type: str : One of JobChannel.NUMBER, JobChannel.TEXT
    :param xaxis: dict
    :param yaxis: dict
    :param layout: dict
    """
    return JobChannel(name, traces, main, kpi, kpiTrace, max_optimization, type, xaxis, yaxis, layout)


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
