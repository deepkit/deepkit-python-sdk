# -*- coding: utf-8 -*-
from __future__ import division

import math
import os
import sys
import time
import numpy as np

if 'keras' in sys.modules:
    import keras
else:
    import tensorflow.keras as keras

import deepkit


def is_generator(obj):
    import inspect

    return obj is not None and (
            inspect.isgeneratorfunction(obj)
            or inspect.isgenerator(obj) or hasattr(obj, 'next') or hasattr(obj, '__next__'))


def ensure_dir(d):
    if not os.path.isdir(d):
        if os.path.isfile(d):  # but a file, so delete it
            print("Deleted", d, "because it was a file, but needs to be an directory.")
            os.remove(d)

        os.makedirs(d)


def get_total_params(model):
    total_params = 0

    flattened_layers = model.flattened_layers if hasattr(model, 'flattened_layers') else model.layers

    for i in range(len(flattened_layers)):
        total_params += flattened_layers[i].count_params()

    return total_params


class KerasCallback(keras.callbacks.Callback):
    def __init__(self, debug_model_input=None):
        super(KerasCallback, self).__init__()

        self.experiment = deepkit.experiment()

        self.debug_model_input = debug_model_input

        self.data_validation = None
        self.data_validation_size = None

        self.current = {}
        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.accuracy_metric = None
        self.all_losses = None
        self.loss_metric = None
        self.learning_rate_metric = None

    def set_model(self, model):
        super().set_model(model)
        self.experiment.watch_keras_model(model, self.debug_model_input)

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.last_batch_time = time.time()

        self.experiment.set_info('parameters', get_total_params(self.model))
        self.experiment.set_info('keras.image_data_format', keras.backend.image_data_format())
        self.experiment.set_info('keras.backend', keras.backend.backend())

        # self.job_backend.upload_keras_graph(self.model)

        if self.model.optimizer and hasattr(self.model.optimizer, 'get_config'):
            config = self.model.optimizer.get_config()
            self.experiment.set_info('optimizer', str(type(self.model.optimizer).__name__))
            for i, v in config.items():
                self.experiment.set_info('optimizer.' + str(i), v)

        # compatibility with keras 1.x
        if 'epochs' not in self.params and 'nb_epoch' in self.params:
            self.params['epochs'] = self.params['nb_epoch']
        if 'samples' not in self.params and 'nb_sample' in self.params:
            self.params['samples'] = self.params['nb_sample']

        traces = ['training', 'validation']
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            traces = []
            for output in self.model.output_layers:
                traces.append('train_' + output.name)
                traces.append('val_' + output.name)

        self.accuracy_metric = self.experiment.define_metric('accuracy', traces=traces)
        self.loss_metric = self.experiment.define_metric('loss', traces=['train', 'val'])
        self.learning_rate_metric = self.experiment.define_metric('learning rate', traces=['start', 'end'])

        self.experiment.epoch(1, self.params['epochs'])
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            loss_traces = []
            for output in self.model.output_layers:
                loss_traces.append('train_' + output.name)
                loss_traces.append('val_' + output.name)

            self.all_losses = self.experiment.define_metric('loss_all', traces=loss_traces)

        # if self.force_insights or self.job_model.insights_enabled:
        #     images = self.build_insight_images()
        #     self.job_backend.job_add_insight(0, images, None)

    def on_batch_begin(self, batch, logs={}):
        if 'nb_batches' not in self.current:
            batch_size = logs['size']
            if 'samples' in self.params:
                nb_batches = math.ceil(self.params['samples'] / batch_size)  # normal nb batches
            else:
                nb_batches = self.params['steps']

            self.current['nb_batches'] = nb_batches
            self.current['batch_size'] = batch_size
            self.experiment.set_info('Batch size', batch_size)

    def on_batch_end(self, batch, logs={}):
        self.filter_invalid_json_values(logs)
        self.experiment.batch(batch + 1, self.current['nb_batches'], logs['size'])

    def on_epoch_begin(self, epoch, logs={}):
        self.experiment.epoch(epoch + 1, self.params['epochs'])
        self.learning_rate_start = self.get_learning_rate()

    def on_epoch_end(self, epoch, logs={}):
        log = logs.copy()

        self.filter_invalid_json_values(log)

        log['created'] = time.time()
        log['epoch'] = epoch + 1

        self.send_metrics(logs, log['epoch'])
        self.send_optimizer_info(log['epoch'])

    def send_metrics(self, log, x):
        if 'acc' in log:
            # tf 1
            accuracy_log_name = 'acc'
            val_accuracy_log_name = 'val_acc'
        else:
            # tf2
            accuracy_log_name = 'accuracy'
            val_accuracy_log_name = 'val_accuracy'

        total_accuracy_validation = log.get(val_accuracy_log_name, None)
        total_accuracy_training = log.get(accuracy_log_name, None)

        if total_accuracy_validation: total_accuracy_validation = float(total_accuracy_validation)
        if total_accuracy_training: total_accuracy_training = float(total_accuracy_training)

        loss = log.get('loss', None)
        val_loss = log.get('val_loss', None)
        if loss is not None or val_loss is not None:
            if loss: loss = float(loss)
            if val_loss: val_loss = float(val_loss)
            print('loss, val_loss', loss, val_loss)
            self.loss_metric.send(loss, val_loss, x=x)

        accuracy = [total_accuracy_training, total_accuracy_validation]
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            accuracy = []
            losses = []
            for layer in self.model.output_layers:
                accuracy.append(log.get(layer.name + '_acc', None))
                accuracy.append(log.get('val_' + layer.name + '_acc', None))

                losses.append(log.get(layer.name + '_loss', None))
                losses.append(log.get('val_' + layer.name + '_loss', None))

            self.all_losses.send(*losses, x=x)

        self.accuracy_metric.send(*accuracy, x=x)

    def send_optimizer_info(self, epoch):
        self.learning_rate_metric.send(self.learning_rate_start, self.get_learning_rate(), x=epoch)

    def get_learning_rate(self):
        if hasattr(self.model, 'optimizer'):
            config = self.model.optimizer.get_config()

            if 'lr' in config and 'decay' in config and hasattr(self.model.optimizer, 'iterations'):
                iterations = self.model.optimizer.iterations
                # if hasattr(iterations, 'var') and hasattr(iterations.var, 'as_ndarray'):
                #     # plaidML
                #     ndarray = iterations.var.as_ndarray(None)
                #     iterations = float(ndarray)
                # else:
                iterations = float(keras.backend.get_value(iterations))

                return config['lr'] * (1. / (1. + config['decay'] * iterations))

            elif 'lr' in config:
                return config['lr']

    def has_multiple_inputs(self):
        return len(self.model.inputs) > 1

    def filter_invalid_json_values(self, dict: dict):
        for k, v in dict.items():
            if isinstance(v, (np.ndarray, np.generic)):
                dict[k] = v.tolist()
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1
