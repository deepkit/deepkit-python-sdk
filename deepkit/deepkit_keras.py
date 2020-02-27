# -*- coding: utf-8 -*-
from __future__ import division

import base64
import io
import math
import os
import sys
import time
from struct import pack

import PIL.Image
import numpy as np

if 'keras' in sys.modules:
    import keras
    from keras import Model
else:
    import tensorflow.keras as keras
    from tensorflow.keras import Model

import deepkit
from deepkit.tf import extract_model_graph, layer_visitor
from deepkit.utils.image import get_image_tales, get_layer_vis_square


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
    def __init__(self, debug_x=None):
        super(KerasCallback, self).__init__()

        self.context = deepkit.context()

        self.debug_x = debug_x

        self.epoch = 0
        self.batch = 0

        self.data_validation = None
        self.data_validation_size = None

        self.current = {}
        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.accuracy_metric = None
        self.all_losses = None
        self.loss_metric = None
        self.learning_rate_metric = None

        self.live_debug_data_x = 0
        self.snapshot_debug_data_x = 0

        self.learning_rate_start = 0
        self.last_live_debug_sent = 0
        self.last_live_futures = []
        self.last_snapshot_futures = []
        self.last_snapshot_last_time = 0
        self.last_snapshot_last_epoch = None

    def check_snapshot(self, epoch_end=False):
        """
        Checks if a snapshots needs to be created. Either for records, live view. or both.
        """
        if not self.context.debugger_controller.state: return
        state = self.context.debugger_controller.state
        record_needed = state.recording

        if state.recordingMode == 'second':
            diff = time.time() - self.last_snapshot_last_time
            if diff <= state.recordingSecond:
                # not enough time past, wait for next call
                record_needed = False

        if state.recordingMode == 'epoch':
            if not epoch_end: record_needed = False
            if self.epoch == self.last_snapshot_last_epoch:
                # nothing to do for records
                record_needed = False

        live_needed = state.live and (time.time() - self.last_live_debug_sent) > 1

        layers = list(state.watchingLayers.keys())

        if record_needed and state.recordingLayers == 'all':
            layers = []

            def visit_layer(layer):
                layers.append(layer.name)

            layer_visitor(self.model, visit_layer)
            # we manually create snapshot when layer differ (which is always the case for 'all')
            # from live's watchingLayers
            self.create_debug_snapshot(layers, and_as_live=False)
            if live_needed:
                self.send_live_debug_data()

            return

        # When a record and a live snapshot is needed, we do both at the same time
        if live_needed and record_needed:
            self.create_debug_snapshot(layers, and_as_live=True)
        elif record_needed:
            self.create_debug_snapshot(layers, and_as_live=False)
        elif live_needed:
            self.send_live_debug_data()

    def create_debug_snapshot(self, layer_names=None, and_as_live=False):
        if not self.context.debugger_controller: return
        # we don't send live debug data when not connected
        if not self.context.client.is_connected(): return

        if not len(layer_names): return

        layers = layer_names.copy()

        # wait for all previous to be sent first.
        try:
            for f in self.last_snapshot_futures: f.result()
        except Exception as e:
            print('Failing sending debug data', e)
            pass

        self.snapshot_debug_data_x += 1
        x = self.snapshot_debug_data_x
        self.context.client.job_action_threadsafe('addSnapshot', [
            x,
            time.time(),
            layers,
            self.epoch,
            self.batch,
        ])

        self.last_snapshot_last_time = time.time()

        data = self.get_image_and_histogram_from_layers(x, layers)

        for i, layer_name in enumerate(layers):
            jpeg, activations = data[i]
            whistogram, bhistogram = self.get_weight_histogram_from_layer(x, layer_name)

            self.last_snapshot_futures.append(self.context.client.job_action_threadsafe('setSnapshotLayerData', [
                x,
                layer_name,
                and_as_live,
                base64.b64encode(jpeg).decode(),
                base64.b64encode(activations).decode() if activations else None,
                base64.b64encode(whistogram).decode() if whistogram else None,
                base64.b64encode(bhistogram).decode() if bhistogram else None,
            ]))

    def send_live_debug_data(self):
        if not self.context.debugger_controller: return

        # we don't send live debug data when not connected
        if not self.context.client.is_connected(): return

        if not self.context.debugger_controller.state: return
        if len(self.context.debugger_controller.state.watchingLayers) == 0: return
        layers = self.context.debugger_controller.state.watchingLayers.copy()

        # wait for all previous to be sent first.
        try:
            for f in self.last_live_futures: f.result()
        except Exception as e:
            print('Failing sending debug data', e)
            pass

        self.live_debug_data_x += 1
        self.last_live_futures = []

        names = list(layers.keys())
        data = self.get_image_and_histogram_from_layers(self.live_debug_data_x, list(layers.keys()))
        for i, layer_name in enumerate(names):
            jpeg, activations = data[i]
            whistogram, bhistogram = self.get_weight_histogram_from_layer(self.live_debug_data_x, layer_name)

            self.last_live_futures.append(self.context.client.job_action_threadsafe('addLiveLayerData', [
                layer_name,
                base64.b64encode(jpeg).decode(),
                base64.b64encode(activations).decode() if activations else None,
                base64.b64encode(whistogram).decode() if whistogram else None,
                base64.b64encode(bhistogram).decode() if bhistogram else None,
            ]))

        self.last_live_debug_sent = time.time()

    def set_model(self, model):
        super().set_model(model)
        self.context.set_model_graph(extract_model_graph(self.model))

    def get_weight_histogram_from_layer(self, x, layer_name: str):
        layer_weights = self.model.get_layer(layer_name).get_weights()
        weights = None

        if len(layer_weights) > 0:
            h = np.histogram(layer_weights[0], bins=20)
            # <version><x><bins><...x><...y>, little endian
            # uint8|Uint32|Uint16|...Float32|...Uint32
            # B|L|H|...f|...L
            weights = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        biases = None
        if len(layer_weights) > 1:
            h = np.histogram(layer_weights[1], bins=20)
            biases = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        return weights, biases

    def get_image_and_histogram_from_layers(self, x, layer_names):
        outputs = []
        for layer_name in layer_names:
            layer = self.model.get_layer(layer_name)
            outputs.append(layer.output)

        model = Model(self.model.input, outputs)
        y = model.predict(self.debug_x, steps=1)

        result = []

        for i, name in enumerate(layer_names):
            # [0] we pick only the first in the batch
            result.append(self._image_and_histogram(x, name, y[i][0]))

        return result

    def _image_and_histogram(self, x, layer, data):
        h = np.histogram(data, bins=20)
        if isinstance(layer, (
                keras.layers.Conv2D,
                keras.layers.UpSampling2D,
                keras.layers.MaxPooling2D,
                keras.layers.AveragePooling2D,
                keras.layers.GlobalMaxPool2D,
        )):
            data = np.transpose(data, (2, 0, 1))
            image = PIL.Image.fromarray(get_image_tales(data))
        else:
            if len(data.shape) > 1:
                if len(data.shape) == 3:
                    data = np.transpose(data, (2, 0, 1))
                image = PIL.Image.fromarray(get_layer_vis_square(data))
            else:
                image = self.make_image_from_dense(data)

        histogram = pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()

        return self.pil_image_to_jpeg(image), histogram

    def make_image_from_dense(self, neurons):
        from .utils import array_to_img
        cols = int(math.ceil(math.sqrt(len(neurons))))

        even_length = cols * cols
        diff = even_length - len(neurons)
        if diff > 0:
            neurons = np.append(neurons, np.zeros(diff, dtype=neurons.dtype))

        img = array_to_img(neurons.reshape((1, cols, cols)))
        img = img.resize((cols * 8, cols * 8))

        return img

    def pil_image_to_jpeg(self, image):
        buffer = io.BytesIO()

        image.save(buffer, format="JPEG", optimize=True, quality=70)
        return buffer.getvalue()

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.last_batch_time = time.time()

        self.context.epoch(0, self.params['epochs'])
        self.context.set_info('parameters', get_total_params(self.model))

        # self.job_backend.upload_keras_graph(self.model)

        if self.model.optimizer and hasattr(self.model.optimizer, 'get_config'):
            config = self.model.optimizer.get_config()
            # deepkit.set_info('optimizer', type(self.model.optimizer).__name__)
            for i, v in config.items():
                self.context.set_info('optimizer.' + str(i), v)

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

        self.accuracy_metric = self.context.define_metric('accuracy', traces=traces)
        self.loss_metric = self.context.define_metric('loss')
        self.learning_rate_metric = self.context.define_metric('learning rate', traces=['start', 'end'])

        self.context.epoch(0, self.params['epochs'])
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            loss_traces = []
            for output in self.model.output_layers:
                loss_traces.append('train_' + output.name)
                loss_traces.append('val_' + output.name)

            self.all_losses = self.context.define_metric('loss_all', traces=loss_traces)

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
            deepkit.set_info('Batch size', batch_size)

        self.batch = batch

    def on_batch_end(self, batch, logs={}):
        self.filter_invalid_json_values(logs)

        deepkit.batch(batch + 1, self.current['nb_batches'], logs['size'])
        self.check_snapshot()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        self.learning_rate_start = self.get_learning_rate()

    def on_epoch_end(self, epoch, logs={}):
        log = logs.copy()

        self.filter_invalid_json_values(log)

        log['created'] = time.time()
        log['epoch'] = epoch + 1

        self.send_metrics(logs, log['epoch'])

        deepkit.epoch(log['epoch'], self.params['epochs'])
        self.send_optimizer_info(log['epoch'])
        self.check_snapshot(epoch_end=True)

    def send_metrics(self, log, x):
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
            self.loss_metric.send(x, loss, val_loss)

        accuracy = [total_accuracy_training, total_accuracy_validation]
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            accuracy = []
            losses = []
            for layer in self.model.output_layers:
                accuracy.append(log.get(layer.name + '_acc', None))
                accuracy.append(log.get('val_' + layer.name + '_acc', None))

                losses.append(log.get(layer.name + '_loss', None))
                losses.append(log.get('val_' + layer.name + '_loss', None))

            self.all_losses.send(x, losses)

        self.accuracy_metric.send(x, accuracy)

    def send_optimizer_info(self, epoch):
        self.learning_rate_metric.send(epoch, [self.learning_rate_start, self.get_learning_rate()])

    def get_learning_rate(self):
        if hasattr(self.model, 'optimizer'):
            config = self.model.optimizer.get_config()

            if 'lr' in config and 'decay' in config and hasattr(self.model.optimizer, 'iterations'):
                return config['lr'] * (
                        1. / (1. + config['decay'] * float(self.model.optimizer.iterations)))

            elif 'lr' in config:
                return config['lr']

    def has_multiple_inputs(self):
        return len(self.model.inputs) > 1

    def filter_invalid_json_values(self, dict):
        for k, v in six.iteritems(dict):
            if isinstance(v, (np.ndarray, np.generic)):
                dict[k] = v.tolist()
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1
