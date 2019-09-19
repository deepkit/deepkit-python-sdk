# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import PIL.Image
import math
import deepkit
from keras.callbacks import Callback
from keras import backend as K

import keras.layers.convolutional

import numpy as np

from deepkit.utils.image import get_layer_vis_square, get_layer_vis_square_raw, get_image_tales
import six


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

    flattened_layers = model.flattened_layers if hasattr(
        model, 'flattened_layers') else model.layers

    for i in range(len(flattened_layers)):
        total_params += flattened_layers[i].count_params()

    return total_params


# compatibility with keras 1.x
def image_data_format():
    if hasattr(K, 'image_data_format'):
        return K.image_data_format()

    if K.image_dim_ordering() == 'th':
        return 'channel_first'
    else:
        return 'channel_last'


class JobImage:
    def __init__(self, name, pil_image, label=None, pos=None):
        self.id = name
        if not isinstance(pil_image, PIL.Image.Image):
            raise Exception('JobImage requires a PIL.Image as image argument.')

        self.image = pil_image
        self.label = label
        self.pos = pos

        if self.pos is None:
            self.pos = time.time()


class KerasCallback(Callback):
    def __init__(self, model):
        super(KerasCallback, self).__init__()
        self.validation_per_batch = []

        self.model = model
        self.insight_layer = []

        self.epoch = 0

        self.data_validation = None
        self.data_validation_size = None

        self.current = {}
        self.last_batch_time = time.time()
        self.start_time = time.time()
        self.accuracy_metric = None
        self.all_losses = None
        self.loss_metric = None
        self.learning_rate_metric = None

        self.learning_rate_start = 0

    def add_insight_layer(self, layer):
        self.insight_layer.append(layer)

    def on_train_begin(self, logs={}):
        self.start_time = time.time()
        self.last_batch_time = time.time()

        deepkit.set_status('TRAINING')
        deepkit.set_info('parameters', get_total_params(self.model))
        deepkit.set_info('backend', K.backend())
        deepkit.set_info('keras.version', keras.__version__)
        deepkit.set_info('keras.floatx', K.floatx())

        if hasattr(K, 'image_dim_ordering'):
            deepkit.set_info('keras.format', K.image_dim_ordering())

        # self.job_backend.upload_keras_graph(self.model)

        if self.model.optimizer and hasattr(self.model.optimizer, 'get_config'):
            config = self.model.optimizer.get_config()
            deepkit.set_info('optimizer', type(self.model.optimizer).__name__)
            for i, v in config.items():
                deepkit.set_info('optimizer.' + str(i), v)

        # compatibility with keras 1.x
        if 'epochs' not in self.params and 'nb_epoch' in self.params:
            self.params['epochs'] = self.params['nb_epoch']
        if 'samples' not in self.params and 'nb_sample' in self.params:
            self.params['samples'] = self.params['nb_sample']

        xaxis = {
            # 'range': [1, self.params['epochs']],
            # 'title': u'Epoch ⇢'
        }
        yaxis = {
            'tickformat': '%',
            'hoverformat': '%',
            'rangemode': 'tozero'
        }

        traces = ['training', 'validation']
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers263) > 1:
            traces = []
            for output in self.model.output_layers:
                traces.append('train_' + output.name)
                traces.append('val_' + output.name)

        self.accuracy_metric = deepkit.create_metric(
            'accuracy',
            main=True, traces=traces, xaxis=xaxis, yaxis=yaxis
        )
        self.loss_metric = deepkit.create_loss_metric('loss', xaxis=xaxis)
        self.learning_rate_metric = deepkit.create_metric('learning rate', traces=['start', 'end'], xaxis=xaxis)

        deepkit.epoch(0, self.params['epochs'])
        if hasattr(self.model, 'output_layers') and len(self.model.output_layers) > 1:
            loss_traces = []
            for output in self.model.output_layers:
                loss_traces.append('train_' + output.name)
                loss_traces.append('val_' + output.name)

            self.all_losses = deepkit.create_metric('loss_all', main=True, xaxis=xaxis, traces=loss_traces)

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

    def on_batch_end(self, batch, logs={}):
        self.filter_invalid_json_values(logs)
        loss = logs['loss']

        self.validation_per_batch.append(loss)

        deepkit.batch(batch + 1, self.current['nb_batches'], logs['size'])

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch;
        self.learning_rate_start = self.get_learning_rate()

    def on_epoch_end(self, epoch, logs={}):
        log = logs.copy()

        self.filter_invalid_json_values(log)

        log['created'] = time.time()
        log['epoch'] = epoch + 1

        self.send_metrics(logs, log['epoch'])

        deepkit.epoch(log['epoch'], self.params['epochs'])
        self.send_optimizer_info(log['epoch'])

    def send_metrics(self, log, x):
        accuracy_log_name = 'acc'
        val_accuracy_log_name = 'val_acc'

        total_accuracy_validation = log.get(val_accuracy_log_name, None)
        total_accuracy_training = log.get(accuracy_log_name, None)

        loss = log.get('loss', None), log.get('val_loss', None)
        if loss[0] is not None or loss[1] is not None:
            self.loss_metric.send(x, loss[0], loss[1])

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

            from keras.optimizers import Adadelta, Adam, Adamax, Adagrad, RMSprop, SGD

            if isinstance(self.model.optimizer, Adadelta) or isinstance(self.model.optimizer, Adam) \
                    or isinstance(self.model.optimizer, Adamax) or isinstance(self.model.optimizer, Adagrad) \
                    or isinstance(self.model.optimizer, RMSprop) or isinstance(self.model.optimizer, SGD):
                return config['lr'] * (
                        1. / (1. + config['decay'] * float(K.get_value(self.model.optimizer.iterations))))

            elif 'lr' in config:
                return config['lr']

    def is_image_shape(self, x):
        if len(x.shape) != 3 and len(x.shape) != 2:
            return False

        if len(x.shape) == 2:
            return True

        #  check if it has either 1 or 3 channel
        if K.image_dim_ordering() == 'th':
            return (x.shape[0] == 1 or x.shape[0] == 3)

        if K.image_dim_ordering() == 'tf':
            return (x.shape[2] == 1 or x.shape[2] == 3)

    def has_multiple_inputs(self):
        return len(self.model.inputs) > 1

    def build_insight_images(self):
        if self.insights_x is None:
            print("Insights requested, but no 'insights_x' in create_keras_callback() given.")

        images = []
        input_data_x_sample = []

        if self.has_multiple_inputs():
            if not isinstance(self.insights_x, dict) and not isinstance(self.insights_x, (list, tuple)):
                raise Exception('insights_x must be a list or dict')

            for i, layer in enumerate(self.model.input_layers):
                x = self.insights_x[i] if isinstance(self.insights_x, list) else self.insights_x[layer.name]
                input_data_x_sample.append([x])
        else:
            x = self.insights_x
            input_data_x_sample.append([x])

        for i, layer in enumerate(self.model.input_layers):
            x = input_data_x_sample[i][0]
            if len(x.shape) == 3 and self.is_image_shape(x):
                if K.image_dim_ordering() == 'tf':
                    x = np.transpose(x, (2, 0, 1))

                image = self.make_image(x)
                if image:
                    images.append(JobImage(layer.name, image))

        uses_learning_phase = self.model.uses_learning_phase
        inputs = self.model.inputs[:]

        if uses_learning_phase:
            inputs += [K.learning_phase()]
            input_data_x_sample += [0.]  # disable learning_phase

        layers = self.model.layers + self.insight_layer

        pos = 0
        for layer in layers:
            if isinstance(layer, keras.layers.convolutional.Convolution2D) or isinstance(layer,
                                                                                         keras.layers.convolutional.MaxPooling2D) \
                    or isinstance(layer, keras.layers.convolutional.UpSampling2D):
                fn = K.function(inputs, self.get_layout_output_tensors(layer))

                result = fn(input_data_x_sample)
                Y = result[0]

                data = Y[0]

                if len(data.shape) == 3:
                    if K.image_dim_ordering() == 'tf':
                        data = np.transpose(data, (2, 0, 1))

                    image = PIL.Image.fromarray(get_image_tales(data))
                    pos += 1
                    images.append(JobImage(layer.name, image, pos=pos))

                if layer.get_weights():
                    data = layer.get_weights()[0]

                    # Keras 1 has channel only in last element when dim_ordering=tf
                    is_weights_channel_last = keras.__version__[0] == '1' and K.image_dim_ordering() == 'tf'

                    # Keras > 1 has channel always in last element
                    if keras.__version__[0] != '1':
                        is_weights_channel_last = True

                    # move channel/filters to first elements to generate correct image
                    if is_weights_channel_last:
                        data = np.transpose(data, (2, 3, 0, 1))

                    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

                    image = PIL.Image.fromarray(get_image_tales(data))
                    pos += 1
                    images.append(JobImage(layer.name + '_weights', image, layer.name + ' weights', pos=pos))

            elif isinstance(layer, keras.layers.ZeroPadding2D) or isinstance(layer,
                                                                             keras.layers.ZeroPadding1D) or isinstance(
                layer, keras.layers.ZeroPadding3D):
                pass
            elif isinstance(layer, keras.layers.noise.GaussianDropout) or isinstance(layer,
                                                                                     keras.layers.noise.GaussianNoise):
                pass
            elif isinstance(layer, keras.layers.Dropout):
                pass
            else:
                outputs = self.get_layout_output_tensors(layer)

                if len(outputs) > 0:
                    fn = K.function(inputs, outputs)
                    Y = fn(input_data_x_sample)[0]
                    Y = np.squeeze(Y)

                    if Y.size == 1:
                        Y = np.array([Y])

                    image = None
                    if len(Y.shape) > 1:
                        if len(Y.shape) == 3 and self.is_image_shape(Y) and K.image_dim_ordering() == 'tf':
                            Y = np.transpose(Y, (2, 0, 1))

                        image = PIL.Image.fromarray(get_layer_vis_square(Y))
                    elif len(Y.shape) == 1:
                        image = self.make_image_from_dense(Y)

                    if image:
                        pos += 1
                        images.append(JobImage(layer.name, image, pos=pos))

        return images

    def get_layout_output_tensors(self, layer):
        outputs = []

        if hasattr(layer, 'inbound_nodes'):
            for idx, node in enumerate(layer.inbound_nodes):
                outputs.append(layer.get_output_at(idx))

        return outputs

    def filter_invalid_json_values(self, dict):
        for k, v in six.iteritems(dict):
            if isinstance(v, (np.ndarray, np.generic)):
                dict[k] = v.tolist()
            if math.isnan(v) or math.isinf(v):
                dict[k] = -1

    def make_image(self, data):
        from keras.preprocessing.image import array_to_img
        try:
            if len(data.shape) == 2:
                # grayscale image, just add once channel
                data = data.reshape((data.shape[0], data.shape[1], 1))

            image = array_to_img(data)
        except Exception:
            return None

        # image = image.resize((128, 128))

        return image

    def make_image_from_dense_softmax(self, neurons):
        from .utils import array_to_img

        img = array_to_img(neurons.reshape((1, len(neurons), 1)))
        img = img.resize((9, len(neurons) * 8))

        return img

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
