import datetime

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, datasets
import deepkit

experiment = deepkit.experiment()
experiment.add_file('model.py')

(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
x = x.reshape(x.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x = x / 255.0
y = tf.one_hot(y, depth=10, dtype=tf.float32)
y_val = tf.one_hot(y_val, depth=10)
print('x/y shape:', x.shape, y.shape)


def train_gen():
    global x, y
    for x2, y2 in zip(x, y):
        yield (x2, x2), y2
        # yield x2, y2


train_dataset = tf.data.Dataset.from_generator(
    train_gen,
    ((tf.float32, tf.float32), tf.float32),
    # (tf.TensorShape([28, 28]), tf.TensorShape([10]))
    ((tf.TensorShape([28, 28, 1]), tf.TensorShape([28, 28, 1])), tf.TensorShape([10]))
)
train_dataset = train_dataset.batch(100)
# val_dataset = train_dataset.batch(10)

# train_dataset, val_dataset = mnist_dataset()

# resnet = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=10
# )

# model = tf.keras.Sequential([
#     resnet,
#     layers.Dense(10, name='fickficki')
# ])

input1 = layers.Input((28, 28, 1))
input2 = layers.Input((28, 28, 1))

conv1 = layers.Convolution2D(64, (1, 1), activation='relu')(input1)
conv2 = layers.Convolution2D(64, (1, 1), activation='relu')(conv1)
rs1 = layers.Flatten()(conv2)
rs2 = layers.Flatten()(input2)

d1 = layers.Dense(64, activation='relu')(rs1)
d2 = layers.Dense(64, activation='relu')(rs2)
c1 = layers.Concatenate()([d1, d2])
d3 = layers.Dense(64, name='YoloDense', activation='relu')(c1)

output1 = layers.Dense(10)(d3)
model = Model(inputs=[input1, input2], outputs=[output1])

model.summary()

experiment.watch_keras_model(model)
deepkit_callback = experiment.create_keras_callback()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# no need to use compile if you have no loss/optimizer/metrics involved here.
model.compile(optimizer=optimizers.Adam(0.001),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset.repeat(), epochs=30, steps_per_epoch=500,
          validation_data=train_dataset.repeat(),
          validation_steps=2,
          callbacks=[tensorboard_callback, deepkit_callback]
          )
