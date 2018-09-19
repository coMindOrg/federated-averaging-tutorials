# Copyright 2018 coMind. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://comind.org/
# ==============================================================================

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from time import time

BATCH_SIZE = 32
EPOCHS = 5

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Data loaded')
print('Local dataset size: {}'.format(train_images.shape[0]))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

checkpoint_dir='logs_dir/{}'.format(time())
print('Checkpoint directory: ' + checkpoint_dir)

global_step = tf.train.get_or_create_global_step()

with tf.name_scope('dataset'), tf.device('/cpu:0'):
    images_placeholder = tf.placeholder(train_images.dtype, [None, train_images.shape[1], train_images.shape[2]], name='images_placeholder')
    labels_placeholder = tf.placeholder(train_labels.dtype, [None], name='labels_placeholder')
    batch_size = tf.placeholder(tf.int64, name='batch_size')
    shuffle_size = tf.placeholder(tf.int64, name='shuffle_size')

    dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(EPOCHS)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
    X, y = iterator.get_next()

flatten_layer = tf.layers.flatten(X, name='flatten')

dense_layer = tf.layers.dense(flatten_layer, 128, activation=tf.nn.relu, name='relu')

predictions = tf.layers.dense(dense_layer, 10, activation=tf.nn.softmax, name='softmax')

summary_averages = tf.train.ExponentialMovingAverage(0.9)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, predictions))
    loss_averages_op = summary_averages.apply([loss])
    tf.summary.scalar('cross_entropy', summary_averages.average(loss))

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(y, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_averages_op = summary_averages.apply([accuracy])
    tf.summary.scalar('accuracy', summary_averages.average(accuracy))

with tf.name_scope('train'):
    with tf.control_dependencies([loss_averages_op, accuracy_averages_op]):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

print('Graph definition finished')
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

n_batches = int(train_images.shape[0] / BATCH_SIZE)
last_step = int(n_batches * EPOCHS)
print('Training {} batches...'.format(last_step))

class _LoggerHook(tf.train.SessionRunHook):
  def begin(self):
      self._total_loss = 0
      self._total_acc = 0

  def before_run(self, run_context):
      return tf.train.SessionRunArgs([loss, accuracy, global_step])

  def after_run(self, run_context, run_values):
      loss_value, acc_value, step_value = run_values.results
      self._total_loss += loss_value
      self._total_acc += acc_value
      if (step_value + 1) % n_batches == 0:
          print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(int(step_value / n_batches) + 1, EPOCHS, self._total_loss / n_batches, self._total_acc / n_batches))
          self._total_loss = 0
          self._total_acc = 0

class _InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(dataset_init_op, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels, batch_size: BATCH_SIZE, shuffle_size: train_images.shape[0]})

with tf.name_scope('monitored_session'):
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            hooks=[_LoggerHook(), _InitHook()],
            config=sess_config,
            save_checkpoint_steps=n_batches) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

print('--- Begin Evaluation ---')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    print('Model restored')
    sess.run(dataset_init_op, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels, batch_size: test_images.shape[0], shuffle_size: 1})
    print('Test accuracy: {:4f}'.format(sess.run(accuracy)))
    predicted = sess.run(predictions)

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predicted[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
                                color=color)

plt.show(True)
