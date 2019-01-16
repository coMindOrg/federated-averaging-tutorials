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

# Custom federated hook
from FederatedHook import _FederatedHook

# Helper libraries
import os
import numpy as np
from time import time

flags = tf.app.flags

flags.DEFINE_boolean("is_chief", False, "True if this worker is chief")

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# You can safely tune these variables
BATCH_SIZE = 32
EPOCHS = 5
INTERVAL_STEPS = 100 # Steps between averages
WAIT_TIME = 30 # How many seconds to wait for new workers to connect
# -----------------

# Set these IPs to your own, can leave as localhost for local testing
CHIEF_PUBLIC_IP = 'localhost:7777' # Public IP of the chief worker
CHIEF_PRIVATE_IP = 'localhost:7777' # Private IP of the chief worker

# Create the custom hook
federated_hook = _FederatedHook(FLAGS.is_chief, CHIEF_PRIVATE_IP, CHIEF_PUBLIC_IP, WAIT_TIME, INTERVAL_STEPS)

# Load dataset as numpy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Split dataset
train_images = np.array_split(train_images, federated_hook.num_workers)[federated_hook.task_index]
train_labels = np.array_split(train_labels, federated_hook.num_workers)[federated_hook.task_index]

# You can safely tune this variable
SHUFFLE_SIZE = train_images.shape[0]
# -----------------

print('Local dataset size: {}'.format(train_images.shape[0]))

# Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

CHECKPOINT_DIR = 'logs_dir/{}'.format(time())

global_step = tf.train.get_or_create_global_step()

# Define input pipeline, place these ops in the cpu
with tf.name_scope('dataset'), tf.device('/cpu:0'):
    # Placeholders for the iterator
    images_placeholder = tf.placeholder(train_images.dtype, [None, train_images.shape[1], train_images.shape[2]])
    labels_placeholder = tf.placeholder(train_labels.dtype, [None])
    batch_size = tf.placeholder(tf.int64)
    shuffle_size = tf.placeholder(tf.int64, name='shuffle_size')

    # Create dataset, shuffle, repeat and batch
    dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
    X, y = iterator.get_next()

# Define our model
flatten_layer = tf.layers.flatten(X, name='flatten')

dense_layer = tf.layers.dense(flatten_layer, 128, activation=tf.nn.relu, name='relu')

predictions = tf.layers.dense(dense_layer, 10, activation=tf.nn.softmax, name='softmax')

# Object to keep moving averages of our metrics (for tensorboard)
summary_averages = tf.train.ExponentialMovingAverage(0.9)

# Define cross_entropy loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, predictions))
    loss_averages_op = summary_averages.apply([loss])
    # Store moving average of the loss
    tf.summary.scalar('cross_entropy', summary_averages.average(loss))

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # Compare prediction with actual label
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(y, tf.int64))
    # Average correct predictions in the current batch
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_averages_op = summary_averages.apply([accuracy])
    # Store moving average of the accuracy
    tf.summary.scalar('accuracy', summary_averages.average(accuracy))

# Define optimizer and training op
with tf.name_scope('train'):
    # Make train_op dependent on moving averages ops. Otherwise they will be
    # disconnected from the graph
    with tf.control_dependencies([loss_averages_op, accuracy_averages_op]):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

SESS_CONFIG = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

N_BATCHES = int(train_images.shape[0] / BATCH_SIZE)
LAST_STEP = int(N_BATCHES * EPOCHS)

# Logger hook to keep track of the training
class _LoggerHook(tf.train.SessionRunHook):
    def begin(self):
        """ Run this in session begin """
        self._total_loss = 0
        self._total_acc = 0

    def before_run(self, run_context):
        """ Run this in session before_run """
        return tf.train.SessionRunArgs([loss, accuracy, global_step])

    def after_run(self, run_context, run_values):
        """ Run this in session after_run """
        loss_value, acc_value, step_value = run_values.results
        self._total_loss += loss_value
        self._total_acc += acc_value
        if (step_value + 1) % N_BATCHES == 0:
            print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(
                int(step_value / N_BATCHES) + 1,
                EPOCHS, self._total_loss / N_BATCHES,
                self._total_acc / N_BATCHES))
            self._total_loss = 0
            self._total_acc = 0

class _InitHook(tf.train.SessionRunHook):
    """ Hook to initialize the dataset """
    def after_create_session(self, session, coord):
        """ Run this after creating session """
        session.run(dataset_init_op, feed_dict={
            images_placeholder: train_images,
            labels_placeholder: train_labels,
            shuffle_size: SHUFFLE_SIZE, batch_size: BATCH_SIZE})

print("Worker {} ready".format(federated_hook.task_index))

with tf.name_scope('monitored_session'):
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=CHECKPOINT_DIR,
            hooks=[_LoggerHook(), _InitHook(), federated_hook],
            config=SESS_CONFIG,
            save_checkpoint_steps=N_BATCHES) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

print('--- Begin Evaluation ---')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    print('Model restored')
    sess.run(dataset_init_op, feed_dict={
        images_placeholder: test_images,
        labels_placeholder: test_labels,
        shuffle_size: 1, batch_size: test_images.shape[0]})
    print('Test accuracy: {:4f}'.format(sess.run(accuracy)))
