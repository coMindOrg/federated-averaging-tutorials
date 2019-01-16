"""# Copyright 2018 coMind. All Rights Reserved.
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
# =============================================================================="""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from time import time
from mpi4py import MPI
import sys

# Let the code know about the MPI config
COMM = MPI.COMM_WORLD

# Load dataset as numpy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Split dataset
train_images = np.array_split(train_images, COMM.size)[COMM.rank]
train_labels = np.array_split(train_labels, COMM.size)[COMM.rank]

# You can safely tune these variables
BATCH_SIZE = 32
SHUFFLE_SIZE = train_images.shape[0]
EPOCHS = 5
INTERVAL_STEPS = 100
# -----------------

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
        if (step_value + 1) % N_BATCHES == 0 and COMM.rank == 0:
            print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(
                int(step_value / N_BATCHES) + 1,
                EPOCHS, self._total_loss / N_BATCHES, self._total_acc / N_BATCHES))
            sys.stdout.flush()
            self._total_loss = 0
            self._total_acc = 0

# Custom hook
class _FederatedHook(tf.train.SessionRunHook):
    def __init__(self, comm):
        """ Initialize Hook """
        # Store the MPI config
        self._comm = comm

    def _create_placeholders(self):
        """ Create placeholders for all the trainable variables """
        for var in tf.trainable_variables():
            self._placeholders.append(
                tf.placeholder_with_default(
                    var, var.shape, name="%s/%s" % ("FedAvg", var.op.name)))

    def _assign_vars(self, local_vars):
        """ Assign value feeded to placeholders to local vars """
        reassign_ops = []
        for var, fvar in zip(local_vars, self._placeholders):
            reassign_ops.append(tf.assign(var, fvar))
        return tf.group(*(reassign_ops))

    def _gather_weights(self, session):
        """Gather all weights in the chief worker"""
        gathered_weights = []
        for var in tf.trainable_variables():
            value = session.run(var)
            value = self._comm.gather(value, root=0)
            gathered_weights.append(np.array(value))
        return gathered_weights

    def _broadcast_weights(self, session):
        """Broadcast averaged weights to all workers"""
        broadcasted_weights = []
        for var in tf.trainable_variables():
            value = session.run(var)
            value = self._comm.bcast(value, root=0)
            broadcasted_weights.append(np.array(value))
        return broadcasted_weights

    def begin(self):
        """ Run this in session begin """
        self._placeholders = []
        self._create_placeholders()
        # Op to initialize update the weights
        self._update_local_vars_op = self._assign_vars(tf.trainable_variables())

    def after_create_session(self, session, coord):
        """ Run this after creating session """
        # Broadcast weights
        broadcasted_weights = self._broadcast_weights(session)
        # Initialize the workers at the same point
        if self._comm.rank != 0:
            feed_dict = {}
            for placeh, bweight in zip(self._placeholders, broadcasted_weights):
                feed_dict[placeh] = bweight
            session.run(self._update_local_vars_op, feed_dict=feed_dict)

    def before_run(self, run_context):
        """ Run this in session before_run """
        return tf.train.SessionRunArgs(global_step)

    def after_run(self, run_context, run_values):
        """ Run this in session after_run """
        step_value = run_values.results
        session = run_context.session
        # Check if we should average
        if step_value % INTERVAL_STEPS == 0 and not step_value == 0:
            gathered_weights = self._gather_weights(session)
            # Chief gather weights and averages
            if self._comm.rank == 0:
                print('Average applied, iter: {}/{}'.format(step_value, LAST_STEP))
                sys.stdout.flush()
                for i, elem in enumerate(gathered_weights):
                    gathered_weights[i] = np.mean(elem, axis=0)
                feed_dict = {}
                for placeh, gweight in zip(self._placeholders, gathered_weights):
                    feed_dict[placeh] = gweight
                session.run(self._update_local_vars_op, feed_dict=feed_dict)
            # The rest get the averages and update their local model
            broadcasted_weights = self._broadcast_weights(session)
            if self._comm.rank != 0:
                feed_dict = {}
                for placeh, bweight in zip(self._placeholders, broadcasted_weights):
                    feed_dict[placeh] = bweight
                session.run(self._update_local_vars_op, feed_dict=feed_dict)

# Hook to initialize the dataset
class _InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        """ Run this after creating session """
        session.run(dataset_init_op, feed_dict={
            images_placeholder: train_images,
            labels_placeholder: train_labels,
            batch_size: BATCH_SIZE, shuffle_size: SHUFFLE_SIZE})

print("Worker {} ready".format(COMM.rank))
sys.stdout.flush()

with tf.name_scope('monitored_session'):
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=CHECKPOINT_DIR,
            hooks=[_LoggerHook(), _InitHook(), _FederatedHook(COMM)],
            config=SESS_CONFIG,
            save_checkpoint_steps=N_BATCHES) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

if COMM.rank == 0:
    print('--- Begin Evaluation ---')
    sys.stdout.flush()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
        sys.stdout.flush()
        sess.run(dataset_init_op, feed_dict={
            images_placeholder: test_images, labels_placeholder: test_labels,
            batch_size: test_images.shape[0], shuffle_size: 1})
        print('Test accuracy: {:4f}'.format(sess.run(accuracy)))
        sys.stdout.flush()
