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

# TensorFlow
import tensorflow as tf

# Helper libraries
import os
import numpy as np
from time import time
import multiprocessing

# You can safely tune these variables
BATCH_SIZE = 128
SHUFFLE_SIZE = BATCH_SIZE * 100
EPOCHS = 250
EPOCHS_PER_DECAY = 50
BATCHES_TO_PREFETCH = 1
# ----------------

# Dataset dependent constants
num_train_images = 50000
num_test_images = 10000
height = 32
width = 32
channels = 3
num_batch_files = 5

# Path to TFRecord files (check readme for instructions on how to get these files)
cifar10_train_files = ['cifar-10-tf-records/train{}.tfrecords'.format(i) for i in range(num_batch_files)]
cifar10_test_file = 'cifar-10-tf-records/test.tfrecords'

# Shuffle filenames before loading them
np.random.shuffle(cifar10_train_files)

checkpoint_dir='logs_dir/{}'.format(time())
print('Checkpoint directory: ' + checkpoint_dir)

global_step = tf.train.get_or_create_global_step()

# Check number of available CPUs
cpu_count = multiprocessing.cpu_count()

# Define input pipeline, place these ops in the cpu
with tf.name_scope('dataset'), tf.device('/cpu:0'):
    # Map function to decode data and preprocess it
    def preprocess(serialized_examples):
        # Parse a batch
        features = tf.parse_example(serialized_examples, {'image': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})
        # Decode and reshape image
        image = tf.map_fn(lambda img: tf.reshape(tf.decode_raw(img, tf.uint8), tf.stack([height, width, channels])), features['image'], dtype=tf.uint8, name='decode')
        # Cast image
        casted_image = tf.cast(image, tf.float32, name='input_cast')
        # Resize image for testing
        resized_image = tf.image.resize_image_with_crop_or_pad(casted_image, 24, 24)
        # Augment images for training
        distorted_image = tf.map_fn(lambda img: tf.random_crop(img, [24, 24, 3]), casted_image, name='random_crop')
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, 63)
        distorted_image = tf.image.random_contrast(distorted_image, 0.2, 1.8)
        # Check if test or train mode
        result = tf.cond(train_mode, lambda: distorted_image, lambda: resized_image)
        # Standardize images
        processed_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), result, name='standardization')
        return processed_image, features['label']
    # Placeholders for the iterator
    filename_placeholder = tf.placeholder(tf.string, name='input_filename')
    batch_size = tf.placeholder(tf.int64, name='batch_size')
    shuffle_size = tf.placeholder(tf.int64, name='shuffle_size')
    train_mode = tf.placeholder(tf.bool, name='train_mode')

    # Create dataset, shuffle, repeat, batch, map and prefetch
    dataset = tf.data.TFRecordDataset(filename_placeholder)
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess, cpu_count)
    dataset = dataset.prefetch(BATCHES_TO_PREFETCH)
    # Define a feedable iterator and the initialization op
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
    X, y = iterator.get_next()

# Define our model
first_conv = tf.layers.conv2d(X, 64, 5, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2), name='first_conv')

first_pool = tf.nn.max_pool(first_conv, [1, 3, 3 ,1], [1, 2, 2, 1], padding='SAME', name='first_pool')

first_norm = tf.nn.lrn(first_pool, 4, alpha=0.001 / 9.0, beta=0.75, name='first_norm')

second_conv = tf.layers.conv2d(first_norm, 64, 5, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2), name='second_conv')

second_norm = tf.nn.lrn(second_conv, 4, alpha=0.001 / 9.0, beta=0.75, name='second_norm')

second_pool = tf.nn.max_pool(second_norm, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='second_pool')

flatten_layer = tf.layers.flatten(second_pool, name='flatten')

first_relu = tf.layers.dense(flatten_layer, 384, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04), name='first_relu')

second_relu = tf.layers.dense(first_relu, 192, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04), name='second_relu')

logits = tf.layers.dense(second_relu, 10, kernel_initializer=tf.truncated_normal_initializer(stddev=1/192.0), name='logits')

# Object to keep moving averages of our metrics (for tensorboard)
summary_averages = tf.train.ExponentialMovingAverage(0.9)

# Define cross_entropy loss
with tf.name_scope('loss'):
    base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='base_loss')
    # Add regularization loss to both relu layers
    regularizer_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'relu/kernel' in v.name], name='regularizer_loss') * 0.004
    loss = tf.add(base_loss, regularizer_loss)
    loss_averages_op = summary_averages.apply([loss])
    # Store moving average of the loss
    tf.summary.scalar('cross_entropy', summary_averages.average(loss))

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # Compare prediction with actual label
        correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    # Average correct predictions in the current batch
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_metric')
    accuracy_averages_op = summary_averages.apply([accuracy])
    # Store moving average of the accuracy
    tf.summary.scalar('accuracy', summary_averages.average(accuracy))

n_batches = int(num_train_images / BATCH_SIZE)
last_step = int(n_batches * EPOCHS)

# Define moving averages of the trainable variables. This sometimes improve
# the performance of the trained model
with tf.name_scope('variable_averages'):
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

# Define optimizer and training op
with tf.name_scope('train'):
    # Make decaying learning rate
    lr = tf.train.exponential_decay(0.1, global_step, n_batches * EPOCHS_PER_DECAY, 0.1, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # Make train_op dependent on moving averages ops. Otherwise they will be
    # disconnected from the graph
    with tf.control_dependencies([loss_averages_op, accuracy_averages_op, variable_averages_op]):
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

print('Graph definition finished')
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

print('Training {} batches...'.format(last_step))

# Logger hook to keep track of the training
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

# Hook to initialize the dataset
class _InitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        session.run(dataset_init_op, feed_dict={filename_placeholder: cifar10_train_files, batch_size: BATCH_SIZE, shuffle_size: SHUFFLE_SIZE, train_mode: True})

with tf.name_scope('monitored_session'):
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            hooks=[_LoggerHook(), _InitHook(), tf.train.CheckpointSaverHook(checkpoint_dir=checkpoint_dir, save_steps=n_batches, saver=tf.train.Saver(variable_averages.variables_to_restore()))],
            config=sess_config,
            save_checkpoint_secs=None) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

print('--- Begin Evaluation ---')
# Reset graph and place ops in cpu to avoid OOM
tf.reset_default_graph()
with tf.device('/cpu:0'), tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Model restored')
    graph = tf.get_default_graph()
    filename_placeholder = graph.get_tensor_by_name('dataset/input_filename:0')
    batch_size = graph.get_tensor_by_name('dataset/batch_size:0')
    shuffle_size = graph.get_tensor_by_name('dataset/shuffle_size:0')
    train_mode = graph.get_tensor_by_name('dataset/train_mode:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy_metric:0')
    dataset_init_op = graph.get_operation_by_name('dataset/dataset_init')
    sess.run(dataset_init_op, feed_dict={filename_placeholder: cifar10_test_file, batch_size: num_test_images, shuffle_size: 1, train_mode: False})
    print('Test accuracy: {:4f}'.format(sess.run(accuracy)))
