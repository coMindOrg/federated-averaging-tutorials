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
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt

# Import custom optimizer
import federated_averaging_optimizer

flags = tf.app.flags
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task that performs the variable "
                     "initialization ")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

# You can safely tune these variables
BATCH_SIZE = 32
EPOCHS = 5
INTERVAL_STEPS = 100
# ----------------

FLAGS = flags.FLAGS

if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

# Only enable GPU for worker 1 (not needed if training with separate machines)
if FLAGS.task_index == 0:
    print('--- GPU Disabled ---')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

#Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")

# Get the number of workers.
num_workers = len(worker_spec)
print('{} workers defined'.format(num_workers))

cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
# Parameter server will block here
if FLAGS.job_name == "ps":
    print('--- Parameter Server Ready ---')
    server.join()

# Load dataset as numpy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('Data loaded')

# List with class names to see the labels of the images with matplotlib
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Split dataset between workers
train_images = np.array_split(train_images, num_workers)[FLAGS.task_index]
train_labels = np.array_split(train_labels, num_workers)[FLAGS.task_index]
print('Local dataset size: {}'.format(train_images.shape[0]))

# Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

is_chief = (FLAGS.task_index == 0)

checkpoint_dir='logs_dir/federated_worker_{}/{}'.format(FLAGS.task_index, time())
print('Checkpoint directory: ' + checkpoint_dir)

worker_device = "/job:worker/task:%d" % FLAGS.task_index
print('Worker device: ' + worker_device + ' - is_chief: {}'.format(is_chief))

# Place all ops in the local worker by default
with tf.device(worker_device):
    global_step = tf.train.get_or_create_global_step()

    # Define input pipeline, place these ops in the cpu
    with tf.name_scope('dataset'), tf.device('/cpu:0'):
        # Placeholders for the iterator
        images_placeholder = tf.placeholder(train_images.dtype, [None, train_images.shape[1], train_images.shape[2]], name='images_placeholder')
        labels_placeholder = tf.placeholder(train_labels.dtype, [None], name='labels_placeholder')
        batch_size = tf.placeholder(tf.int64, name='batch_size')
        shuffle_size = tf.placeholder(tf.int64, name='shuffle_size')

        # Create dataset from numpy arrays, shuffle, repeat and batch
        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
        dataset = dataset.repeat(EPOCHS)
        dataset = dataset.batch(batch_size)
        # Define a feedable iterator and the initialization op
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

    # Define accuracy metric
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # Compare prediction with actual label
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(y, tf.int64))
        # Average correct predictions in the current batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_metric')
        accuracy_averages_op = summary_averages.apply([accuracy])
        # Store moving average of the accuracy
        tf.summary.scalar('accuracy', summary_averages.average(accuracy))

    # Define optimizer and training op
    with tf.name_scope('train'):
        # Define device setter to place copies of local variables
        device_setter = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
        # Wrap optimizer in a FederatedAveragingOptimizer for federated training
        optimizer = federated_averaging_optimizer.FederatedAveragingOptimizer(tf.train.AdamOptimizer(0.001), replicas_to_aggregate=num_workers, interval_steps=INTERVAL_STEPS, is_chief=is_chief, device_setter=device_setter)
        # Make train_op dependent on moving averages ops. Otherwise they will be
        # disconnected from the graph
        with tf.control_dependencies([loss_averages_op, accuracy_averages_op]):
            train_op = optimizer.minimize(loss, global_step=global_step)
        # Define a hook for optimizer initialization
        federated_hook = optimizer.make_session_run_hook()

    n_batches = int(train_images.shape[0] / BATCH_SIZE)
    last_step = int(n_batches * EPOCHS)

    print('Graph definition finished')

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        operation_timeout_in_ms=20000,
        device_filters=["/job:ps",
        "/job:worker/task:%d" % FLAGS.task_index])

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
            session.run(dataset_init_op, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels, batch_size: BATCH_SIZE, shuffle_size: train_images.shape[0]})

    # Hook to save just trainable_variables
    class _SaverHook(tf.train.SessionRunHook):
        def begin(self):
            self._saver = tf.train.Saver(tf.trainable_variables())

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(global_step)

        def after_run(self, run_context, run_values):
            step_value = run_values.results
            if step_value % n_batches == 0 and not step_value == 0:
                self._saver.save(run_context.session, checkpoint_dir+'/model.ckpt', step_value)

        def end(self, session):
            self._saver.save(session, checkpoint_dir+'/model.ckpt', session.run(global_step))

    # Make sure we do not define a chief worker
    with tf.name_scope('monitored_session'):
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                checkpoint_dir=checkpoint_dir,
                hooks=[_LoggerHook(), _InitHook(), _SaverHook(), federated_hook],
                config=sess_config,
                stop_grace_period_secs=10,
                save_checkpoint_secs=None) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if is_chief:
    print('--- Begin Evaluation ---')
    # Reset graph and load it again to clean tensors placed in other devices
    tf.reset_default_graph()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta', clear_devices=True)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Model restored')
        graph = tf.get_default_graph()
        images_placeholder = graph.get_tensor_by_name('dataset/images_placeholder:0')
        labels_placeholder = graph.get_tensor_by_name('dataset/labels_placeholder:0')
        batch_size = graph.get_tensor_by_name('dataset/batch_size:0')
        shuffle_size = graph.get_tensor_by_name('dataset/shuffle_size:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy_metric:0')
        predictions = graph.get_tensor_by_name('softmax/BiasAdd:0')
        dataset_init_op = graph.get_operation_by_name('dataset/dataset_init')
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
