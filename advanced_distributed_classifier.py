# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import numpy as np
from time import time
import matplotlib.pyplot as plt

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

BATCH_SIZE = 128
EPOCHS = 250
EPOCHS_PER_DECAY = 50

FLAGS = flags.FLAGS

if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

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
if FLAGS.job_name == "ps":
    print('--- Parameter Server Ready ---')
    server.join()

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print('Data loaded')

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

train_images = np.split(train_images, num_workers)[FLAGS.task_index]
train_labels = np.split(train_labels, num_workers)[FLAGS.task_index]
print('Local dataset size: {}'.format(train_images.shape[0]))

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

is_chief = (FLAGS.task_index == 0)

checkpoint_dir='logs_dir/{}'.format(time())
print('Checkpoint directory: ' + checkpoint_dir)

worker_device = "/job:worker/task:%d" % FLAGS.task_index
print('Worker device: ' + worker_device + ' - is_chief: {}'.format(is_chief))

with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          cluster=cluster)):
    global_step = tf.train.get_or_create_global_step()

    with tf.name_scope('dataset'):
        def preprocess(image, label):
            casted_image = tf.cast(image, tf.float32, name='input_cast')
            resized_image = tf.image.resize_image_with_crop_or_pad(casted_image, 24, 24)
            distorted_image = tf.random_crop(casted_image, [24, 24, 3], name='random_crop')
            distorted_image = tf.image.random_flip_left_right(distorted_image)
            distorted_image = tf.image.random_brightness(distorted_image, 63)
            distorted_image = tf.image.random_contrast(distorted_image, 0.2, 1.8)
            processed_image = tf.image.per_image_standardization(distorted_image)
            result = tf.cond(train_mode, lambda: processed_image, lambda: resized_image)
            return result, label
        images_placeholder = tf.placeholder(train_images.dtype, [None, train_images.shape[1], train_images.shape[2], train_images.shape[3]], name='images_placeholder')
        labels_placeholder = tf.placeholder(train_labels.dtype, [None, 1], name='labels_placeholder')
        batch_size = tf.placeholder(tf.int64, name='batch_size')
        train_mode = tf.placeholder(tf.bool, name='train_mode')

        dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))
        dataset = dataset.map(lambda x, y: preprocess(x, y))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(EPOCHS)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        X, y = iterator.get_next()

    first_conv = tf.layers.conv2d(X, 64, 5, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2),name='first_conv')

    first_pool = tf.nn.max_pool(first_conv, [1, 3, 3 ,1], [1, 2, 2, 1], padding='SAME', name='first_pool')

    first_norm = tf.nn.lrn(first_pool, 4, alpha=0.001 / 9.0, beta=0.75, name='first_norm')

    second_conv = tf.layers.conv2d(first_norm, 64, 5, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2), name='second_conv')

    second_norm = tf.nn.lrn(second_conv, 4, alpha=0.001 / 9.0, beta=0.75, name='second_norm')

    second_pool = tf.nn.max_pool(second_norm, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='second_pool')

    flatten_layer = tf.layers.flatten(second_pool, name='flatten')

    first_relu = tf.layers.dense(flatten_layer, 384, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04), name='first_relu')

    second_relu = tf.layers.dense(first_relu, 192, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04), name='second_relu')

    predictions = tf.layers.dense(second_relu, 10, activation=tf.nn.softmax, kernel_initializer=tf.truncated_normal_initializer(stddev=1/192.0), name='softmax')

    summary_averages = tf.train.ExponentialMovingAverage(0.9)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, predictions))
        loss_averages_op = summary_averages.apply([loss])
        tf.summary.scalar('cross_entropy', summary_averages.average(loss))

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.cast(y, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_metric')
        accuracy_averages_op = summary_averages.apply([accuracy])
        tf.summary.scalar('accuracy', summary_averages.average(accuracy))

    with tf.name_scope('train'):
        lr = tf.train.exponential_decay(0.1, global_step, n_batches * EPOCHS_PER_DECAY, 0.1, staircase=True)
        optimizer = tf.train.SyncReplicasOptimizer(tf.train.AdamOptimizer(np.sqrt(num_workers) * lr), replicas_to_aggregate=num_workers)
        with tf.control_dependencies([loss_averages_op, accuracy_averages_op]):
            train_op = optimizer.minimize(loss, global_step=global_step)
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)

    print('Graph definition finished')

    n_batches = int(train_images.shape[0] / (BATCH_SIZE * num_workers))
    last_step = int(n_batches * EPOCHS)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps",
        "/job:worker/task:%d" % FLAGS.task_index])

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
          if (step_value + 1) % n_batches == 0 and not step_value == 0:
              print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(int(step_value / n_batches) + 1, EPOCHS, self._total_loss / n_batches, self._total_acc / n_batches))
              self._total_loss = 0
              self._total_acc = 0

      def end(self, session):
          print("Epoch {}/{} - loss: {:.4f} - acc: {:.4f}".format(int(session.run(global_step) / n_batches) + 1, EPOCHS, self._total_loss / n_batches, self._total_acc / n_batches))

    class _InitHook(tf.train.SessionRunHook):
        def after_create_session(self, session, coord):
            session.run(dataset_init_op, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels, batch_size: BATCH_SIZE, train_mode: True})

    with tf.name_scope('monitored_session'):
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                checkpoint_dir=checkpoint_dir,
                hooks=[_LoggerHook(), _InitHook(), sync_replicas_hook],
                config=sess_config,
                stop_grace_period_secs=10,
                save_checkpoint_steps=n_batches) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if is_chief:
    print('--- Begin Evaluation ---')
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
        train_mode = graph.get_tensor_by_name('dataset/train_mode:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy_metric:0')
        predictions = graph.get_tensor_by_name('softmax/BiasAdd:0')
        dataset_init_op = graph.get_operation_by_name('dataset/dataset_init')
        sess.run(dataset_init_op, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels, batch_size: test_images.shape[0], train_mode: False})
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
        true_label = test_labels[i][0]
        if predicted_label == true_label:
          color = 'green'
        else:
          color = 'red'
        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    class_names[true_label]),
                                    color=color)

    plt.show(True)
