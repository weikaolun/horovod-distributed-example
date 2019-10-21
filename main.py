# MIT License, see LICENSE
# Copyright (c) 2019 Paperspace Inc.
# Author: Michal Kulaczkowski

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
import errno
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow import keras

tf.logging.set_verbosity(tf.logging.INFO)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)


def parse_args():
    """Parse arguments"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='''Train a convolution neural network with MNIST dataset.
                            For distributed mode, you must run this with mpirun. See README.md''')

    # Experiment related parameters
    parser.add_argument('--local_data_root', type=str, default=os.path.join(FILE_DIR, 'data'),
                        help='Path to dataset. This path will be /data on Gradient.')
    parser.add_argument('--local_log_root', type=str, default=os.path.join(FILE_DIR, 'logs'),
                        help='Path to store logs and checkpoints. This path will be /logs on Gradient.')
    parser.add_argument('--data_subpath', type=str, default='',
                        help='Which sub-directory the data will sit inside local_data_root (locally) ' +
                             'or /data/ (on Gradient).')

    # CNN model params
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Size of the CNN kernels to use.')
    parser.add_argument('--hidden_units', type=str, default='32,64',
                        help='Comma-separated list of integers. Number of hidden units to use in CNN model.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate used in Adam optimizer.')
    parser.add_argument('--learning_decay', type=float, default=0.0001,
                        help='Exponential decay rate of the learning rate per step.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate used after each convolutional layer.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size to use during training and evaluation.')

    # Training params
    parser.add_argument('--verbosity', type=str, default='INFO', choices=['CRITICAL', 'ERROR', 'WARN', 'INFO', 'DEBUG'],
                        help='TF logging level. To see intermediate results printed, set this to INFO or DEBUG.')
    parser.add_argument('--fashion', action='store_true',
                        help='Download and use fashion MNIST data instead of the default handwritten digit MNIST.')
    parser.add_argument('--parallel_batches', type=int, default=2,
                        help='Number of parallel batches to prepare in data pipeline.')
    parser.add_argument('--max_ckpts', type=int, default=2,
                        help='Maximum number of checkpoints to keep.')
    parser.add_argument('--ckpt_steps', type=int, default=100,
                        help='How frequently to save a model checkpoint.')
    parser.add_argument('--save_summary_steps', type=int, default=10,
                        help='How frequently to save TensorBoard summaries.')
    parser.add_argument('--log_step_count_steps', type=int, default=10,
                        help='How frequently to log loss & global steps/s.')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='How frequently to run evaluation step.')
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='Maximum number of steps to run.')

    # Parse args
    opts = parser.parse_args()
    opts.data_dir = os.path.abspath(os.environ.get('PS_JOBSPACE', os.getcwd()) + '/data')
    opts.log_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/models') + '/mnist')

    opts.hidden_units = [int(n) for n in opts.hidden_units.split(',')]

    return opts


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Horovod: scale learning rate by the number of workers.
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001 * hvd.size(), momentum=0.9)

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)

        # Save accuracy scalar to Tensorboard output.
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.summary.scalar('train_accuracy', accuracy[1])

        tf.summary.scalar('loss', loss)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def run_mnist(flags_obj, train_data, train_labels, eval_data, eval_labels):
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd() + '/models') + '/horovod-mnist')

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(session_config=config))

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Horovod: adjust number of steps based on number of GPUs.
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000 // hvd.size(),
        hooks=[logging_hook, bcast_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    if flags_obj.export_dir:
        tf.logging.debug('Starting to Export model to {}'.format(str(flags_obj.export_dir)))
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'image': image,
        })
        mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn,
                                           strip_default_attrs=True)
        tf.logging.debug('Model Exported')

    return eval_results


def main(args):
    # Horovod: initialize Horovod.
    hvd.init()

    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (train_data, train_labels), (eval_data, eval_labels) = \
        keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    train_data = np.reshape(train_data, (-1, 784)) / 255.0
    eval_data = np.reshape(eval_data, (-1, 784)) / 255.0

    eval_results = run_mnist(args, train_data, train_labels, eval_data, eval_labels)

    print(eval_results)


if __name__ == "__main__":
    args = parse_args()
    tf.logging.set_verbosity(args.verbosity)

    tf.logging.debug('=' * 20 + ' Environment Variables ' + '=' * 20)
    for k, v in os.environ.items():
        tf.logging.debug('{}: {}'.format(k, v))

    tf.logging.debug('=' * 20 + ' Arguments ' + '=' * 20)
    for k, v in sorted(args.__dict__.items()):
        if v is not None:
            tf.logging.debug('{}: {}'.format(k, v))

    tf.logging.info('=' * 20 + ' Train starting ' + '=' * 20)
    main(args)
