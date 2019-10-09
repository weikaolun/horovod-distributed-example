# MIT License, see LICENSE
# Copyright (c) 2019 Paperspace Inc.
# Author: Michal Kulaczkowski

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import horovod.tensorflow as hvd
import tensorflow as tf

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from utils import train_dataset, test_dataset


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


def get_input_fn(opts, is_train=True):
    """Returns input_fn.  is_train=True shuffles and repeats data indefinitely"""

    def input_fn():
        with tf.device('/cpu:0'):
            if is_train:
                dataset = train_dataset(opts.data_dir, fashion=opts.fashion)
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=5 * opts.batch_size, count=None))
            else:
                dataset = test_dataset(opts.data_dir, fashion=opts.fashion)
            dataset = dataset.batch(batch_size=opts.batch_size)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()

    return input_fn


def cnn_net(input_tensor, opts):
    """Return logits output from CNN net"""
    temp = tf.reshape(input_tensor, shape=(-1, 28, 28, 1), name='input_image')
    for i, n_units in enumerate(opts.hidden_units):
        temp = tf.layers.conv2d(temp, filters=n_units, kernel_size=opts.kernel_size, strides=(2, 2),
                                activation=tf.nn.relu, name='cnn' + str(i))
        temp = tf.layers.dropout(temp, rate=opts.dropout)
    temp = tf.reduce_mean(temp, axis=(2, 3), keepdims=False, name='average')
    return tf.layers.dense(temp, 10)


def get_model_fn(opts):
    """Return model fn to be used for Estimator class"""

    def model_fn(features, labels, mode):
        """Returns EstimatorSpec for different mode (train/eval/predict)"""
        logits = cnn_net(features, opts)
        pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions={'logits': logits, 'pred': pred})

        cent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
        loss = tf.reduce_mean(cent, name='loss')

        metrics = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred, name='accuracy')}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        lr = tf.train.exponential_decay(learning_rate=opts.learning_rate * hvd.size(),
                                        global_step=tf.train.get_global_step(),
                                        decay_steps=1,
                                        decay_rate=1.0 - opts.learning_decay)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn


def main(opts):
    """Main"""
    hvd.init()
    tf.logging.info('Horovod size: {}'.format(hvd.size()))
    tf.logging.info('Horovod local rank: {}'.format(hvd.local_rank()))
    tf.logging.info('Horovod rank: {}'.format(hvd.rank()))

    # Create an estimator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    runconfig = tf.estimator.RunConfig(
        model_dir=opts.log_dir,
        save_summary_steps=opts.save_summary_steps,
        save_checkpoints_steps=opts.ckpt_steps,
        keep_checkpoint_max=opts.max_ckpts,
        log_step_count_steps=opts.log_step_count_steps,
        session_config=config)

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(opts),
        config=runconfig)

    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    training_hooks = [bcast_hook]

    # Create input fn
    train_input_fn = get_input_fn(opts, is_train=True)
    eval_input_fn = get_input_fn(opts, is_train=False)

    for _ in range(opts.max_steps//opts.eval_steps):
        estimator.train(
            input_fn=train_input_fn,
            steps=opts.eval_steps,
            hooks=training_hooks)

        estimator.evaluate(
            input_fn=eval_input_fn)


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
