#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import tensorflow as tf

from nlp.nn import initializer, regularizer


def CNN_encode(seqs, filter_size=3, dilation=1,
               num_filters=None, direction='forward', act_fn=tf.nn.relu,
               scope=None,
               reuse=None, **kwargs):
    input_dim = seqs.get_shape().as_list()[-1]
    num_filters = num_filters if num_filters else input_dim

    # add causality: shift the whole seq to the right
    padding = (filter_size - 1) * dilation
    if direction == 'forward':
        pad_seqs = tf.pad(seqs, [[0, 0], [padding, 0], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'backward':
        pad_seqs = tf.pad(seqs, [[0, 0], [0, padding], [0, 0]])
        padding_scheme = 'VALID'
    elif direction == 'none':
        pad_seqs = seqs  # no padding, must set to SAME so that we have same length
        padding_scheme = 'SAME'
    else:
        raise NotImplementedError

    with tf.variable_scope(scope or 'causal_conv_%s_%s' % (filter_size, direction), reuse=reuse):
        return tf.layers.conv1d(
            pad_seqs,
            num_filters,
            filter_size,
            activation=act_fn,
            padding=padding_scheme,
            dilation_rate=dilation,
            kernel_initializer=initializer,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=regularizer)
