#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import tensorflow as tf

from nlp.encode_blocks import CNN_encode
from nlp.nn import linear_logit, dropout_res_layernorm


def AttentiveCNN_match(context, query, context_mask, query_mask,
                       scope='AttentiveCNN_Block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        cnn_wo_att = CNN_encode(context, filter_size=3, direction='none', act_fn=None)
        att_context, _ = Attentive_match(context, query, context_mask, query_mask)
        cnn_att = CNN_encode(att_context, filter_size=1, direction='none', act_fn=None)
        output = tf.nn.tanh(cnn_wo_att + cnn_att)
        return dropout_res_layernorm(context, output, **kwargs)


def Attentive_match(context, query, context_mask, query_mask,
                    score_func='dot', causality=False,
                    scope='attention_match_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, context_length, num_units = context.get_shape().as_list()
        _, query_length, _ = query.get_shape().as_list()
        if score_func == 'dot':
            score = tf.matmul(context, query, transpose_b=True)
        elif score_func == 'bilinear':
            score = tf.matmul(linear_logit(context, num_units, scope='context_x_We'), query, transpose_b=True)
        elif score_func == 'scaled':
            score = tf.matmul(linear_logit(context, num_units, scope='context_x_We'), query, transpose_b=True) / \
                    (num_units ** 0.5)
        elif score_func == 'additive':
            score = tf.squeeze(linear_logit(
                tf.tanh(tf.tile(tf.expand_dims(linear_logit(context, num_units, scope='context_x_We'), axis=2),
                                [1, 1, query_length, 1]) +
                        tf.tile(tf.expand_dims(linear_logit(query, num_units, scope='query_x_We'), axis=1),
                                [1, context_length, 1, 1])), 1, scope='x_ve'), axis=3)
        else:
            raise NotImplementedError

        mask = tf.matmul(tf.expand_dims(context_mask, -1), tf.expand_dims(query_mask, -1), transpose_b=True)
        paddings = tf.ones_like(mask) * (-2 ** 32 + 1)
        masked_score = tf.where(tf.equal(mask, 0), paddings, score)  # B, Lc, Lq

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(masked_score[0, :, :])  # (Lc, Lq)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (Lc, Lq)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(masked_score)[0], 1, 1])  # B, Lc, Lq

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            masked_score = tf.where(tf.equal(masks, 0), paddings, masked_score)  # B, Lc, Lq

        query2context_score = tf.nn.softmax(masked_score, axis=2) * mask  # B, Lc, Lq
        query2context_attention = tf.matmul(query2context_score, query)  # B, Lc, D

        context2query_score = tf.nn.softmax(masked_score, axis=1) * mask  # B, Lc, Lq
        context2query_attention = tf.matmul(context2query_score, context, transpose_a=True)  # B, Lq, D

        return (query2context_attention,  # B, Lc, D
                context2query_attention)  # B, Lq, D
