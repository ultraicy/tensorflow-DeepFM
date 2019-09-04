#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:15:37 2019

@author: nickwu
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
from DeepFM import DeepFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)


    dfTrain = dfTrain
    dfTest = dfTest

    cols = [c for c in dfTrain.columns if c not in ["index", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["index"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices,

def get_batch( Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index+1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

def _run_base_model_dfm(dfTrain, dfTest, ):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest, has_label=True)
    ckpt = tf.train.get_checkpoint_state( './model/checkpoint')
    saver = tf.train.import_meta_graph('./model/my-model-9.meta')   
    batch_index = 0
    batch_size = 1024
    dummy_y = [1] * len(Xi_test)
    Xi_batch, Xv_batch, y_batch = get_batch(Xi_test, Xv_test, dummy_y, batch_size, batch_index)


    y_pred = None
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model'))
        graph = tf.get_default_graph()   
        feat_index = graph.get_tensor_by_name('feat_index:0')
        feat_value = graph.get_tensor_by_name('feat_value:0')
        dropout_keep_fm = graph.get_tensor_by_name('dropout_keep_fm:0')
        dropout_keep_deep = graph.get_tensor_by_name('dropout_keep_deep:0')
        train_phase = graph.get_tensor_by_name('train_phase:0')
        label = graph.get_tensor_by_name('label:0')
        # 加载模型中的操作节点	
        feed_dict = {feat_index: Xi_batch,
                         feat_value: Xv_batch,
                         label: y_batch,
                         dropout_keep_fm: [1.0] * 2,
                         dropout_keep_deep: [1.0] * 3,
                         train_phase: False}
        y = tf.get_collection('pred_network')[0]
        y_pred = sess.run(y,feed_dict = feed_dict)
    return y_pred

dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()       
y11 = _run_base_model_dfm(dfTrain, dfTest, )