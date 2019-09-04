
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


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    
    dfm = DeepFM(**dfm_params)
    dfm.fit(Xi_train, Xv_train, y_train, )

    y_test_meta[:,0] = dfm.predict(Xi_test, Xv_test)

 

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
 

    return y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"index": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 10,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": config.RANDOM_SEED
}
y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# ------------------ FM Model ------------------


