# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings("ignore")

def process_feats(data, dense_feats, sparse_feats):
    """
    (1)对于数值型特征，用0填充，然后进行log变换
    (2)对于类别型特征，用-1填充，然后进行类别编码
    """
    d = data.copy()
    d[dense_feats]  = d[dense_feats].fillna(0.0)
    d[sparse_feats] = d[sparse_feats].fillna("-1")
    for col in d.columns:
        if col in dense_feats:
            d[col] = d[col].apply(lambda x: np.log(x+1) if x>-1 else -1)
        elif col in sparse_feats:
            d[col] = LabelEncoder().fit_transform(d[col])
    return d

def init_model(data, dense_feats, sparse_feats):
    # ------------- linear -----------------
    # 对于数值型特征，构造其模型输入与加权求和的代码
    # 注释中 ? 表示输入数据的 batch_size
    dense_inputs = []
    for f in dense_feats:
        _input = Input([1], name=f)
        dense_inputs.append(_input)
    # 将输入拼接到一起，方便连接到 Dense 层
    concat_dense_inputs = Concatenate(axis=1)(dense_inputs)     # ?, 13
    # 然后连上输出为1个单元的全连接层，表示对 dense 变量的加权求和
    fst_order_dense_layer = Dense(1)(concat_dense_inputs)       # ?, 1

    # 对于离散型特征，需要进行 embedding 变换
    # 例如原一个离散特征，可能枚举值有n个，则embedding为n*1
    embedding_size = 1
    sparse_inputs = []
    for f in sparse_feats:
        _input = Input([1], name=f)
        sparse_inputs.append(_input)
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_feats[i]
        voc_size = data[f].nunique()
        # 使用 l2 正则化防止过拟合
        reg = tf.keras.regularizers.l2(0.5)
        _embed = Embedding(voc_size, embedding_size, embeddings_regularizer=reg)(_input)
        # 由于 Embedding 的结果是二维的，
        # 因此需要在 Embedding 之后加入 Dense 层，则需要先连接上 Flatten 层
        _embed = Flatten()(_embed)
        sparse_1d_embed.append(_embed)
    # 对每个 embedding lookup 的结果 wi 求和
    fst_order_sparse_layer = Add()(sparse_1d_embed)

    # 将数值型特征与离散型特征的结果求和
    liner_part = Add()([fst_order_dense_layer, fst_order_sparse_layer])

    # ------------- fm -----------------
    # 仅对离散特征进行交叉
    # embedding_size 
    embedding_size = 8
    # 只考虑 sparse 的二阶交叉
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_feats[i]
        voc_size = data[f].nunique()
        reg = tf.keras.regularizers.l2(0.7)
        _embed = Embedding(voc_size, embedding_size, embeddings_regularizer=reg)(_input)
        sparse_kd_embed.append(_embed)
    # 通过拆隐向量求和公式可得到 和的平方 - 平方的和
    # (1) 将所有 sparse 特征 (?, 1, k) 的embedding拼接起来
    #     得到 (?, n, k) 的矩阵，其中 n 为特征数，k为embedding_size
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)   # ?, n, k
    # (2) 先求和再平方
    sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(concat_sparse_kd_embed)   # ?, k
    square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])
    # (3) 先平方再求和
    square_kd_embed = Multiply()([concat_sparse_kd_embed,concat_sparse_kd_embed])   # ?, n, k
    sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)   # ?, k
    # (4) 相减除以2
    sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])    # ?, k
    sub = Lambda(lambda x: x*0.5)(sub)  # ?, k
    snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(sub) # ?, 1

    # ------------- dnn -----------------
    # 注意 dnn 部分是对 fm 中已embedding的隐向量学习
    flatten_sparse_embed = Flatten()(concat_sparse_kd_embed)    # ?, n*k
    fc_layer = Dropout(0.5)(Dense(128, activation="relu")(flatten_sparse_embed))    # ?, 128
    fc_layer = Dropout(0.5)(Dense(128, activation="relu")(fc_layer))    # ?, 128
    fc_layer = Dropout(0.5)(Dense(128, activation="relu")(fc_layer))    # ?, 128
    fc_layer_output = Dense(1)(fc_layer)    # ?, 1

    # ------------- deepfm -----------------
    output_layer = Add()([liner_part, snd_order_sparse_layer, fc_layer_output])
    output_layer = Activation("sigmoid")(output_layer)

    # 模型编译
    model = Model(dense_inputs+sparse_inputs, output_layer)
    return model

if __name__ == "__main__":
    train = pd.read_csv("/Users/didi/Desktop/DeepFM-criteo/data/train_1m.txt", sep="\t")
    cols  = train.columns

    dense_feats  = [f for f in cols if f[0] == "I"]
    sparse_feats = [f for f in cols if f[0] == "C"]

    data = process_feats(train, dense_feats, sparse_feats)

    train_data = data.loc[:800000-1]    # 等价于 data.iloc[:800000]
    valid_data = data.loc[800000:]

    train_dense_x  = [train_data[f].values for f in dense_feats]
    train_sparse_x = [train_data[f].values for f in sparse_feats]
    train_label    = [train_data["Label"].values]

    val_dense_x  = [valid_data[f].values for f in dense_feats]
    val_sparse_x = [valid_data[f].values for f in sparse_feats]
    val_label    = [valid_data["Label"].values]

    model = init_model(data, dense_feats, sparse_feats)
    # tf.keras.utils.plot_model(model, to_file="deepfm-criteo.png",
    #                       show_shapes=True, show_dtype=True,
    #                      show_layer_names=True)
    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["binary_crossentropy", tf.keras.metrics.AUC(name="auc")])
    model.fit(train_dense_x+train_sparse_x, train_label,
              epochs=5, batch_size=256,
              validation_data=(val_dense_x+val_sparse_x, val_label)
              )
    