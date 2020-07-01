import numpy as np
from numpy import linalg as la
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv2D,
    Flatten, LSTM, concatenate)
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss

from main.preprocess import get_segmented_spectrograms
from main.constants import *


class SpectrogramConvolution:
    def __init__(self, _shape):
        self.conv1 = Conv2D(filters=1, input_shape=_shape,
                            kernel_size=(3, 3), strides=1,
                            activation=relu, padding='same')
        self.conv2 = Conv2D(filters=1, kernel_size=(3, 3),
                            strides=1, activation=relu, padding='same')
        self.flat = Flatten()

    def convolute(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        return self.flat(inputs)


def get_convolution_vectors():
    anch_specs, pos_specs, neg_specs, labels = get_segmented_spectrograms(
        audio_direct, ext, num_speakers,
        triplet_len, alpha, rate
    )

    _input_shape = anch_specs.shape[2:]
    spec_conv = SpectrogramConvolution(_input_shape)

    conv_anchors = []
    conv_pos = []
    conv_neg = []

    for anch, pos, neg in zip(anch_specs, pos_specs, neg_specs):
        conv = spec_conv.convolute(anch)
        conv_anchors.append(conv.numpy())

        conv = spec_conv.convolute(pos)
        conv_pos.append(conv.numpy())

        conv = spec_conv.convolute(neg)
        conv_neg.append(conv.numpy())

    conv_anchors = np.asarray(conv_anchors)
    conv_pos = np.asarray(conv_pos)
    conv_neg = np.asarray(conv_neg)
    return conv_anchors, conv_pos, conv_neg, labels


class TripletLoss(Loss):
    def __init__(self, margin):
        self.margin = margin
        super(TripletLoss, self).__init__()

    def call(self, y_true, y_pred):
        assert y_pred.shape[0] == 3
        anchor, positive, negative = tf.unstack(y_pred, axis=0)
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        base_loss = pos_dist - neg_dist + self.margin
        return K.mean(K.maximum(base_loss, 0.0), axis=0)


def base_rnn(_shape, out_units):
    inp = Input(shape=_shape, name='input')
    lstm1 = LSTM(3, return_sequences=True,
                 name='seq2seq')(inp)
    lstm2 = LSTM(out_units, name='seq2one')(lstm1)
    out = Flatten(name='flattened')(lstm2)
    model = Model(inputs=inp, outputs=out)
    return model


def build_siam(input_shape, out_units,
               optimizer, loss):
    anchor_input = Input(shape=input_shape, name='anchor')
    positive_input = Input(shape=input_shape, name='positive')
    negative_input = Input(shape=input_shape, name='negative')

    rnn = base_rnn(input_shape, out_units)
    anch_out = rnn(anchor_input)
    pos_out = rnn(positive_input)
    neg_out = rnn(negative_input)

    out1 = tf.expand_dims(anch_out, axis=0)
    out2 = tf.expand_dims(pos_out, axis=0)
    out3 = tf.expand_dims(neg_out, axis=0)
    output = concatenate([out1, out2, out3], axis=0)

    model = Model(inputs=[
        anchor_input, positive_input, negative_input
    ], outputs=output)

    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_siamese():
    conv_anchors, conv_pos, conv_neg, labels = get_convolution_vectors()

    siam_input = conv_anchors.shape[1:]
    optim = SGD()
    triplet = TripletLoss(margin=margin)

    siam = build_siam(input_shape=siam_input,
                      out_units=out_units, optimizer=optim,
                      loss=triplet)

    train_data = [conv_anchors, conv_pos, conv_neg]

    # Dummy 'true' labels to provide gradients
    dummy_labels = np.zeros((len(train_data[0]),))

    history = siam.fit(train_data, y=dummy_labels,
                       epochs=num_epochs, batch_size=batch_size)
    return history, siam, conv_anchors, labels


def create_database(conv_anchors, targets, model):
    user_labels = targets.to_numpy()
    target_convs = np.asarray([
        conv_anchors[i] for i in range(0, len(conv_anchors), triplet_len)
    ])

    dummy_x = np.empty(target_convs.shape)
    verify_data = [target_convs, dummy_x, dummy_x]

    verify_vects = model(verify_data)[0]

    database = {label: verify_vects[i]
                for i, label in enumerate(user_labels)}
    return database


def identity_verification(audio_conv, db: dict, model, dist_thresh):
    assert len(audio_conv.shape) == 2
    result_dist = 1
    verified = False
    identity = None

    audio_conv = tf.expand_dims(audio_conv, axis=0)
    dummy_vect = np.empty(audio_conv.shape)
    datas = [audio_conv, dummy_vect, dummy_vect]

    encoding = model(datas)

    for label, db_enc in db.items():
        dist = la.norm(db_enc - encoding)
        if dist < dist_thresh and dist < result_dist:
            result_dist = dist
            identity = label
            verified = True

    return result_dist, verified, identity

