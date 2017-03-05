# coding: utf-8
from keras.layers import Input, LSTM, RepeatVector, MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D, Cropping2D
from keras.layers.core import Dense, Dropout, Flatten, Reshape, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.regularizers import activity_l1l2, activity_l1, activity_l2, l1, l2, l1l2
from math import ceil
from config import *


def get_encoder(timesteps, f_weights, load_wights=True):
    inputs = Input(shape=(timesteps, Nmel))

    encoded = Flatten()(inputs)  # reshape to timesteps*NP
    encoded = Dense(Nmel*6, name='encoded1')(encoded)
    encoded = Dense(Nmel*3, name='encoded2')(encoded)
    encoded = Dense(n_features, name='encoded3')(encoded)

    decoded = Dense(Nmel*3, name='decoded1')(encoded)
    decoded = Dense(Nmel*6, name='decoded2')(decoded)
    decoded = Dense(Nmel*10, name='decoded3')(decoded)
    decoded = Reshape((timesteps, Nmel))(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)


    inputs_encoded = Input(shape=(n_features,))
    xxdecoded = Dense(Nmel*3, name='decoded1')(inputs_encoded)
    xxdecoded = Dense(Nmel*6, name='decoded2')(xxdecoded)
    xxdecoded = Dense(Nmel*10, name='decoded3')(xxdecoded)
    xxdecoded = Reshape((timesteps, Nmel))(xxdecoded)
    decoder = Model(inputs_encoded, xxdecoded)


    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    encoder.compile(loss='mean_squared_error', optimizer='adam')
    decoder.compile(loss='mean_squared_error', optimizer='adam')

    if load_wights and os.path.exists(f_weights):
        print 'Load from {}'.format(f_weights)
        autoencoder.load_weights(f_weights, by_name=True)
        encoder.load_weights(f_weights, by_name=True)
        decoder.load_weights(f_weights, by_name=True)

    return autoencoder, encoder, decoder



def get_conv_encoder(batch_size, t_frames, load_wights=True, f_weights=None):
    if f_weights is None:
        f_weights = weights_conv_file

    input_shape = (t_frames, Nin, 1)
    inputs = Input(shape=input_shape)
    encoded = inputs

    # encoded = GaussianDropout(0.001)(encoded)
    # encoded = GaussianNoise(0.05)(encoded)

    reg_intensity = 1e-9
    l1_intensity = 0.8
    l1_k = l1_intensity * reg_intensity
    l2_k = (1 - l1_intensity) * reg_intensity
    wb_reg_intensity = 1e-6
    n_filters = 3
    # subsample = (1, 1)
    core = (5, 3)

    encoded = Convolution2D(n_filters, core[0], core[1],
                         border_mode='same',
                         name='encoder1',
                         # subsample=subsample,
                         # activation='relu',
    )(encoded)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
    encoded = Convolution2D(n_filters**2, core[0], core[1],
                            border_mode='same',
                            name='encoder2',
                            # subsample=subsample,
                            # activation='relu'
                            )(encoded)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)
    encoded = Convolution2D(n_filters**3, core[0], core[1],
                            border_mode='same',
                            name='encoder3',
                            # subsample=subsample,
                            # activation='relu'
                            )(encoded)
    encoded = MaxPooling2D((2, 2), border_mode='same')(encoded)

    def get_decoded(inp):
        decoded = inp
        decoded = Convolution2D(n_filters ** 2, core[0], core[1],
                              border_mode='same',
                              name='decoder1',
                              )(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Convolution2D(n_filters ** 1, core[0], core[1],
                                border_mode='same',
                                name='decoder2',
                                )(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Convolution2D(1, core[0], core[1],
                                border_mode='same',
                                name='decoder3',
                                )(decoded)
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Cropping2D(((0, 0), (0, 1032-1025)))(decoded)
        return decoded

    encoder = Model(inputs, encoded)
    autoencoder = Model(inputs, get_decoded(encoded))

    encoded_input = Input(shape=(8, 129, 27))
    decoder = Model(encoded_input, get_decoded(encoded_input))
    # decoder = None

    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    encoder.compile(loss='mean_squared_error', optimizer='adam')
    if decoder:
        decoder.compile(loss='mean_squared_error', optimizer='adam')

    if load_wights and os.path.exists(f_weights):
        print 'Load from {}'.format(f_weights)
        autoencoder.load_weights(f_weights, by_name=True)
        encoder.load_weights(f_weights, by_name=True)
        if decoder:
            decoder.load_weights(f_weights, by_name=True)

    return autoencoder, encoder, decoder




# def get_conv_encoder(load_wights=True, f_weights=None):
#     if f_weights is None:
#         f_weights = weights_conv_file
#
#     inputs = Input(shape=(None, Nin))
#     encoded = inputs
#
#     # encoded = GaussianDropout(0.001)(encoded)
#     encoded = GaussianNoise(0.05)(encoded)
#
#     reg_intensity = 1e-9
#     l1_intensity = 0.8
#     l1_k = l1_intensity * reg_intensity
#     l2_k = (1 - l1_intensity) * reg_intensity
#
#     encoded = TimeDistributed(
#         Dense(2*n_features,
#               # activation='relu',
#               # activity_regularizer=activity_l1(l1_k),
#               )
#         ,
#         name='encoder1',
#     )(encoded)
#
#     encoded = TimeDistributed(
#         Dense(n_features,
#               activation='relu',
#               activity_regularizer=activity_l1(l1_k),
#               )
#         ,
#         name='encoder2',
#     )(encoded)
#
#     wb_reg_intensity = 1e-6
#     decoded = TimeDistributed(
#         Dense(2*n_features,
#               W_regularizer=l1(wb_reg_intensity),
#               b_regularizer=l1(wb_reg_intensity)
#         ),
#         name='decoder1'
#     )(encoded)
#     decoded = TimeDistributed(
#         Dense(Nin,
#               W_regularizer=l1(wb_reg_intensity),
#               b_regularizer=l1(wb_reg_intensity)
#               ),
#         name='decoder2'
#     )(decoded)
#     autoencoder = Model(inputs, decoded)
#     encoder = Model(inputs, encoded)
#
#
#
#     encoding = Input(shape=(None, n_features))
#     decoder = TimeDistributed(Dense(2*n_features), name='decoder1')(encoding)
#     decoder = TimeDistributed(Dense(Nin), name='decoder2')(decoder)
#     decoder = Model(encoding, decoder)
#
#
#
#     autoencoder.compile(loss='mean_squared_error', optimizer='adam')
#     encoder.compile(loss='mean_squared_error', optimizer='adam')
#     if decoder:
#         decoder.compile(loss='mean_squared_error', optimizer='adam')
#
#     if load_wights and os.path.exists(f_weights):
#         print 'Load from {}'.format(f_weights)
#         autoencoder.load_weights(f_weights, by_name=True)
#         encoder.load_weights(f_weights, by_name=True)
#         if decoder:
#             decoder.load_weights(f_weights, by_name=True)
#
#     return autoencoder, encoder, decoder




# def get_conv_encoder(load_wights=True, f_weights=None):
#     if f_weights is None:
#         f_weights = weights_conv_file
#
#     inputs = Input(shape=(None, Nin))
#     encoded = inputs
#
#     # encoded = GaussianDropout(0.001)(encoded)
#     encoded = GaussianNoise(0.05)(encoded)
#
#     reg_intensity = 1e-9
#     l1_intensity = 0.8
#     l1_k = l1_intensity * reg_intensity
#     l2_k = (1 - l1_intensity) * reg_intensity
#     encoded = TimeDistributed(
#         Dense(n_features,
#               activation='relu',
#               # activity_regularizer=activity_l1(l1_k),
#               # W_regularizer=l1(1),
#               # b_regularizer=l1(1)
#               )
#         ,
#         name='encoder1',
#     )(encoded)
#
#     wb_reg_intensity = 1e-6
#     decoded = TimeDistributed(
#         Dense(Nin,
#               W_regularizer=l1(wb_reg_intensity),
#               b_regularizer=l1(wb_reg_intensity)
#         ),
#         name='decoder1'
#     )(encoded)
#     autoencoder = Model(inputs, decoded)
#     encoder = Model(inputs, encoded)
#
#
#
#     encoding = Input(shape=(None, n_features))
#     decoder = TimeDistributed(Dense(Nin), name='decoder1')(encoding)
#     decoder = Model(encoding, decoder)
#
#     autoencoder.compile(loss='mean_squared_error', optimizer='adam')
#     encoder.compile(loss='mean_squared_error', optimizer='adam')
#     if decoder:
#         decoder.compile(loss='mean_squared_error', optimizer='adam')
#
#     if load_wights and os.path.exists(f_weights):
#         print 'Load from {}'.format(f_weights)
#         autoencoder.load_weights(f_weights, by_name=True)
#         encoder.load_weights(f_weights, by_name=True)
#         if decoder:
#             decoder.load_weights(f_weights, by_name=True)
#
#     return autoencoder, encoder, decoder
