# coding: utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianNoise, GaussianDropout
from config import *

# def get_model(load_wights=True, stateful=False):
#     model = Sequential()
#     if stateful:
#         model.add(LSTM(2*NP,
#                        input_dim=NP,
#                        return_sequences=True,
#                        stateful=stateful,
#                        batch_input_shape=(1, 1, NP)))
#     else:
#         model.add(LSTM(2*NP,
#                        input_dim=NP,
#                        return_sequences=True,
#                        stateful=stateful))
#     # model.add(Dropout(0.3))
#
#     model.add(LSTM(3 * NP,
#                    return_sequences=True,
#                    stateful=stateful))
#
#     model.add(LSTM(3*NP,
#                    return_sequences=True,
#                    stateful=stateful))
#
#     model.add(LSTM(2 * NP,
#                    input_dim=NP,
#                    return_sequences=True,
#                    stateful=stateful))
#
#     # model.add(TimeDistributed(Dense(2 * NP)))
#     model.add(TimeDistributed(Dense(NP)))
#
#     # model.compile(loss='mean_squared_error', optimizer='rmsprop')
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     if load_wights and os.path.exists(weights_file):
#         print 'Load from {}'.format(weights_file)
#         model.load_weights(weights_file)
#     return model



def get_model(load_wights=True, stateful=False, f_weights=None):
    if f_weights is None:
        f_weights = weights_file

    model = Sequential()
    if stateful:
        model.add(LSTM(2*n_features,
                       input_dim=n_features,
                       return_sequences=True,
                       stateful=stateful,
                       batch_input_shape=(1, 1, n_features)))
    else:
        model.add(LSTM(2*n_features,
                       input_dim=n_features,
                       return_sequences=True,
                       stateful=stateful))
    # model.add(Dropout(0.3))

    model.add(LSTM(3 * n_features,
                   return_sequences=True,
                   stateful=stateful))

    model.add(LSTM(3*n_features,
                   return_sequences=True,
                   stateful=stateful))

    model.add(LSTM(2 * n_features,
                   input_dim=n_features,
                   return_sequences=True,
                   stateful=stateful))

    # model.add(TimeDistributed(Dense(2 * NP)))
    model.add(TimeDistributed(Dense(n_features)))

    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.compile(loss='mean_squared_error', optimizer='adam')
    if load_wights and os.path.exists(f_weights):
        print 'Load from {}'.format(f_weights)
        model.load_weights(f_weights)
    return model
