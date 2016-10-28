# coding: utf-8
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from config import *


def get_model(load_wights=True, stateful=False):
    model = Sequential()
    if stateful:
        model.add(LSTM(3*NP,
                       input_dim=NP,
                       return_sequences=True,
                       stateful=stateful,
                       batch_input_shape=(1, 1, NP)))
    else:
        model.add(LSTM(3*NP,
                       input_dim=NP,
                       return_sequences=True,
                       stateful=stateful))
    model.add(LSTM(4*NP,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(4 * NP,
                   return_sequences=True,
                   stateful=stateful))
    model.add(LSTM(3*NP,
                   return_sequences=True,
                   stateful=stateful))
    model.add(TimeDistributed(Dense(NP)))

    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.compile(loss='mean_squared_error', optimizer='adam')
    if load_wights and os.path.exists(weights_file):
        print 'Load from {}'.format(weights_file)
        model.load_weights(weights_file)
    return model
