# coding: utf-8

import numpy as np

from config import *
from model.get_autoencoder import get_encoder, get_conv_encoder
import datetime

SENTENCE = np.load(os.path.join(data_dir, 'SENTENCE.npy'))
X = np.load(os.path.join(data_dir, 'X.npy'))
Y = np.load(os.path.join(data_dir, 'Y.npy'))

print 'SENTENCE.shape: {}'.format(SENTENCE.shape)
print 'X.shape: {}'.format(X.shape)
print 'Y.shape: {}'.format(Y.shape)

print('NP = {}'.format(NP))
assert X.shape[2] == NP

cur_epoch = 0
epochs = 50000
batch_size = 5000
epochs_per_iter = 200
check_point_every = 200
assert check_point_every % epochs_per_iter == 0

autoencoder, encoder, decoder = get_conv_encoder(SENTENCE.shape[1], load_wights=0, batch_size=batch_size)
# autoencoder, encoder, decoder = get_encoder(SENTENCE.shape[1])

print('autoencoder.summary() = {}'.format(autoencoder.summary()))
# print('encoder.get_config() = {}'.format(encoder.get_config()))
# print('decoder.get_config() = {}'.format(decoder.get_config()))


history_list = []
while cur_epoch < epochs:
    if cur_epoch % check_point_every == 0:
        loss = autoencoder.evaluate(SENTENCE, SENTENCE, batch_size=batch_size)
        f = '{}-{}-{}'.format(
            weights_file,
            cur_epoch,
            '{0:.6f}'.format(loss)
            # history.history['loss']
        )
        autoencoder.save_weights(f)
        print '{}: epoch {} loss {} save: {}'.format(
            datetime.datetime.now(),
            cur_epoch,
            '{0:.6f}'.format(loss),
            f
        )

        if history_list:
            # print [history.history['loss'] for history in history_list]
            sum_loss = sum([sum(history.history['loss'])
                            for history in history_list
                            ])
            n_history = sum([len(history.history['loss'])
                             for history in history_list
                             ])
            avg_loss = sum_loss / n_history
            print 'avg_loss: {}'.format(avg_loss)
            history_list = []

    print 'cur_epoch: {} / {}'.format(cur_epoch, epochs)
    print 'batch_size: {}'.format(batch_size)

    history = autoencoder.fit(SENTENCE, SENTENCE,
                              batch_size=batch_size,
                              nb_epoch=epochs_per_iter,
                              verbose=0,
                              validation_split=0.0,
                              shuffle=1)
    history_list.append(history)
    cur_epoch += epochs_per_iter

autoencoder.save_weights(weights_file)
print 'save: {}'.format(weights_file)
