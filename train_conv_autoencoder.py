# coding: utf-8

import numpy as np

from config import *
from model.get_autoencoder import get_encoder, get_conv_encoder
import datetime

SENTENCE = np.load(os.path.join(data_dir, 'SENTENCE.npy'))
X = np.load(os.path.join(data_dir, 'X.npy'))
Y = np.load(os.path.join(data_dir, 'Y.npy'))
Xconcat = np.load(os.path.join(data_dir, 'Xconcat.npy'))

# print 'SENTENCE.shape: {}'.format(SENTENCE.shape)
# print 'X.shape: {}'.format(X.shape)
# print 'Y.shape: {}'.format(Y.shape)
# print 'Xconcat.shape: {}'.format(Xconcat.shape)
# Xconcat = np.reshape(Xconcat, (-1, 1, Xconcat.shape[1]))
# print 'Xconcat.shape: {}'.format(Xconcat.shape)

cur_epoch = 0
epochs = 50000
batch_size = 500
epochs_per_iter = 20
check_point_every = 20
assert check_point_every % epochs_per_iter == 0

t_frames = X.shape[1]
print 't_frames: {}'.format(t_frames)

Xtrain = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
Ytrain = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
print 'Xtrain.shape: {}'.format(Xtrain.shape)
print 'Ytrain.shape: {}'.format(Ytrain.shape)

autoencoder, encoder, decoder = get_conv_encoder(batch_size, t_frames)
autoencoder.summary()

enc = encoder.predict(Xtrain)
print 'enc.shape: {}'.format(enc.shape)
dec = decoder.predict(enc)
print 'dec.shape: {}'.format(dec.shape)

# exit()
#
history_list = []
while cur_epoch < epochs:
    if cur_epoch and cur_epoch % check_point_every == 0:
        loss = autoencoder.evaluate(Xtrain, Ytrain, batch_size=batch_size)
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

    history = autoencoder.fit(Xtrain, Ytrain,
                              batch_size=batch_size,
                              nb_epoch=epochs_per_iter,
                              verbose=0,
                              validation_split=0.0,
                              shuffle=1)
    history_list.append(history)
    cur_epoch += epochs_per_iter

autoencoder.save_weights(weights_file)
print 'save: {}'.format(weights_file)
