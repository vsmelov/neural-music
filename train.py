# coding: utf-8

import numpy as np

from config import *
from model.get_model import get_model
from model.get_autoencoder import get_encoder
import datetime

X = np.load(os.path.join(data_dir, 'X10.npy'))
Y = np.load(os.path.join(data_dir, 'Y10.npy'))

print 'X.shape: {}'.format(X.shape)

# autoencoder, encoder, decoder = get_encoder(10)
#
# X_new = np.zeros((X.shape[0], X.shape[1]/10, n_features))
# Y_new = np.zeros((X.shape[0], X.shape[1]/10, n_features))
#
# for i in range(X.shape[0]):
#     for j in range(9):
#         x = X[i, j*10:(j+1)*10, :]
#         y = Y[i, j*10:(j+1)*10, :]
#
#         x = np.reshape(x, (1, 10, 128))
#         y = np.reshape(y, (1, 10, 128))
#
#         x = encoder.predict(x)
#         y = encoder.predict(y)
#
#         X_new[i, j, :] = np.reshape(x, (1, 1, n_features))
#         Y_new[i, j, :] = np.reshape(y, (1, 1, n_features))
#
# X = X_new
# Y = Y_new
#
# np.save(os.path.join(data_dir, 'X10.npy'), X)
# np.save(os.path.join(data_dir, 'Y10.npy'), Y)
#
# exit()
#
# print 'X.shape: {}'.format(X.shape)

# print('NP = {}'.format(NP))
# assert X.shape[2] == NP

model = get_model(load_wights=0)
print('model.summary() = {}'.format(model.summary()))
print('model.get_config() = {}'.format(model.get_config()))

epochs = 50000
epochs_per_iter = 1
cur_epoch = 0

history_list = []
while cur_epoch < epochs:
    batch_size = 750
    epochs_per_iter = 50
    check_point_every = 50
    assert check_point_every % epochs_per_iter == 0

    if cur_epoch % check_point_every == 0:
        loss = model.evaluate(X, Y, batch_size=batch_size)
        f = '{}-{}-{}'.format(
            weights_file,
            cur_epoch,
            '{0:.6f}'.format(loss)
            # history.history['loss']
        )
        model.save_weights(f)
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

    history = model.fit(X, Y,
                        batch_size=batch_size,
                        nb_epoch=epochs_per_iter,
                        verbose=0,
                        validation_split=0.0,
                        shuffle=1)
    history_list.append(history)
    cur_epoch += epochs_per_iter

model.save_weights(weights_file)
print 'save: {}'.format(weights_file)
