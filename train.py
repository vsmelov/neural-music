# coding: utf-8

import numpy as np

from config import *
from model.get_model import get_model

X = np.load(os.path.join(data_dir, 'X.npy'))
# X = np.power(X*2, 2)
Y = np.load(os.path.join(data_dir, 'Y.npy'))
# Y = np.power(Y*2, 2)
shift_mX = np.loadtxt(os.path.join(data_dir, 'shift_mX.txt'))
var_mX = np.loadtxt(os.path.join(data_dir, 'var_mX.txt'))

print 'X.shape: {}'.format(X.shape)

print('NP = {}'.format(NP))
assert X.shape[2] == NP

model = get_model(NP)
print('model.summary() = {}'.format(model.summary()))
print('model.get_config() = {}'.format(model.get_config()))

# epochs = 500
epochs = 10000
epochs_per_iter = 1
cur_epoch = 0

# batch_size = 22
# batch_size = 35
# batch_size = 13
# batch_size = 6
# batch_size = 52
# batch_size = 101

while cur_epoch < epochs:
    # batch_size_list = [52, 35, 52, 52, 22, 52, 52, 52, 52, 52, 52, 52, 52]
    # batch_size = batch_size_list[int(0.1 * cur_epoch) % len(batch_size_list)]
    batch_size = 17

    print 'cur_epoch: {} / {}'.format(cur_epoch, epochs)
    print 'batch_size: {}'.format(batch_size)

    history = model.fit(X, Y,
                        batch_size=batch_size,
                        nb_epoch=epochs_per_iter,
                        verbose=1,
                        validation_split=0.0,
                        shuffle=1)
    cur_epoch += epochs_per_iter
    if cur_epoch and cur_epoch % 10 == 0:
        f = weights_file + '-' + str(cur_epoch)
        model.save_weights(f)
        print 'save: {}'.format(f)

model.save_weights(weights_file)
print 'save: {}'.format(weights_file)
