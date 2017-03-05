# coding: utf-8
import os
from scipy.signal import get_window
import tools.stft as STFT
import tools.utilFunctions as UF
import numpy as np
from config import *


if __name__ == '__main__':
    fname = '/neural-music/tmp/cepstra.txt'
    data = np.loadtxt(fname, delimiter=',')

    data = data[::1, :]
    mean_data = np.mean(np.mean(data))
    var_data = np.sqrt(
        np.mean(np.mean(
            np.abs(data - mean_data) ** 2)))
    data = (data - mean_data) / var_data

    sentences = []
    for i in range(0, data.shape[0] - max_sentence_len, sentences_step):
        sentences.append(data[i:i + max_sentence_len, :])
    print('len(sequences):', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]), dtype=np.float)
    Y = np.zeros((len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]), dtype=np.float)
    for i, sentence in enumerate(sentences):
        X[i, :, :] = sentence[skip_first:-1, :]
        Y[i, :, :] = sentence[skip_first+1:, :]

    print('X.shape: {}'.format(X.shape))

    np.save(os.path.join(data_dir, 'mean_data.npy'), mean_data)
    np.savetxt(os.path.join(data_dir, 'mean_data.txt'), [mean_data])

    np.save(os.path.join(data_dir, 'var_data.npy'), var_data)
    np.savetxt(os.path.join(data_dir, 'var_data.txt'), [var_data])

    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.savetxt(os.path.join(data_dir, 'X_0.txt'), X[0, :, :], fmt='%6.2f')

    np.save(os.path.join(data_dir, 'Y.npy'), X)
    np.savetxt(os.path.join(data_dir, 'Y_0.txt'), Y[0, :, :], fmt='%6.2f')
