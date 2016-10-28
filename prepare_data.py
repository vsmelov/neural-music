# coding: utf-8
import os
from scipy.signal import get_window
import tools.stft as STFT
import tools.utilFunctions as UF
import numpy as np
from config import *


def get_input_file_list():
    input_file_list = []
    for file_name in os.listdir(data_dir):
        if os.path.splitext(file_name)[1] != '.wav':
            print 'SKIP: file {}'.format(file_name)
            continue
        else:
            print 'ADD: file {}'.format(file_name)
            input_file_list.append(file_name)
    return input_file_list


if __name__ == '__main__':
    mX_list = []  # list of signal-magnitude for each file
    # mean_mX_list = []
    # std_mX_list = []
    input_file_list = get_input_file_list()
    print 'START: calc STFT (it can take a long time)'
    for i_input_file, input_file in enumerate(input_file_list):
        input_file_path = os.path.join(data_dir, input_file)
        fs, x = UF.wavread(input_file_path)
        w = get_window(window, M)
        # TODO: find C-lib, very slow!
        mX, pX = STFT.stftAnal(x, w, N, H)  # find magnitude and phase
        for i in range(mX.shape[0]):
            for j in range(mX.shape[1]):
                # звуки ниже zero_db считаем неслышимыми
                # это будет наш "абсолютный ноль"
                if mX[i, j] < zero_db:
                    mX[i, j] = zero_db
                pass

        mXaudio = mX[:, min_k:max_k]  # обрезанный спектр
        mX_list.append(mXaudio)
        # параметры нормализации для этого файла
        # mean_mX = np.mean(np.mean(mXaudio))
        # STD across num examples and num timesteps
        # std_mX = np.sqrt(
        #     np.mean(np.mean(
        #         np.abs(mXaudio - mean_mX) ** 2)))
        # mean_mX_list.append(mean_mX)
        # std_mX_list.append(std_mX)
    print 'END: calc STFT'


    # общие параметры нормализации
    # mean_mX = np.mean(np.mean(
    #     np.concatenate(mX_list)
    # ))
    # std_mX = np.sqrt(
    #     np.mean(np.mean(
    #         (np.concatenate(mX_list) - mean_mX) ** 2)))  # STD across audio-files, num examples and num timesteps


    # благодаря этому, у нас входные параметры всегда будут в интервале [0; 1]
    shift_mX = -80  # TODO: rename to shift
    var_mX = 80


    sentences = []
    for mXaudio in mX_list:
        mXaudio = (mXaudio - shift_mX) / var_mX
        for i in range(0, mXaudio.shape[0] - max_sentence_len, sentences_step):
            sentences.append(mXaudio[i:i + max_sentence_len, :])
    print('len(sequences):', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]), dtype=np.float)
    Y = np.zeros((len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]), dtype=np.float)
    for i, sentence in enumerate(sentences):
        X[i, :, :] = sentence[skip_first:-1, :]
        Y[i, :, :] = sentence[skip_first+1:, :]

    print('X.shape: {}'.format(X.shape))

    np.save(os.path.join(data_dir, 'shift_mX.npy'), shift_mX)
    np.savetxt(os.path.join(data_dir, 'shift_mX.txt'), [shift_mX])

    np.save(os.path.join(data_dir, 'var_mX.npy'), var_mX)
    np.savetxt(os.path.join(data_dir, 'var_mX.txt'), [var_mX])

    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.savetxt(os.path.join(data_dir, 'X_0.txt'), X[0, :, :], fmt='%6.2f')

    np.save(os.path.join(data_dir, 'Y.npy'), X)
    np.savetxt(os.path.join(data_dir, 'Y_0.txt'), Y[0, :, :], fmt='%6.2f')
