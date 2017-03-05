# coding: utf-8
import os
from scipy.signal import get_window
import tools.stft as STFT
import tools.utilFunctions as UF
import numpy as np
from config import *
from tools.fft2melmx import melFilterBank
from tools.mX2x import audio2mel, mel2audio, fft2audio, crop_fft2audio, data2audio
import pickle


def get_input_file_list(dir):
    input_file_list = []
    for file_name in os.listdir(dir):
        if os.path.splitext(file_name)[1] != '.wav':
            print 'SKIP: file {}'.format(file_name)
            continue
        else:
            print 'ADD: file {}'.format(file_name)
            input_file_list.append(file_name)
    return input_file_list


def get_mX_list():
    mX_list = []
    input_file_list = get_input_file_list(music_dir)
    print 'START: calc STFT (it can take a long time)'

    if PREPARE_USING_MEL:
        bank = melFilterBank(hN, Nmel, min_freq, max_freq)

    for i_input_file, input_file in enumerate(input_file_list):
        print 'START: {}'.format(input_file)
        input_file_path = os.path.join(music_dir, input_file)
        fs, x = UF.wavread(input_file_path)
        w = get_window(window, M)
        # TODO: slow!

        if PREPARE_USING_FFT:
            mX, pX = STFT.stftAnal(x, w, N, H)
            # звуки ниже zero_db считаем неслышимыми
            # это будет наш "абсолютный ноль"
            mX[mX < zero_db] = zero_db
            data = mX
        elif PREPARE_USING_SQUARED_FFT:
            mX, pX = STFT.stftAnal(x, w, N, H, db=False)
            # звуки ниже zero_db считаем неслышимыми
            # это будет наш "абсолютный ноль"
            mX[mX < zero_db] = zero_db
            mX -= zero_db
            mX **= 2
            data = mX
        elif PREPARE_USING_CROP_FFT:
            mX, pX = STFT.stftAnal(x, w, N, H)
            # звуки ниже zero_db считаем неслышимыми
            # это будет наш "абсолютный ноль"
            mX[mX < zero_db] = zero_db
            data = mX[:, min_k:max_k]  # обрезанный спектр
        elif PREPARE_USING_MEL:
            data = audio2mel(x)
        else:
            raise ValueError('unknown data preparation')
        mX_list.append(data)
        print 'FINISH: {}'.format(input_file)
    print 'END: calc STFT'

    shift_mX = np.mean(np.mean(
        np.concatenate(mX_list)
    ))
    print 'shift_mX: {}'.format(shift_mX)

    var_mX = np.sqrt(
        np.mean(
            np.mean(np.abs(np.concatenate(mX_list) - shift_mX) ** 2)
        )
    )
    print 'var_mX: {}'.format(var_mX)

    # normalize
    for i in range(len(mX_list)):
        mX_list[i] = (mX_list[i] - shift_mX) / var_mX

    return mX_list, shift_mX, var_mX


if __name__ == '__main__':
    mX_list, shift_mX, var_mX = get_mX_list()

    with open(os.path.join(data_dir, 'mX_list.dat'), 'wb') as f:
        pickle.dump(mX_list, f)

    sentences = []
    for mXaudio in mX_list:
        for i in range(0, mXaudio.shape[0] - max_sentence_len, sentences_step):
            sentences.append(mXaudio[i:i + max_sentence_len, :])
    print('len(sequences):', len(sentences))

    print('Vectorization...')
    SENTENCE = np.zeros(
        (len(sentences), max_sentence_len, sentences[0].shape[1]),
        dtype=np.float)
    X = np.zeros(
        (len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]),
        dtype=np.float)
    Y = np.zeros(
        (len(sentences), max_sentence_len-skip_first-1, sentences[0].shape[1]),
        dtype=np.float)
    for i, sentence in enumerate(sentences):
        SENTENCE[i, :, :] = sentence[:, :]
        X[i, :, :] = sentence[skip_first:-1, :]
        Y[i, :, :] = sentence[skip_first+1:, :]

    Xconcat = np.concatenate(mX_list)

    print('SENTENCE.shape: {}'.format(SENTENCE.shape))
    print('X.shape: {}'.format(X.shape))
    print('Y.shape: {}'.format(Y.shape))
    print('Xconcat.shape: {}'.format(Xconcat.shape))

    np.save(os.path.join(data_dir, 'shift_data.npy'), shift_mX)
    np.savetxt(os.path.join(data_dir, 'shift_data.txt'), [shift_mX])

    np.save(os.path.join(data_dir, 'var_data.npy'), var_mX)
    np.savetxt(os.path.join(data_dir, 'var_data.txt'), [var_mX])

    np.save(os.path.join(data_dir, 'SENTENCE.npy'), SENTENCE)
    np.savetxt(os.path.join(data_dir, 'SENTENCE_0.txt'), SENTENCE[0, :, :], fmt='%6.2f')

    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.savetxt(os.path.join(data_dir, 'X_0.txt'), X[0, :, :], fmt='%6.2f')

    np.save(os.path.join(data_dir, 'Y.npy'), X)
    np.savetxt(os.path.join(data_dir, 'Y_0.txt'), Y[0, :, :], fmt='%6.2f')

    np.save(os.path.join(data_dir, 'Xconcat.npy'), Xconcat)
    np.savetxt(os.path.join(data_dir, 'Xconcat.txt'), Xconcat, fmt='%6.2f')

    print 'FINISH: save vectors to files'

    for i in range(10):
        num_examples = X.shape[0]
        randIdx = np.random.randint(num_examples, size=1)[0]
        data = X[randIdx, :, :]
        data = data * var_mX + shift_mX
        audio = data2audio(data, normalized=False)
        fname = os.path.join(data_dir, 'resynth_{}.wav'.format(i))
        UF.wavwrite(audio, fs, fname)
        print 'OK: {}'.format(fname)

    countinous_sentence_list = np.concatenate(mX_list)
    all_original = countinous_sentence_list * var_mX + shift_mX
    all_original_audio = data2audio(all_original, normalized=False)
    UF.wavwrite(all_original_audio, fs, 'all_original_audio.wav')
