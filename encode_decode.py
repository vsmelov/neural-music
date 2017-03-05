# coding: utf-8
import os
from scipy.signal import get_window
import tools.stft as STFT
import tools.utilFunctions as UF
import numpy as np
from config import *
from tools.fft2melmx import melFilterBank
from tools.mX2x import audio2mel, data2audio
from model.get_autoencoder import get_encoder, get_conv_encoder
import pickle


SENTENCE = np.load(os.path.join(data_dir, 'SENTENCE.npy'))
print 'SENTENCE.shape: {}'.format(SENTENCE.shape)

shift_data = np.loadtxt(os.path.join(data_dir, 'shift_data.txt'))
var_data = np.loadtxt(os.path.join(data_dir, 'var_data.txt'))


autoencoder, encoder, decoder = get_conv_encoder()

print('autoencoder.summary() = {}\n'.format(autoencoder.summary()))
print('encoder.summary() = {}\n'.format(encoder.summary()))
print('decoder.summary() = {}\n'.format(decoder.summary()))


def main():
    with open(os.path.join(data_dir, 'mX_list.dat'), 'rb') as f:
        mX_list = pickle.load(f)

    countinous_sentence_list = np.concatenate(mX_list)
    print 'countinous_sentence_list.shape: {}'.format(countinous_sentence_list.shape)

    countinous_sentence_list_in = np.reshape(
        countinous_sentence_list,
        (1, countinous_sentence_list.shape[0], countinous_sentence_list.shape[1])
    )

    encoded = encoder.predict(countinous_sentence_list_in)
    print 'encoded.shape: {}'.format(
        encoded.shape)

    # r = np.random.normal(1, 0.3, encoded.shape)
    # r[r > 1.1] = 1.0
    # encoded = encoded * r

    decoded = decoder.predict(encoded)
    print 'decoded.shape: {}'.format(
        decoded.shape)

    decoded = np.reshape(decoded, (decoded.shape[1], decoded.shape[2]))
    print 'decoded.shape: {}'.format(
        decoded.shape)

    # original = np.concatenate(countinous_sentence_list)
    # output = np.concatenate(decoded)

    print countinous_sentence_list[10, :]
    print decoded[10, :]

    original = countinous_sentence_list * var_data + shift_data
    output = decoded * var_data + shift_data

    original_audio = data2audio(original, False)
    output_audio = data2audio(output, False)

    UF.wavwrite(original_audio, fs, 'original_audio.wav')
    UF.wavwrite(output_audio, fs, 'output_audio.wav')

if __name__ == '__main__':
    main()
