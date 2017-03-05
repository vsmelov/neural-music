# coding: utf-8

import numpy as np
import random
from config import *
from model.get_autoencoder import get_encoder, get_conv_encoder
from tools.mX2x import audio2mel, mel2audio, data2audio
import tools.utilFunctions as UF
import pickle

SENTENCE = np.load(os.path.join(data_dir, 'SENTENCE.npy'))
print 'SENTENCE.shape: {}'.format(SENTENCE.shape)

with open(os.path.join(data_dir, 'mX_list.dat'), 'rb') as f:
    mX_list = pickle.load(f)
    countinous_sentence_list = np.concatenate(mX_list)

shift_data = np.loadtxt(os.path.join(data_dir, 'shift_data.txt'))
var_data = np.loadtxt(os.path.join(data_dir, 'var_data.txt'))

autoencoder, encoder, decoder = get_conv_encoder()

print('autoencoder.summary() = {}'.format(autoencoder.summary()))
# print('encoder.summary() = {}'.format(encoder.summary()))
# print('decoder.summary() = {}'.format(decoder.summary()))

rand_i = random.randint(0, SENTENCE.shape[0])
print 'rand_i: '.format(rand_i)
sentence = SENTENCE[rand_i, :, :]
sentence = countinous_sentence_list

sentence_2d = np.reshape(sentence, (sentence.shape[-2], sentence.shape[-1]))
np.savetxt(os.path.join(data_dir, 'original_sentence.txt'), sentence_2d,
           fmt='%.2f')

sentence_in = np.reshape(sentence, (1, -1, sentence.shape[1]))
print 'sentence_in.shape: {}'.format(
    sentence_in.shape)


encoded = encoder.predict(sentence_in)
print 'encoded.shape: {}'.format(
    encoded.shape)

mean_encoded = np.mean(np.mean(encoded))
print 'mean_encoded: {}'.format(mean_encoded)

std_encoded = np.sqrt(np.mean(np.mean(
    (encoded - mean_encoded) ** 2
)))
print 'std_encoded: {}'.format(std_encoded)

original = sentence * var_data + shift_data
original_audio = data2audio(original, False)
UF.wavwrite(original_audio, fs,
                os.path.join(data_dir, 'original_audio.wav'))

encoded_2d = np.reshape(encoded, (encoded.shape[1], encoded.shape[2]))
np.savetxt(os.path.join(data_dir, 'encoded.txt'), encoded_2d,
           fmt='%.2f')

decoded = decoder.predict(encoded)
decoded = decoded[0, :, :] * var_data + shift_data
decoded_audio = data2audio(decoded, False)
UF.wavwrite(decoded_audio, fs,
                os.path.join(data_dir, 'decoded_audio.wav'))
