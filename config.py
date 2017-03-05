# coding: utf-8

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data-mel-autoencoder-3')

weights_dir = os.path.join(data_dir, 'weights')
weights_file = os.path.join(weights_dir, 'weights')
weights_conv_file = os.path.join(weights_dir, 'weightsCONV')
weights_lstm_file = os.path.join(weights_dir, 'weightsLSTM')

music_dir = os.path.join(data_dir, 'music')
music_file = os.path.join(weights_dir, 'music')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

N = 1024 * 2
M = N
fs = 44100
H = round(N / 2.0)  # hop size
NN = N / 2 + 1  # number of meaning-ful Fourier coefficients

# Sin Model
sin_t = -80
# minSineDur = 0.001
minSineDur = 0.025
maxnSines = 200
freqDevOffset = 50
freqDevSlope = 0.001
Ns = N  # size of fft used in synthesis

mem_sec = 3
mem_n = int(mem_sec * fs / H)

gen_time = 1 * 60 / 10
sequence_length = int(gen_time * fs / H)
print 'sequence_length: {}'.format(sequence_length)


# DataSet Vectorization params
max_sentence_duration = 1  # 40 * H / fs  # seconds
print 'max_sentence_duration: {}'.format(max_sentence_duration)
max_sentence_len = int(round(fs * max_sentence_duration / H))
max_sentence_len = 64+1  # 1 + (max_sentence_len // 2) * 2
print 'max_sentence_len: {}'.format(max_sentence_len)
# sentences_overlapping = 0.85
sentences_step = 1  # int(max_sentence_len * (1 - sentences_overlapping))
print 'sentences_step: {}'.format(sentences_step)
assert sentences_step > 0

# сколько фреймов пропустим для анализа в начале каждой проверки,
# чтобы прогреть нейронку и дать ей
# угадать мелодию перед тем как делать предсказания
skip_first = 0

window = 'hamming'
zero_db = -160
hN = (N / 2) + 1


PREPARE_USING_MEL = False
if PREPARE_USING_MEL:
    Nmel = 128
# привычные границы слышимости
min_freq = 0
max_freq = 8000

PREPARE_USING_CROP_FFT = False
if PREPARE_USING_CROP_FFT:
    min_k = int(min_freq * N / fs)
    max_k = int(max_freq * N / fs)
    print 'max_k - min_k: {}'.format(max_k - min_k)

PREPARE_USING_FFT = True
PREPARE_USING_SQUARED_FFT = False

n_features = 16
NP = n_features
Nin = NN
