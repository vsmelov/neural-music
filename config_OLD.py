# coding: utf-8

import os

base_dir = os.path.dirname(os.path.realpath(__file__))
music_dir = os.path.join(base_dir, 'music-3')
data_dir = os.path.join(base_dir, 'data-3')

weights_dir = os.path.join(data_dir, 'weights')
weights_file = os.path.join(weights_dir, 'weights')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# choice of rectangular, hanning, hamming, blackman, blackmanharris
window = 'hanning'

SEC = 1

if not SEC:
    N = 2*512  # fft size, must be power of 2 and >= M
    M = N-1   # window size
    H = int(M / 2)  # hop size
    fs = 44100
    NN = N / 2 + 1  # number of meaning-ful Fourier coefficients
else:
    N = 2048
    M = N
    fs = 44100
    H = int(fs*0.010)  # hop size
    NN = N / 2 + 1  # number of meaning-ful Fourier coefficients

# границы слышимости человека
min_freq = 20
max_freq = 20000

# привычные границы слышимости
min_freq = 20
max_freq = 8000

min_freq = 0
max_freq = 4000

# минимальные и максимальные номера коэффициентов Фурье
# соответствующие ограничениям на спектр
min_k = int(min_freq * N / fs)
max_k = int(max_freq * N / fs)
NP = max_k - min_k
print 'NP: {}'.format(NP)

zero_db = -80  # граница слышимости

mem_sec = 3
mem_n = int(mem_sec * fs / H)
gen_time = 5*60
sequence_length = int(gen_time * fs / H)
print 'sequence_length: {}'.format(sequence_length)

# DataSet Vectorization params
max_sentence_duration = 1.5  # seconds
max_sentence_len = int(fs * max_sentence_duration / H)
sentences_overlapping = 0.25
sentences_step = int(max_sentence_len * (1 - sentences_overlapping))

# сколько фреймов пропустим для анализа в начале каждой проверки,
# чтобы прогреть нейронку и дать ей
# угадать мелодию перед тем как делать предсказания
skip_first = 0


# Sin Model
sin_t = -80
minSineDur = 0.001
maxnSines = 200
freqDevOffset = 50
freqDevSlope = 0.001
Ns = N  # size of fft used in synthesisNs = 512  # size of fft used in synthesis
