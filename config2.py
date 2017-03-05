# coding: utf-8

N = 1024*2
M = N
fs = 44100
H = round(N / 2.0)  # hop size
NN = N / 2 + 1  # number of meaning-ful Fourier coefficients

# Sin Model
sin_t = -80
minSineDur = 0.001
maxnSines = 200
freqDevOffset = 50
freqDevSlope = 0.001
Ns = N  # size of fft used in synthesis




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

zero_db = -160  # граница слышимости
