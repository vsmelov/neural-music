# coding: utf-8

import tools.utilFunctions as UF
import numpy as np
import tools.stft as STFT
from scipy.signal import get_window


fname = 'for_elise_20s.wav';
N = 10000;
NFFT = 2048;
WINDOW = 2048;
NOVERLAP = int(round(NFFT * 1.0 / 3));
HOP = NFFT - NOVERLAP;
ZERO_PHASE_WINDOWING = 1;

Fs, x  = UF.wavread(fname)
x = x[:N]

w = get_window('hanning', WINDOW, False)
mX, pX = STFT.stftAnal(x, w, NFFT, HOP,
                       zero_phase_windowing=ZERO_PHASE_WINDOWING)
print 'mX: {}'.format(mX)
print 'pX: {}'.format(pX)
