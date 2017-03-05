# coding: utf-8

import numpy as np
import tools.stft as STFT
from scipy.signal import get_window


N = 8*6
NFFT = 8
WINDOW = 8
NOVERLAP = 0
HOP = NFFT-NOVERLAP
ZERO_PHASE_WINDOWING = 1


def main():
    x = np.array(range(N))
    w = get_window('hanning', WINDOW, False)
    mX, pX = STFT.stftAnal(x, w, NFFT, HOP,
                           zero_phase_windowing=ZERO_PHASE_WINDOWING)
    print 'mX: {}'.format(mX)
    print 'pX: {}'.format(pX)


if __name__ == '__main__':
    main()
