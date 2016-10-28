# coding: utf-8

from config import *
from tools.sineModel import XXsineModelAnal, XXsineModelSynth
import numpy as np


def mX2audio(mX, N, H, shift_X, var_X):
    # mX = np.sqrt(mX)
    mX = mX * var_X + shift_X
    mX_full = np.zeros((mX.shape[0], NN)) + zero_db
    print 'mX.shape: {}'.format(mX.shape)
    print 'mX_full.shape: {}'.format(mX_full.shape)
    mX_full[:, min_k:max_k] = mX
    tfreq, tmag, tphase = XXsineModelAnal(mX, fs, N, H, sin_t, maxnSines,
                                          minSineDur, freqDevOffset,
                                          freqDevSlope)
    audio = XXsineModelSynth(tfreq, tmag, Ns, H, fs)
    return audio

if __name__ == '__main__':
    import tools.utilFunctions as UF

    with open(os.path.join(data_dir, 'X.npy'), 'rb') as f:
        sentences = np.load(f)
    mX = np.concatenate(sentences[:30])
    audio = mX2audio(mX, N, H, -80, 80)
    UF.wavwrite(audio, fs, 'test.wav')
