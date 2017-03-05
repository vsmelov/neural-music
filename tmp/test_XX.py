# coding: utf-8
import numpy as np
import tools.utilFunctions as UF
from tools.mX2x import fft2audio
from config import *
from scipy.interpolate import interp1d

# fname = 'tmp/mX.txt'
# fname = 'tmp/inv_mX.txt'
fname = 'tmp/gen_mX.txt'


def main():
    mX = np.loadtxt(fname, delimiter=',')
    mX[mX < -180] = -180
    print 'mX.shape: {}'.format(mX.shape)

    # if interp_factor > 1:
    #     crop_mX = mX
    #     new_mX = np.zeros((mX.shape[0]*2, mX.shape[1]))
    #     # crop_mX = mX[::factor, :]
    #     L = crop_mX.shape[0]
    #     for i in range(crop_mX.shape[1]):
    #         x = np.arange(0, L, 1.0)
    #         y = crop_mX[:, i]
    #         f = interp1d(x, y, kind='nearest')
    #         # f = interp1d(x, y)
    #         new_x = np.arange(0, L, 1.0/interp_factor)
    #         print i
    #         new_y = f(new_x[:-interp_factor])
    #         new_mX[:new_y.shape[0], i] = new_y
    #     mX = new_mX

    fout = 'df/for_elise_GEN.wav'
    audio = fft2audio(mX, N, H)
    UF.wavwrite(audio, fs, fout)
    print 'OK: convert MFCC to {}'.format(fout)

if __name__ == '__main__':
    main()
