# import numpy as np
#
#
# def hz2mel(f):
#     """Convert an array of frequency in Hz into mel."""
#     return 1127.01048 * np.log(f/700 +1)
#
#
# def mel2hz(m):
#     """Convert an array of frequency in Hz into mel."""
#     return (np.exp(m / 1127.01048) - 1) * 700
#
#
# def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
#     """Compute triangular filterbank for MFCC computation."""
#     # Total number of filters
#     nfilt = nlinfilt + nlogfilt
#
#     #------------------------
#     # Compute the filter bank
#     #------------------------
#     # Compute start/middle/end points of the triangular filters in spectral
#     # domain
#     freqs = np.zeros(nfilt+2)
#     freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
#     freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
#     heights = 2./(freqs[2:] - freqs[0:-2])
#
#     # Compute filterbank coeff (in fft domain, in bins)
#     fbank = np.zeros((nfilt, nfft))
#     # FFT bins (in Hz)
#     nfreqs = np.arange(nfft) / (1. * nfft) * fs
#     for i in range(nfilt):
#         low = freqs[i]
#         cen = freqs[i+1]
#         hi = freqs[i+2]
#
#         lid = np.arange(np.floor(low * nfft / fs) + 1,
#                         np.floor(cen * nfft / fs) + 1, dtype=np.int)
#         lslope = heights[i] / (cen - low)
#         rid = np.arange(np.floor(cen * nfft / fs) + 1,
#                         np.floor(hi * nfft / fs) + 1, dtype=np.int)
#         rslope = heights[i] / (hi - cen)
#         fbank[i][lid] = lslope * (nfreqs[lid] - low)
#         fbank[i][rid] = rslope * (hi - nfreqs[rid])
#
#     return fbank, freqs

import numpy
import math


def melFilterBank(blockSize, numCoefficients, minHz, maxHz):
    numBands = int(numCoefficients)
    maxMel = int(freqToMel(maxHz))
    minMel = int(freqToMel(minHz))

    # Create a matrix for triangular filters, one row per filter
    # hN = (blockSize / 2) + 1  # size of positive spectrum, it includes sample 0
    hN = blockSize
    filterMatrix = numpy.zeros((numBands, hN))

    melRange = numpy.array(xrange(numBands + 2))

    melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1) + minMel

    # each array index represent the center of each triangular filter
    aux = numpy.log(1 + 1000.0 / 700.0) / 1000.0
    aux = (numpy.exp(melCenterFilters * aux) - 1) / 22050
    aux = 0.5 + 700 * blockSize * aux
    aux = numpy.floor(aux)  # Arredonda pra baixo
    centerIndex = numpy.array(aux, int)  # Get int values

    for i in xrange(numBands):
        start, centre, end = centerIndex[i:i + 3]
        k1 = numpy.float32(centre - start)
        k2 = numpy.float32(end - centre)
        up = (numpy.array(xrange(start, centre)) - start) / k1
        down = (end - numpy.array(xrange(centre, end))) / k2

        filterMatrix[i][start:centre] = up
        filterMatrix[i][centre:end] = down

    return filterMatrix.transpose()


def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)


def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)


if __name__ == '__main__':
    from scipy.signal import get_window
    import tools.stft as STFT
    import tools.utilFunctions as UF
    import numpy as np
    from config import *

    fs, x = UF.wavread('/neural-music/df/for_elise_20s.wav')
    w = get_window(window, M)
    # TODO: slow!
    mX, pX = STFT.stftAnal(x, w, N, H, db=False)  # find magnitude and phase
    mX[mX < zero_db] = zero_db

    print 'mX.shape: {}'.format(mX.shape)
    np.savetxt('mX_.txt', mX)

    bank = melFilterBank(hN, Nmel, 0, 8000)
    print 'bank.shape: {}'.format(bank.shape)
    np.savetxt('bank_.txt', bank)

    bank_inv = np.linalg.pinv(bank)
    print 'bank_inv.shape: {}'.format(bank_inv.shape)
    np.savetxt('bank_inv_.txt', bank_inv)

    bank_bank_inv = np.dot(bank, bank_inv)
    print 'bank*bank_inv.shape: {}'.format(bank_bank_inv.shape)
    np.savetxt('bank*bank_inv_.txt', bank_bank_inv)

    filtered_mX = np.dot(mX, bank)
    print 'filtered_mX.shape: {}'.format(filtered_mX.shape)
    np.savetxt('filtered_mX_.txt', filtered_mX)

    inv_mX = np.dot(filtered_mX, bank_inv)
    print 'inv_mX.shape: {}'.format(inv_mX.shape)
    np.savetxt('inv_mX_.txt', inv_mX)
