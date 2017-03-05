# coding: utf-8

from config import *
from tools.sineModel import XXsineModelAnal, XXsineModelSynth
from tools.fft2melmx import melFilterBank
import numpy as np
import tools.utilFunctions as UF
import tools.stft as STFT
from scipy.signal import get_window


def data2audio(data, normalized=True):
    if normalized:
        shift_data = np.loadtxt(os.path.join(data_dir, 'shift_data.txt'))
        var_data = np.loadtxt(os.path.join(data_dir, 'var_data.txt'))
        data = data * var_data / shift_data

    if PREPARE_USING_FFT:
        return fft2audio(data)
    elif PREPARE_USING_SQUARED_FFT:
        data **= 0.5
        data += zero_db
        return fft2audio(data)
    elif PREPARE_USING_CROP_FFT:
        return crop_fft2audio(data)
    elif PREPARE_USING_MEL:
        return mel2audio(data)


def crop_fft2audio(mX):
    mX_full = np.zeros((mX.shape[0], NN)) + zero_db
    mX_full[:, min_k:max_k] = mX
    return fft2audio(mX_full, N, H)


def fft2audio(mX):
    tfreq, tmag, tphase = XXsineModelAnal(mX, fs, N, H, sin_t, maxnSines,
                                          minSineDur, freqDevOffset,
                                          freqDevSlope)
    audio = XXsineModelSynth(tfreq, tmag, Ns, H, fs)
    return audio


def audio2mel(audio):
    w = get_window(window, M)
    mX, pX = STFT.stftAnal(audio, w, N, H, db=False)
    bank = melFilterBank(hN, Nmel, min_freq, max_freq)
    mel = np.dot(mX, bank)
    mel[mel < 10**(zero_db/20)] = 10**(zero_db/20)
    mel = 20 * np.log10(mel)
    # print np.sort(list(set(np.concatenate(mel))))
    return mel


def mel2audio(mel):
    mel = 10 ** (mel / 20)
    bank = melFilterBank(hN, Nmel, min_freq, max_freq)
    bank_inv = np.linalg.pinv(bank)
    mX = np.dot(mel, bank_inv)
    mX[mX < 10 ** (zero_db / 20)] = 10 ** (zero_db / 20)
    mX = 20 * np.log10(mX)
    audio = fft2audio(mX, N, H)
    return audio


def reconvert(fin, fout, min_k=None, max_k=None):
    fs, x = UF.wavread(fin)
    if PREPARE_USING_CROP_FFT:
        w = get_window('hamming', M)
        mX, pX = STFT.stftAnal(x, w, N, H)
        mX[:, :min_k] = zero_db
        mX[:, max_k:] = zero_db
        print 'mX cut min_k: {}, max_k: {}, NP: {}'.format(
            min_k, max_k, NP)
        audio = fft2audio(mX, N, H)
    elif PREPARE_USING_MEL:
        mel = audio2mel(x)
        print 'min(min(mel)): {}'.format(np.min(np.min(mel)))
        print 'max(max(mel)): {}'.format(np.max(np.max(mel)))
        print 'mean(mean(mel)): {}'.format(np.mean(np.mean(mel)))
        audio = mel2audio(mel)
    UF.wavwrite(audio, fs, fout)
    print 'OK: {} --> {}'.format(fin, fout)


if __name__ == '__main__':
    fin = 'df/for_elise_20s.wav'
    fout = 'df/for_elise_20s_reconverted_2048_mel_{}.wav'.format(
        'fft_pow_mel_log_128'
    )
    reconvert(fin, fout)
