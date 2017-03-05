# coding: utf-8
import tools.utilFunctions as UF
import numpy

if __name__ == "__main__":
    f = 'for_elise_20s'
    fname = "/neural-music/{f}.wav".format(f=f)
    fout = "/neural-music/{f}_out.wav".format(f=f)
    fs, signal = UF.wavread(fname)
    pre_emphasis = 0.98
    emphasized_signal = numpy.append(signal[0],
                                     signal[1:] - pre_emphasis * signal[:-1])
    UF.wavwrite(emphasized_signal, fs, fout)
