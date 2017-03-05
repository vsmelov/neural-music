import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift
import math
import tools.dftModel as DFT
import tools.utilFunctions as UF
import tools.stft as STFT
from scipy.signal import get_window
import os


def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20,
                 freqDevSlope=0.01):
    """
    Tracking sinusoids from one frame to the next
    pfreq, pmag, pphase: frequencies and magnitude of current frame
    tfreq: frequencies of incoming tracks from previous frame
    freqDevOffset: minimum frequency deviation at 0Hz
    freqDevSlope: slope increase of minimum frequency deviation
    returns tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
    """

    tfreqn = np.zeros(tfreq.size)  # initialize array for output frequencies
    tmagn = np.zeros(tfreq.size)  # initialize array for output magnitudes
    tphasen = np.zeros(tfreq.size)  # initialize array for output phases
    pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[
        0]  # indexes of current peaks
    incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[
        0]  # indexes of incoming tracks
    newTracks = np.zeros(tfreq.size,
                         dtype=np.int) - 1  # initialize to -1 new tracks
    magOrder = np.argsort(-pmag[pindexes])  # order current peaks by magnitude
    pfreqt = np.copy(pfreq)  # copy current peaks to temporary array
    pmagt = np.copy(pmag)  # copy current peaks to temporary array
    pphaset = np.copy(pphase)  # copy current peaks to temporary array

    # continue incoming tracks
    if incomingTracks.size > 0:  # if incoming tracks exist
        for i in magOrder:  # iterate over current peaks
            if incomingTracks.size == 0:  # break when no more incoming tracks
                break
            track = np.argmin(abs(pfreqt[i] - tfreq[
                incomingTracks]))  # closest incoming track to peak
            freqDistance = abs(pfreq[i] - tfreq[
                incomingTracks[track]])  # measure freq distance
            if freqDistance < (freqDevOffset + freqDevSlope * pfreq[
                i]):  # choose track if distance is small
                newTracks[incomingTracks[
                    track]] = i  # assign peak index to track index
                incomingTracks = np.delete(incomingTracks,
                                           track)  # delete index of track in incomming tracks
    indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[
        0]  # indexes of assigned tracks
    if indext.size > 0:
        indexp = newTracks[indext]  # indexes of assigned peaks
        tfreqn[indext] = pfreqt[indexp]  # output freq tracks
        tmagn[indext] = pmagt[indexp]  # output mag tracks
        tphasen[indext] = pphaset[indexp]  # output phase tracks
        pfreqt = np.delete(pfreqt, indexp)  # delete used peaks
        pmagt = np.delete(pmagt, indexp)  # delete used peaks
        pphaset = np.delete(pphaset, indexp)  # delete used peaks

    # create new tracks from non used peaks
    emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[
        0]  # indexes of empty incoming tracks
    peaksleft = np.argsort(-pmagt)  # sort left peaks by magnitude
    if ((peaksleft.size > 0) & (
        emptyt.size >= peaksleft.size)):  # fill empty tracks
        tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
        tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
        tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif ((peaksleft.size > 0) & (
        emptyt.size < peaksleft.size)):  # add more tracks if necessary
        tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
        tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
        tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
        tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
        tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
        tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    return tfreqn, tmagn, tphasen


def cleaningSineTracks(tfreq, minTrackLength=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    tfreq: frequency of tracks
    minTrackLength: minimum duration of tracks in number of frames
    returns tfreqn: output frequency of tracks
    """

    if tfreq.shape[1] == 0:  # if no tracks return input
        return tfreq
    nFrames = tfreq[:, 0].size  # number of frames
    nTracks = tfreq[0, :].size  # number of tracks in a frame
    for t in range(nTracks):  # iterate over all tracks
        trackFreqs = tfreq[:, t]  # frequencies of one track
        trackBegs = \
        np.nonzero((trackFreqs[:nFrames - 1] <= 0)  # begining of track contours
                   & (trackFreqs[1:] > 0))[0] + 1
        if trackFreqs[0] > 0:
            trackBegs = np.insert(trackBegs, 0, 0)
        trackEnds = \
        np.nonzero((trackFreqs[:nFrames - 1] > 0)  # end of track contours
                   & (trackFreqs[1:] <= 0))[0] + 1
        if trackFreqs[nFrames - 1] > 0:
            trackEnds = np.append(trackEnds, nFrames - 1)
        trackLengths = 1 + trackEnds - trackBegs  # lengths of trach contours
        for i, j in zip(trackBegs, trackLengths):  # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i:i + j] = 0
    return tfreq


def sineModel(x, fs, w, N, t):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB
    returns y: output array sound
    """

    hM1 = int(
        math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns / 4  # Hop size used for analysis and synthesis
    hNs = Ns / 2  # half of synthesis FFT size
    pin = max(hNs, hM1)  # init sound pointer in middle of anal window
    pend = x.size - max(hNs, hM1)  # last sample to start a frame
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    w = w / sum(w)  # normalize analysis window
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[
                                                hNs - H:hNs + H]  # normalized synthesis window
    while pin < pend:  # while input sound pointer is within sound
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX,
                                              ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # -----synthesis-----
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns,
                            fs)  # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))  # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[
        pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
        pin += H  # advance sound pointer
    return y


def sineModelAnal(x, fs, w, N, H, t, maxnSines=100, minSineDur=.01,
                  freqDevOffset=20, freqDevSlope=0.01):
    """
    Analysis of a sound using the sinusoidal model with sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
    maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
    freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
    returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
    """

    if (minSineDur < 0):  # raise error if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = int(
        math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(np.zeros(hM2),
                  x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x,
                  np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    tfreq = np.array([])
    while pin < pend:  # while input sound pointer is within sound
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX,
                                              ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq,
                                           freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines,
                                     tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines,
                                   tmag.size))  # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines,
                                       tphase.size))  # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)  # temporary output array
        jtmag = np.zeros(maxnSines)  # temporary output array
        jtphase = np.zeros(maxnSines)  # temporary output array
        jtfreq[:tfreq.size] = tfreq  # save track frequencies to temporary array
        jtmag[:tmag.size] = tmag  # save track magnitudes to temporary array
        jtphase[
        :tphase.size] = tphase  # save track magnitudes to temporary array
        if pin == hM1:  # if first frame initialize output sine tracks
            xtfreq = jtfreq
            xtmag = jtmag
            xtphase = jtphase
        else:  # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
        pin += H
    # delete sine tracks shorter than minSineDur
    xtfreq = cleaningSineTracks(xtfreq, round(fs * minSineDur / H))
    return xtfreq, xtmag, xtphase


def XXsineModelAnal(mX_list, fs, N, H, t, maxnSines=100, minSineDur=.01,
                  freqDevOffset=20, freqDevSlope=0.01):
    """
    Analysis of a sound using the sinusoidal model with sine tracking
    mX_list: input spectrum,
    N: size of complex spectrum,
    H: hop-size, t: threshold in negative dB
    maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
    freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
    returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
    """

    if (minSineDur < 0):  # raise error if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")
    tfreq = np.array([])

    pin = 0
    while pin < mX_list.shape[0]:  # while input sound pointer is within sound
        mX = mX_list[pin, :]
        pX = np.zeros(mX.shape)
        ploc = UF.peakDetection(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX,
                                              ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq,
                                           freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines,
                                     tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines,
                                   tmag.size))  # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines,
                                       tphase.size))  # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)  # temporary output array
        jtmag = np.zeros(maxnSines)  # temporary output array
        jtphase = np.zeros(maxnSines)  # temporary output array
        jtfreq[:tfreq.size] = tfreq  # save track frequencies to temporary array
        jtmag[:tmag.size] = tmag  # save track magnitudes to temporary array
        jtphase[:tphase.size] = tphase  # save track magnitudes to temporary array
        if pin == 0:  # if first frame initialize output sine tracks
            xtfreq = jtfreq
            xtmag = jtmag
            xtphase = jtphase
        else:  # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
        pin += 1

    # delete sine tracks shorter than minSineDur
    xtfreq = cleaningSineTracks(xtfreq, round(fs * minSineDur / H))

    return xtfreq, xtmag, xtphase


def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal model
    tfreq,tmag,tphase: frequencies, magnitudes and phases of sinusoids
    N: synthesis FFT size, H: hop size, fs: sampling rate
    returns y: output array sound
    """

    hN = N / 2  # half of FFT size for synthesis
    L = tfreq.shape[0]  # number of frames
    pout = 0  # initialize output sound pointer
    ysize = H * (L + 3)  # output sound size
    y = np.zeros(ysize)  # initialize output array
    sw = np.zeros(N)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hN - H:hN + H] = ow  # add triangular window
    bh = blackmanharris(N)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hN - H:hN + H] = sw[hN - H:hN + H] / bh[
                                            hN - H:hN + H]  # normalized synthesis window
    lastytfreq = tfreq[0, :]  # initialize synthesis frequencies
    ytphase = 2 * np.pi * np.random.rand(
        tfreq[0, :].size)  # initialize synthesis phases
    for l in range(L):  # iterate over all frames
        if (tphase.size > 0):  # if no phases generate them
            ytphase = tphase[l, :]
        else:
            ytphase += (np.pi * (
            lastytfreq + tfreq[l, :]) / fs) * H  # propagate phases
        Y = UF.genSpecSines(tfreq[l, :], tmag[l, :], ytphase, N,
                            fs)  # generate sines in the spectrum
        lastytfreq = tfreq[l, :]  # save frequency for phase propagation
        ytphase = ytphase % (2 * np.pi)  # make phase inside 2*pi
        yw = np.real(fftshift(ifft(Y)))  # compute inverse FFT
        y[pout:pout + N] += sw * yw  # overlap-add and apply a synthesis window
        pout += H  # advance sound pointer
    y = np.delete(y, range(hN))  # delete half of first window
    y = np.delete(y,
                  range(y.size - hN, y.size))  # delete half of the last window
    return y


def XXsineModelSynth(tfreq, tmag, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal model
    tfreq, tmag: frequencies and magnitudes of sinusoids
    N: synthesis FFT size, H: hop size, fs: sampling rate
    returns y: output array sound
    """
    hN = N / 2  # half of FFT size for synthesis
    L = tfreq.shape[0]  # number of frames
    pout = 0  # initialize output sound pointer
    ysize = H * (L + 3)  # output sound size
    y = np.zeros(ysize)  # initialize output array
    sw = np.zeros(N)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hN - H:hN + H] = ow  # add triangular window
    bh = blackmanharris(N)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hN - H:hN + H] = sw[hN - H:hN + H] / bh[hN - H:hN + H]  # normalized synthesis window
    lastytfreq = tfreq[0, :]  # initialize synthesis frequencies
    ytphase = 2 * np.pi * np.random.rand(tfreq[0, :].size)  # initialize synthesis phases
    for l in range(L-1):  # iterate over all frames
        ytphase += (np.pi * (lastytfreq + tfreq[l, :]) / fs) * H  # propagate phases
        ytphase = ytphase % (2 * np.pi)  # make phase inside 2*pi
        Y = UF.genSpecSines(tfreq[l, :], tmag[l, :], ytphase, N, fs)  # generate sines in the spectrum
        lastytfreq = tfreq[l, :]  # save frequency for phase propagation
        yw = np.real(fftshift(ifft(Y)))  # compute inverse FFT
        # print 'y.shape: {}, sw.shape: {}, yw.shape: {}'.format(
        #     y.shape,
        #     sw.shape,
        #     yw.shape,
        # )
        # print 'tfreq[l, :] - {}'.format(tfreq[l, :])
        # print 'tmag[l, :] - {}'.format(tmag[l, :])
        # print 'yw: {}'.format(yw)
        y[pout:pout + N] += sw * yw  # overlap-add and apply a synthesis window
        # exit()
        pout += H  # advance sound pointer
        # if pout > L:
        #     print 'BREAK on pout: {}, L: {}'.format(pout, L)
        #     break
    y = np.delete(y, range(hN))  # delete half of first window
    y = np.delete(y, range(y.size - hN, y.size))  # delete half of the last window
    return y


if __name__ == '__main__':
    """
    Perform analysis/synthesis using the sinusoidal model
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size; N: fft size (power of two, bigger or equal than M)
    t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    """
    # inputFile = './male.wav'
    # inputFile = './krovostok_short.wav'
    inputFile = 'bestrafe_mich_27s.wav'
    inputFile = 'for_elise_20s.wav'
    window = 'hamming'
    # M = 2001
    M = 512
    # N = 2048
    N = 512
    t = -80
    # minSineDur = 0.02
    minSineDur = 0.005
    maxnSines = 20
    freqDevOffset = 10
    freqDevOffset = 150
    freqDevSlope = 0.001

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = Ns / 2

    # read input sound
    fs, x = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    print 'x.shape', x.shape
    print 'w.shape', w.shape
    print 'N', N
    print 'H', H

    if 0:
        # analyze the sound with the sinusoidal model
        print 'sineModelAnal'
        tfreq, tmag, tphase = sineModelAnal(x, fs, w, N, H, t, maxnSines,
                                               minSineDur, freqDevOffset,
                                               freqDevSlope)
        print 'tfreq.shape', tfreq.shape
        print 'tmag.shape', tmag.shape
        print 'tphase.shape', tphase.shape

        # synthesize the output sound from the sinusoidal representation
        print 'XXsineModelSynth'
        y = XXsineModelSynth(tfreq, tmag, Ns, H, fs)
        print 'y.shape', y.shape

        # output sound file name
        outputFile = os.path.basename(inputFile)[:-4] + '_XXsineModel.wav'

        # write the synthesized sound obtained from the sinusoidal synthesis
        UF.wavwrite(y, fs, outputFile)
        print 'write', outputFile

    if 1:
        print 'STFT.stftAnal'
        mX, pX = STFT.stftAnal(x, w, N, H)  # find magnitude and phase
        print 'mX.shape', mX.shape
        print 'pX.shape', pX.shape

        print 'XXsineModelAnal'
        tfreq, tmag, tphase = XXsineModelAnal(mX, fs, N, H, t, maxnSines,
                                              minSineDur, freqDevOffset,
                                              freqDevSlope)
        print 'tfreq.shape', tfreq.shape
        print 'tmag.shape', tmag.shape
        print 'tphase.shape', tphase.shape


        # synthesize the output sound from the sinusoidal representation
        print 'XXsineModelSynth'
        y = XXsineModelSynth(tfreq, tmag, Ns, H, fs)
        print 'y.shape', y.shape

        # output sound file name
        outputFile = os.path.basename(inputFile)[:-4] + '_XXXXsineModel.wav'

        # write the synthesized sound obtained from the sinusoidal synthesis
        UF.wavwrite(y, fs, outputFile)
        print 'OK write', outputFile