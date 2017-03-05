fname = 'for_elise_20s.wav';
N = 10000;
NFFT = 2048;
WINDOW = 2048;
NOVERLAP = round(NFFT / 3);
HOP = NFFT - NOVERLAP;
ZERO_PHASE_WINDOWING = 1;

[x, Fs] = audioread(fname);
x = x(1:N);

w = vertcat([0], hanning(WINDOW-2), [0]);
w = w ./ sum(w);

spec = xxspecgram(x, NFFT, Fs, w, NOVERLAP, ZERO_PHASE_WINDOWING);
mX = 20 * log10(abs(spec))';
pX = unwrap(angle(spec))';