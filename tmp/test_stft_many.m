N = 8*6;
NFFT = 8;
Fs = 8000; % used for scaling plots
WINDOW = NFFT;
NOVERLAP = 0;
HOP = NFFT - NOVERLAP;
ZERO_PHASE_WINDOWING = 1;

x = 0:N-1;

w = vertcat([0], hanning(WINDOW-2), [0]);
w = w ./ sum(w);

y = xxspecgram(x, NFFT, Fs, w, NOVERLAP, ZERO_PHASE_WINDOWING);

mX = 20 * log10(abs(y))'
pX = unwrap(angle(y))'
