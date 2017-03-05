N = 8;
NFFT = N;
Fs = 8000; % used for scaling plots
WINDOW = N;
NOVERLAP = N / 2;
ZERO_PHASE = 1;

x = 0:N-1;

w_tmp = vertcat([0], hanning(WINDOW-2), [0]);
w_tmp = w_tmp ./ sum(w_tmp);
w = w_tmp;

y = xxspecgram(x, NFFT, Fs, w, NOVERLAP, ZERO_PHASE);

mX = 20 * log10(abs(y))'
pX = unwrap(angle(y))'
