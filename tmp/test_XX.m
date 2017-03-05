% 'bestrafe_mich_27s.wav'
% 'for_elise_20s.wav'
% 'sm1_cln.wav'
fname = 'brit.wav'
% fname = 'for_elise_20s.wav';

NFFT = 2048*1;
WINDOW = 2048*1;
NOVERLAP = round(NFFT / 2);
HOP = NFFT - NOVERLAP;
ZERO_PHASE_WINDOWING = 1;

[x, Fs] = audioread(fname);

w = hamming(WINDOW);
w = w ./ sum(w);

spec = xxspecgram(x, NFFT, Fs, w, NOVERLAP, ZERO_PHASE_WINDOWING);
mX = 20 * log10(abs(spec));
pX = unwrap(angle(spec));

dlmwrite('/home/vs/ML/neural-music/tmp/mX.txt',mX');

inv_mX = 20 * log10(abs(sqrt(inv_pspectrum)));
dlmwrite('/home/vs/ML/neural-music/tmp/inv_mX.txt',inv_mX');