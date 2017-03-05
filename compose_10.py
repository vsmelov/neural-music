# coding: utf-8

import numpy as np

from config import *
from model.get_model import get_model
from tools.mX2x import crop_fft2audio, mel2audio
import tools.utilFunctions as UF

X = np.load(os.path.join(data_dir, 'X10.npy'))
shift_data = np.loadtxt(os.path.join(data_dir, 'shift_data.txt'))
var_data = np.loadtxt(os.path.join(data_dir, 'var_data.txt'))

print 'X.shape: {}'.format(X.shape)

print('NP = {}'.format(NP))
assert X.shape[2] == NP

model = get_model(load_wights=1, f_weights=weights_file+'LSTM', stateful=True)
print('model.summary() = {}'.format(model.summary()))
print('model.get_config() = {}'.format(model.get_config()))


def generate_copy_seed_sequence(training_data):
    num_examples = training_data.shape[0]
    # TODO: random choice or may be linear combination
    randIdx = np.random.randint(num_examples, size=1)[0]
    print 'randIdx: {}'.format(randIdx)
    randSeed = training_data[randIdx, :, :]
    seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))
    return seedSeq


def generate_from_seed(model, seed, sequence_length):
    seedSeq = seed.copy()
    output = []
    for it in xrange(sequence_length):
        seedSeqNew = model.predict(seedSeq)  # Step 1. Generate X_n + 1
        print('Gen {}/{}, seedSeqNew.shape: {}'.format(
            it+1, sequence_length, seedSeqNew.shape))
        if it == 0:
            for i in xrange(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][-1].copy())
        newSeq = seedSeqNew[0][-1]
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)[:, -mem_n:, :]
        print('seedSeq.shape: {}'.format(seedSeq.shape))
    return output


def xxgenerate_from_seed(model, seed, sequence_length):
    output = []
    for i in xrange(seed.shape[1]):
        s = np.reshape(seed[0, i, :], (1, 1, NP))
        prediction = model.predict(s)
        print 'prediction.shape: {}'.format(prediction.shape)
        output.append(prediction[0][0].copy())

    s = np.reshape(output[-1], (1, 1, NP))
    for it in xrange(sequence_length):
        prediction = model.predict(s)
        output.append(prediction[0][0].copy())

        # add some noise for prevent from cycling in one tact
        if it and it % 80 == 0:
            model.reset_states()
            print 'reset'*10

        if it % 40 < 10:
            prediction[0][0] *= 1 + (np.random.random_sample(
                prediction[0][0].shape) - 0.5) * 2 * 0.2
        else:
            prediction[0][0] *= 1 + (np.random.random_sample(
                prediction[0][0].shape) - 0.5) * 2 * 0.3

        s = np.reshape(prediction[0][0], (1, 1, NP))
        print('Gen {}/{}, prediction.shape: {}'.format(
                it + 1, sequence_length, prediction.shape))
    return output

print ('Starting generation!')
seed_len = 1
seed_seq = generate_copy_seed_sequence(X)

GEN = 1
if GEN:
    print('seed_seq.shape: {}'.format(seed_seq.shape))
    output = xxgenerate_from_seed(model=model, seed=seed_seq, sequence_length=sequence_length)
    print('Finished generation!')
    output = np.array(output)
else:
    # synthesis from seed only
    output = np.reshape(seed_seq, (seed_seq.shape[1], seed_seq.shape[2]))






# DECODE and RESHAPE to (None, NP)
print 'DECODE and RESHAPE to (None, NP)'
from model.get_autoencoder import get_encoder

autoencoder, encoder, decoder = get_encoder(10, weights_file+'CONV')

print '1output.shape: {}'.format(output.shape)
output = decoder.predict(output)
print '2output.shape: {}'.format(output.shape)

new_output = np.zeros((output.shape[0]*output.shape[1], output.shape[2]))
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        new_output[output.shape[1]*i+j, :] = output[i, j]

output = new_output
print '3output.shape: {}'.format(output.shape)
print 'Start synth'









output = var_data * output + shift_data
np.savetxt(os.path.join(data_dir, 'out_data.txt'), output, fmt='%6.2f', delimiter=',')

print 'output.shape: {}'.format(output.shape)

if PREPARE_USING_MEL:
    audio = mel2audio(output)
else:
    raise ValueError('unknown data preparetion')
# audio = crop_fft2audio(out_mX, N, H, -80, 80)
UF.wavwrite(audio, fs, 'out.wav')
