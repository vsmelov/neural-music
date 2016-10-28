# coding: utf-8

import numpy as np

from config import *
from model.get_model import get_model
from tools.mX2x import mX2audio
import tools.utilFunctions as UF

X = np.load(os.path.join(data_dir, 'X.npy'))
Y = np.load(os.path.join(data_dir, 'Y.npy'))
shift_mX = np.loadtxt(os.path.join(data_dir, 'shift_mX.txt'))
var_mX = np.loadtxt(os.path.join(data_dir, 'var_mX.txt'))

print 'X.shape: {}'.format(X.shape)

print('NP = {}'.format(NP))
assert X.shape[2] == NP

model = get_model(NP, stateful=True)
# model = get_model(NP)
print('model.summary() = {}'.format(model.summary()))
print('model.get_config() = {}'.format(model.get_config()))


def generate_copy_seed_sequence(training_data):
    num_examples = training_data.shape[0]
    # TODO: random choice or may be linear combination
    randIdx = np.random.randint(num_examples, size=1)[0]
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
        if it and it % 2000 == 0:
            model.reset_states()
            print 'reset'
        if it % 300 < 50:
            prediction[0][0] += (np.random.random_sample(prediction[0][0].shape)-0.5)*2

        s = np.reshape(prediction[0][0], (1, 1, NP))
        print('Gen {}/{}, prediction.shape: {}'.format(
                it + 1, sequence_length, prediction.shape))
    return output

print ('Starting generation!')
seed_len = 1
seed_seq = generate_copy_seed_sequence(X)
print('seed_seq.shape: {}'.format(seed_seq.shape))
output = xxgenerate_from_seed(model=model, seed=seed_seq, sequence_length=sequence_length)
print('Finished generation!')
output = np.array(output)

out_mX = output
np.savetxt(os.path.join(data_dir, 'out_mX.txt'), out_mX, fmt='%6.2f')

print('len(output): {}'.format(len(output)))
print('len(output[0]): {}'.format(len(output[0])))
print('out_mX.shape: {}'.format(out_mX.shape))

audio = mX2audio(out_mX, N, H, -80, 80)
UF.wavwrite(audio, fs, 'out.wav')
