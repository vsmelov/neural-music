# coding: utf-8
import numpy as np
import tools.dftModel as DFT


def main():
    N = 16
    x = np.array(range(N))
    w = np.ones(N)
    mX, pX = DFT.dftAnal(x, w, N)
    print 'mX: {}'.format(mX)
    print 'pX: {}'.format(pX)


if __name__ == '__main__':
    main()
