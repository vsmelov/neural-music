# coding: utf-8
from random import randint

A = [randint(0, 3) for i in range(10)]
print A
N_zeros = 0
i = 0
while i < len(A):
    if A[i] == 0:
        N_zeros += 1
    else:
        A[i-N_zeros] = A[i]
    i += 1
A = A[:-N_zeros]
print A