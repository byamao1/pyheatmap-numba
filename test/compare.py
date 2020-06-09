#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 0:11
# @Author  : Tom
# @File    : t_numba.py

"""
Compare Cupy/numba/numpy

numba: https://numba.pydata.org/numba-doc/latest/user/5minguide.html
"""
import time
import timeit

import numba as nb
import numpy as np
from numba import cuda

def go_cupy():
    import cupy as cp
    a = cp.arange(100).reshape(10, 10)
    trace = 0.0
    for i in range(a.shape[0]):
        trace += cp.tanh(a[i, i])
    return a + trace

@nb.jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def go_jit():  # Function is compiled to machine code when called the first time
    a = np.arange(100).reshape(10, 10)
    trace = 0.0
    for i in range(a.shape[0]):  # Numba likes loops
        trace += np.tanh(a[i, i])  # Numba likes NumPy functions
    return a + trace  # Numba likes NumPy broadcasting

# @cuda.jit
def go_slow():
    a = np.arange(100).reshape(10, 10)
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

@nb.njit
def go_arr(arr):
    t = [1+e for e in arr]
    for i in range(arr.shape[0]):
        arr[i] = 1


if __name__ == '__main__':

    # go_arr(np.zeros(10))
    
    number = 10 ** 5  # 10^5 时才能看到优势
    elapse = timeit.timeit(go_jit, number=number)
    print(f"Jit costs {elapse}")
    elapse = timeit.timeit(go_slow, number=number)
    print(f"Slow costs {elapse}")
    # elapse = timeit.timeit(go_cupy, number=number)
    # print(f"Cupy costs {elapse}")
