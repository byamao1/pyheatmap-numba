#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 0:11
# @Author  : Tom
# @File    : t_numba.py

# https://numba.pydata.org/numba-doc/latest/user/5minguide.html
import timeit

import numba as nb
import numpy as np
from numba import cuda


@nb.jit(nopython=True)  # Set "nopython" mode for best performance, equivalent to @njit
def go_fast():  # Function is compiled to machine code when called the first time
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


if __name__ == '__main__':
    number = 10 ** 6
    elapse = timeit.timeit(go_fast, number=number)
    print(f"Fast costs {elapse}")
    elapse = timeit.timeit(go_slow, number=number)
    print(f"Slow costs {elapse}")
