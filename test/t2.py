#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 9:24
# @Author  : Tom
# @File    : t2.py

import os
import numba

@numba.jit()
def c(n):
    count=0
    for i in range(n):
        for i in range(n):
            count+=1
    return count
n=99999
c(n)