#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 21:53
# @Author  : Tom
# @File    : test_heatmap.py
import datetime
import numba as nb
import numpy as np
from pyheatmap.heatmap import HeatMap


def main():
    # download test data
    # url = "https://raw.github.com/oldj/pyheatmap/master/examples/test_data.txt"
    # sdata = urllib.request.urlopen(url).read().split("\n")
    with open("test_data.txt", 'r') as f:
        sdata = [line.replace("\n", "") for line in f.readlines()]
    data = []
    for ln in sdata:
        a = ln.split(",")
        if len(a) != 2:
            continue
        a = [int(i) for i in a]
        data.append(a)
    # list -> np.ndarray
    data = np.array(data)


    starttime = datetime.datetime.now()

    # start painting
    for i in range(5):
        hm = HeatMap(data)
        doHeatmap(hm)

    endtime = datetime.datetime.now()
    print(f"It costs {(endtime - starttime).seconds}s.")

def doHeatmap(hm):
    # hm.heatmap(save_as="heat.png")
    hm.heatmap(save_as=None)  # Don't save

if __name__ == "__main__":
    main()
