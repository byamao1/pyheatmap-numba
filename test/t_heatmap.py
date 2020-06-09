#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 21:53
# @Author  : Tom
# @File    : test_heatmap.py

import os
import sys
project_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.append(project_dir)

import datetime

import cv2
import numba as nb
import numpy as np
from PIL import Image

from pyheatmap.heatmap import HeatMap


def get_test_data():
    return [[137, 531], [80, 505], [411, 694], [176, 638], [239, 377], [175, 685], [323, 570], [143, 468], [272, 495],
            [81, 658], [124, 461], [180, 460], [289, 658], [137, 591], [60, 655], [226, 494], [304, 315], [162, 638],
            [393, 661], [103, 389], [206, 650], [271, 456], [428, 532], [167, 711], [403, 493], [91, 565], [251, 363], [30, 696], [105, 435],
            [53, 607], [188, 557], [70, 486], [396, 524], [370, 261], [260, 455], [146, 656], [329, 688], [159, 222],
            [387, 300], [376, 163], [392, 333], [321, 669], [346, 266], [106, 642], [386, 371], [339, 556], [176, 535], [449, 384], [8, 566],
            [355, 394], [414, 529], [204, 420], [358, 315], [140, 231], [113, 703], [33, 337], [86, 371], [35, 498], [134, 645],
            [44, 635], [434, 451], [380, 334], [414, 440], [438, 232], [293, 376], [336, 346], [246, 131], [264, 629],
            [383, 651], [272, 657], ]

def get_bg(image_path:str="bg.png"):
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return Image.new("RGB", (image.shape[1], image.shape[0]), color=0)

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

    # data = get_test_data()

    # list -> np.ndarray
    data = np.array(data)
    bg_image = get_bg()

    starttime = datetime.datetime.now()

    # start painting
    for i in range(5):
        hm = HeatMap(data,)
        doHeatmap(hm, base=bg_image)

    endtime = datetime.datetime.now()
    print(f"It costs {(endtime - starttime).seconds}s.")


def doHeatmap(hm, base):
    # hm.heatmap(save_as="heat.png", base=base, r=30)
    hm.heatmap(save_as=None, base=base, r=30)  # Don't save


if __name__ == "__main__":
    main()
