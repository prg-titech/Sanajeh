# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from sanajeh import __pyallocator__
import random
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import HTMLWriter
from benchmarks.nbody import Body

kNumIterations: int = 3000
kNumBodies: int = 30
inx = 0


# __ global__ in cuda
def kernel_initialize_bodies():
    __pyallocator__.parallel_new(Body, 3000)


def _update(frame):
    start_time = time.time()
    global inx
    # 現在のグラフを消去する
    plt.cla()
    __pyallocator__.parallel_do(Body, Body.compute_force)
    __pyallocator__.parallel_do(Body, Body.body_update)
    start_time_r = time.time()
    for j in range(kNumBodies):
        x = __pyallocator__.classDictionary["Body"][j].pos_x
        y = __pyallocator__.classDictionary["Body"][j].pos_y
        plt.scatter(x, y, color='k')
    inx += 1
    end_time = time.time()
    print("ループ%-4d実行時間は%.3f秒 描画時間は%.3f秒" % (inx, end_time - start_time, end_time - start_time_r))
    plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)


if __name__ == '__main__':

    kernel_initialize_bodies()
    fig = plt.figure(figsize=(5, 5))
    plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)

    params = {
        'fig': fig,
        'func': _update,  # グラフを更新する関数
        # 'fargs': (),  # 関数の引数 (フレーム番号を除く)
        'interval': 10,  # 更新間隔 (ミリ秒)
        'frames': np.arange(0, 10, 0.1),  # フレーム番号を生成するイテレータ
        'repeat': False,  # 繰り返す
    }
    anime = animation.FuncAnimation(**params)
    writer = animation.HTMLWriter()
    anime.save('output.html', writer=writer)

