from __future__ import annotations

import numpy as np

from benchmarks.nbodyclass import *
from sanajeh import __pyallocator__
import random
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import HTMLWriter

kNumIterations: int = 3000
kNumBodies: int = 30
inx = 0


# __ global__ in cuda
def kernel_initialize_bodies():
    for x in range(kNumBodies):
        px_ = 2.0 * random.random() - 1.0
        py_ = 2.0 * random.random() - 1.0
        vx_ = (random.random() - 0.5) / 1000.0
        vy_ = (random.random() - 0.5) / 1000.0
        ms_ = (random.random() / 2.0 + 0.5) * kMaxMass
        __pyallocator__.new_(Body, px_, py_, vx_, vy_, ms_)


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
    # cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    #                     cudaMemcpyHostToDevice);

    # cudaカーネルを呼び出し、Bodyのobjectを初期化する
    # kernel_initialize_bodies << < 128, 128 >> > ();
    # gpuErrchk(cudaDeviceSynchronize());

    kernel_initialize_bodies()
    fig = plt.figure(figsize=(5, 5))
    plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)
    # print(__pyallocator__.classDictionary)

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

