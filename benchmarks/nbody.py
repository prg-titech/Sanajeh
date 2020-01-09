from __future__ import annotations

import numpy as np

import math
from sanajeh import __pyallocator__
import random
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import HTMLWriter

kNumIterations: int = 3000
kNumBodies: int = 30
inx = 0

kMaxMass: float = 1000.0
kDt: float = 0.02
kGravityConstant: float = 6.673e-5
kDampeningFactor: float = 0.05


# DynaSOArの初期化----------------------------------------------------------------------------------------------------
# //何らかのシンタックスでここで使いたいクラスをすべて宣言する
# class Body;
# using AllocatorT = SoaAllocator<kNumObjects, Body>;
#
# //Allocatorを宣言する
# AllocatorHandle<AllocatorT>* allocator_handle;
# __device__ AllocatorT* device_allocator;
# Allocatorを作成する
# allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
# AllocatorT* dev_ptr = allocator_handle->device_pointer();
# cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);


# -------------------------------------------------------------------------------------------------------------------

class Body:  # クラスをDynaSOArを使う必要があることを何らかのシンタックスで宣言すべき（DynaSOArだとAllocator::Baseの子クラスにする）

    # ここでAllocatorのFieldを呼び出す----------------------------------------------------------------------------------
    # declare_field_types(Body, float, float, float, float, float, float, float);
    # Field<Body, 0> pos_x_;
    # Field<Body, 1> pos_y_;
    # Field<Body, 2> vel_x_;
    # Field<Body, 3> vel_y_;
    # Field<Body, 4> force_x_;
    # Field<Body, 5> force_y_;
    # Field<Body, 6> mass_;
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float
    # ---------------------------------------------------------------------------------------------------------------

    def __init__(self, px: float, py: float, vx: float, vy: float, m: float):
        self.pos_x = px
        self.pos_y = py
        self.vel_x = vx
        self.vel_y = vy
        self.mass = m
        self.force_x = 0.0
        self.force_y = 0.0

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        # ここでdevice_doを呼び出す-------------------------------------------------------------------------------------
        # device_allocator->template device_do<Body>(&Body::apply_force, this);
        __pyallocator__.device_do(Body, Body.apply_force, self)
        # -----------------------------------------------------------------------------------------------------------

    def apply_force(self, other: Body):
        if other is not self:
            dx: float = self.pos_x - other.pos_x
            dy: float = self.pos_x - other.pos_y
            dist: float = math.sqrt(dx * dx + dy * dy)
            f: float = kGravityConstant * self.mass * other.mass / (dist * dist + kDampeningFactor)

            other.force_x += f * dx / dist
            other.force_y += f * dy / dist

    def body_update(self):

        """
        test code
        """
        dy: int = kNumBodies

        self.vel_x += self.force_x * kDt / self.mass
        self.vel_y += self.force_y * kDt / self.mass
        self.pos_x += self.vel_x * kDt
        self.pos_y += self.vel_y * kDt

        if self.pos_x < -1 or self.pos_x > 1:
            self.vel_x = -self.vel_x

        if self.pos_y < -1 or self.pos_y > 1:
            self.vel_y = -self.vel_y


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

