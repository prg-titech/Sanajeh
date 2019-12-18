from __future__ import annotations

from benchmarks.nbodyclass import *
from sanajeh import __pyallocator__
import random
import time
import matplotlib.pyplot as plt


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


kSeed: int = 42
kMaxMass: float = 1000.0
kNumIterations: int = 3000
kNumBodies: int = 30
kDt: float = 0.02
kGravityConstant: float = 6.673e-4
kDampeningFactor: float = 0.05
kNumObjects: int = 64 * 64 * 64 * 64


# __ global__ in cuda
def kernel_initialize_bodies():
    for x in range(kNumBodies):
        px_ = 2.0 * random.random() - 1.0
        py_ = 2.0 * random.random() - 1.0
        vx_ = (random.random() - 0.5) / 1000.0
        vy_ = (random.random() - 0.5) / 1000.0
        ms_ = (random.random() / 2.0 + 0.5) * kMaxMass
        __pyallocator__.new_(Body, px_, py_, vx_, vy_, ms_)


if __name__ == '__main__':
    # cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    #                     cudaMemcpyHostToDevice);

    # cudaカーネルを呼び出し、Bodyのobjectを初期化する
    # kernel_initialize_bodies << < 128, 128 >> > ();
    # gpuErrchk(cudaDeviceSynchronize());

    kernel_initialize_bodies()
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)
    # print(__pyallocator__.classDictionary)

    # 並列
    for i in range(kNumIterations):
        start_time = time.time()
        # parallel_doを呼び出してforceを計算し、適用する---------------------------------------------------------------
        # allocator_handle->parallel_do < Body, & Body::compute_force > ();
        # allocator_handle->parallel_do < Body, & Body::update > ();
        __pyallocator__.parallel_do(Body, Body.compute_force)
        __pyallocator__.parallel_do(Body, Body.body_update)

        for j in range(kNumBodies):
            plt.scatter(__pyallocator__.classDictionary["Body"][j].pos_x,
                        __pyallocator__.classDictionary["Body"][j].pos_y,
                        color='k'
                        )
        # ---------------------------------------------------------------------------------------------------------
        plt.pause(0.00001)
        plt.clf()
        plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)
        end_time = time.time()
        print("ループ%-4d実行時間は%.2f秒" % (i, (end_time - start_time)))
