import math
from AllocatorProto import *
from Config import *
from typing import TypeVar
from typing import List
import random
import time
import matplotlib.pyplot as plt
from matplotlib import animation

# from __future__
# import annotations

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
Body = TypeVar("Body")
pAT = PyAllocatorT(kNumObjects, Body)  # Allocatorを宣言する(devide_do)
py_allocator_handle = PyAllocatorTHandle(pAT)  # Host側のAllocatorのHandleを作る(parallel_do)


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
        pAT.device_do(Body, Body.apply_force)
        # -----------------------------------------------------------------------------------------------------------

    def apply_force(self, other: Body):
        if other is not self:
            dx: float = self.pos_x - other.pos_x
            dy: float = self.pos_x - other.pos_y
            dist: float = math.sqrt(dx * dx + dy * dy)
            f: float = kGravityConstant * self.mass * other.mass \
                       / (dist * dist + kDampeningFactor)

            other.force_x += f * dx / dist
            other.force_y += f * dy / dist

    def body_update(self):
        self.vel_x += self.force_x * kDt / self.mass
        self.vel_y += self.force_y * kDt / self.mass
        self.pos_x += self.vel_x * kDt
        self.pos_y += self.vel_y * kDt

        if self.pos_x < -1 or self.pos_x > 1:
            self.vel_x = -self.vel_x

        if self.pos_y < -1 or self.pos_y > 1:
            self.vel_y = -self.vel_y

    # この関数はcudaを使う必要がある
    # __global__ void kernel_initialize_bodies() {
    # int tid = threadIdx.x + blockDim.x * blockIdx.x;
    # curandState rand_state;
    # curand_init(kSeed, tid, 0, &rand_state);
    #
    # for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
    #     new(device_allocator) Body(
    #         /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
    #         /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
    #         /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
    #         /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
    #         /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);

    def kernel_initialize_bodies(self):
        pass


if __name__ == '__main__':
    # cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
    #                     cudaMemcpyHostToDevice);

    # cudaカーネルを呼び出し、Bodyのobjectを初期化する
    # kernel_initialize_bodies << < 128, 128 >> > ();
    # gpuErrchk(cudaDeviceSynchronize());

    body_list: List[Body] = []
    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)

    for i in range(kNumBodies):
        px_ = 2.0 * random.random() - 1.0
        py_ = 2.0 * random.random() - 1.0
        vx_ = (random.random() - 0.5) / 1000.0
        vy_ = (random.random() - 0.5) / 1000.0
        ms_ = (random.random() / 2.0 + 0.5) * kMaxMass
        body_list.append(Body(px_, py_, vx_, vy_, ms_))
    # for i in range(10):
    #     print(body_list[i].vel_y)

    start_time = time.time()

    # 直列
    for i in range(kNumIterations):
        for j in range(kNumBodies):
            body_list[j].compute_force()
        for j in range(kNumBodies):
            for k in range(kNumBodies):
                body_list[j].apply_force(body_list[k])
        for j in range(kNumBodies):
            body_list[j].body_update()
            plt.scatter(body_list[j].pos_x, body_list[j].pos_y, color="k")
        # print(body_list[1].force_x)
        plt.pause(0.0001)
        plt.clf()
        plt.axis([-1, 1, -1, 1], frameon=False, aspect=1)

    # 並列
    # for i in range(kNumIterations):
    #     # parallel_doを呼び出してforceを計算し、適用する---------------------------------------------------------------
    #     # allocator_handle->parallel_do < Body, & Body::compute_force > ();
    #     # allocator_handle->parallel_do < Body, & Body::update > ();
    #     py_allocator_handle.parallel_do(Body, Body.compute_force)
    #     py_allocator_handle.parallel_do(Body, Body.body_update)
    #     # -----------------------------------------------------------------------------------------------------------

    end_time = time.time()

    print("実行時間は%.2f秒" % (end_time - start_time))
