from typing import List
import math
from configuration import *

# cuda_block_size = 256

# DynaSOArの初期化
"""
//何らかのシンタックスでここで使いたいクラスをすべて宣言する
class Body;
using AllocatorT = SoaAllocator<kNumObjects, Body>;

//Allocatorを宣言する
AllocatorHandle<AllocatorT>* allocator_handle;
__device__ AllocatorT* device_allocator;


"""


class Body:  # クラスをDynaSOArを使う必要があることを何らかのシンタックスで宣言すべき（DynaSOArだとAllocator::Baseの子クラスにする）
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float
    # ここでAllocatorのFieldを呼び出す
    """
    declare_field_types(Body, float, float, float, float, float, float, float);
    Field<Body, 0> pos_x_;
    Field<Body, 1> pos_y_;
    Field<Body, 2> vel_x_;
    Field<Body, 3> vel_y_;
    Field<Body, 4> force_x_;
    Field<Body, 5> force_y_;
    Field<Body, 6> mass_;
    """

    def __init__(self, px: float, py: float, vx: float, vy: float, m: float):
        self.pos_x = px
        self.pos_y = py
        self.vel_x = vx
        self.vel_y = vy
        self.mass = m

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        # ここでdevice_doを呼び出す
        """
        device_allocator->template device_do<Body>(&Body::apply_force, this);
        """

    def apply_force(self, other: Body):
        if other is not self:
            dx: float = self.pos_x - other.pos_x
            dy: float = self.pos_x - other.pos_y
            dist: float = math.sqrt(dx * dx + dy * dy)
            f: float = (kGravityConstant * self.mass * other.mass /
                        (dist * dist + kDampeningFactor))
            other.force_x += f * dx / dist
            other.force_y += f * dy / dist

    def body_update(self):
        self.vel_x += self.force_x * kDt / self.mass
        self.vel_y += self.force_y * kDt / self.mass
        self.pos_x += self.vel_x * kDt
        self.pos_y += self.vel_y * kDt

        if self.pos_x < -1 or self.pos_x > 1:
            self.vel_x = -self.vel_x

        if self.pos_y < -1 or selfpos_y > 1:
            self.vel_y = -self.vel_y

    # この関数はcudaを使う必要がある
    """
    __global__ void kernel_initialize_bodies() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    curandState rand_state;
    curand_init(kSeed, tid, 0, &rand_state);

    for (int i = tid; i < kNumBodies; i += blockDim.x * gridDim.x) {
        new(device_allocator) Body(
            /*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
            /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
            /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
            /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
            /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5) * kMaxMass);
    """

    def kernel_initialize_bodies(self):
        pass


if __name == '__main__':

    # Allocatorを作成する
    """
    allocator_handle = new AllocatorHandle<AllocatorT>(/*unified_memory=*/ true);
    AllocatorT* dev_ptr = allocator_handle->device_pointer();
    cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
                        cudaMemcpyHostToDevice);
    """

    # cudaカーネルを呼び出し、Bodyのobjectを初期化する
    """
    kernel_initialize_bodies << < 128, 128 >> > ();
    gpuErrchk(cudaDeviceSynchronize());
    """

    for i in range(kNumIterations):
        # parallel_doを呼び出してforceを計算し、適用する
        """
        allocator_handle->parallel_do < Body, & Body::compute_force > ();
        allocator_handle->parallel_do < Body, & Body::update > ();
        """

    # rendering part
