
from __future__ import annotations
import math
from sanajeh import DeviceAllocator
kNumIterations: int = 3000
kNumBodies: int = 30
inx = 0
kSeed: int = 3000
kMaxMass: float = 1000.0
kDt: float = 0.02
kGravityConstant: float = 6.673e-05
kDampeningFactor: float = 0.05

class Body():
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float

    def __init__(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos_x = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.pos_y = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.vel_x = ((DeviceAllocator.rand_uniform() - 0.5) / 1000.0)
        self.vel_y = ((DeviceAllocator.rand_uniform() - 0.5) / 1000.0)
        self.mass = (((DeviceAllocator.rand_uniform() / 2.0) + 0.5) * kMaxMass)
        self.force_x = 0.0
        self.force_y = 0.0

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if (other is not self):
            dx: float = (self.pos_x - other.pos_x)
            dy: float = (self.pos_x - other.pos_y)
            dist: float = math.sqrt(((dx * dx) + (dy * dy)))
            f: float = (((kGravityConstant * self.mass) * other.mass) / ((dist * dist) + kDampeningFactor))
            other.force_x += ((f * dx) / dist)
            other.force_y += ((f * dy) / dist)

    def body_update(self):
        self.vel_x += ((self.force_x * kDt) / self.mass)
        self.vel_y += ((self.force_y * kDt) / self.mass)
        self.pos_x += (self.vel_x * kDt)
        self.pos_y += (self.vel_y * kDt)
        if ((self.pos_x < (- 1)) or (self.pos_x > 1)):
            self.vel_x = (- self.vel_x)
        if ((self.pos_y < (- 1)) or (self.pos_y > 1)):
            self.vel_y = (- self.vel_y)

    @staticmethod
    def parallel_new(cpp_lib, object_num):
        return cpp_lib.parallel_new_Body(object_num)

    @staticmethod
    def do_all(cpp_lib, func):
        return cpp_lib.Body_do_all(func)

def specify_device_class():
    DeviceAllocator.device_class(Body)

def specify_parallel_do():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
