
from __future__ import annotations
import math
from sanajeh import DeviceAllocator
kSeed: int = 45
kMaxMass: float = 1000.0
kDt: float = 0.01
kGravityConstant: float = 4e-06
kDampeningFactor: float = 0.05

class Vector():

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def add(self, other: Vector) -> Vector:
        self.x += other.x
        self.y += other.y
        return self

    def plus(self, other: Vector) -> Vector:
        return Vector((self.x + other.x), (self.y + other.y))

    def subtract(self, other: Vector) -> Vector:
        self.x -= other.x
        self.y -= other.y
        return self

    def minus(self, other: Vector) -> Vector:
        return Vector((self.x - other.x), (self.y - other.y))

    def scale(self, ratio: float) -> Vector:
        self.x *= ratio
        self.y *= ratio
        return self

    def multiply(self, multiplier: float) -> Vector:
        return Vector((self.x * multiplier), (self.y * multiplier))

    def divide_by(self, divisor: float) -> Vector:
        self.x /= divisor
        self.y /= divisor
        return self

    def divide(self, divisor: float) -> Vector:
        return Vector((self.x / divisor), (self.y / divisor))

    def dist_origin(self) -> float:
        return math.sqrt(((self.x * self.x) + (self.y * self.y)))

    def to_zero(self) -> Vector:
        self.x = 0.0
        self.y = 0.0
        return self

    def to_test1(self) -> VectorForTest1:
        return VectorForTest1(self.x, self.y)

    def add1(self, other: Vector) -> Vector:
        self.x += other.x
        self.y += other.y
        return self

    def minus1(self, other: Vector) -> Vector:
        return Vector((self.x - other.x), (self.y - other.y))

class VectorForTest1():

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def to_test2(self) -> VectorForTest2:
        return VectorForTest2(self.x, self.y)

class VectorForTest2():

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def add(self, other: Vector) -> VectorForTest2:
        self.x -= other.x
        self.y -= other.y
        return self

class Body():
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float

    def __init__(self, px: float, py: float, vx: float, vy: float, fx: float, fy: float, m: float):
        self.pos_x = px
        self.pos_y = py
        self.vel_x = vx
        self.vel_y = vy
        self.force_x = fx
        self.force_y = fy
        self.mass = m

    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos_x = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.pos_y = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.force_x = 0.0
        self.force_y = 0.0
        self.mass = (((DeviceAllocator.rand_uniform() / 2.0) + 0.5) * kMaxMass)

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if (other is not self):
            d_x: float = (self.pos_x - other.pos_x)
            d_y: float = (self.pos_y - other.pos_y)
            dist: float = math.sqrt(((d_x * d_x) + (d_y * d_y)))
            f: float = (((kGravityConstant * self.mass) * other.mass) / ((dist * dist) + kDampeningFactor))
            __auto_v0_x: float = (d_x * f)
            __auto_v0_y: float = (d_y * f)
            __auto_v1_x: float = (__auto_v0_x / dist)
            __auto_v1_y: float = (__auto_v0_y / dist)
            other.force_x += __auto_v1_x
            other.force_y += __auto_v1_y

    def body_update(self):
        __auto_v0_x: float = (self.force_x * kDt)
        __auto_v0_y: float = (self.force_y * kDt)
        __auto_v1_x: float = (__auto_v0_x / self.mass)
        __auto_v1_y: float = (__auto_v0_y / self.mass)
        self.vel_x += __auto_v1_x
        self.vel_y += __auto_v1_y
        __auto_v2_x: float = (self.vel_x * kDt)
        __auto_v2_y: float = (self.vel_y * kDt)
        self.pos_x += __auto_v2_x
        self.pos_y += __auto_v2_y
        if ((self.pos_x < (- 1)) or (self.pos_x > 1)):
            self.vel_x = (- self.vel_x)
        if ((self.pos_y < (- 1)) or (self.pos_y > 1)):
            self.vel_y = (- self.vel_y)

    def test_Expr_1(self):
        __auto_v0_x: float = (self.force_x * kDt)
        __auto_v0_y: float = (self.force_y * kDt)
        __auto_v1_x: float = (__auto_v0_x / self.mass)
        __auto_v1_y: float = (__auto_v0_y / self.mass)
        self.vel_x += __auto_v1_x
        self.vel_y += __auto_v1_y

    def test_Expr_2(self):
        self.force_x *= kDt
        self.force_y *= kDt
        self.force_x /= self.mass
        self.force_y /= self.mass
        self.force_x = 0.0
        self.force_y = 0.0
        self.vel_x -= self.force_x
        self.vel_y -= self.force_y

    def test_Expr_3(self):
        self.vel_x -= self.force_x
        self.vel_y -= self.force_y

    def test_Expr_4(self, other: Body):
        other.vel_x -= self.force_x
        other.vel_y -= self.force_y
        other.vel_x += self.force_x
        other.vel_y += self.force_y

    def test_annotation(self, other: Body):
        __auto_v0_x: float = self.vel_x
        __auto_v0_y: float = self.vel_y
        __auto_v1_x: float = __auto_v0_x
        __auto_v1_y: float = __auto_v0_y
        __auto_v1_x -= other.vel_x
        __auto_v1_y -= other.vel_y

    def test_AnnAssign(self):
        self.vel_x += self.force_x
        self.vel_y += self.force_y
        a_x: float = (self.vel_x - self.vel_x)
        a_y: float = (self.vel_y - self.vel_y)
        a_x -= self.force_x
        a_y -= self.force_y

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
