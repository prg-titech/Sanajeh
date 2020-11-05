
from __future__ import annotations
import math
from sanajeh import DeviceAllocator
kSeed: int = 45
kMaxMass: float = 1000.0
kDt: float = 0.01
kGravityConstant: float = 4e-06
kDampeningFactor: float = 0.05

class Vector():
    x: float
    y: float

    def __init__(self, x_: float, y_: float):
        self.x = x_
        self.y = y_

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
    x: float
    y: float

    def __init__(self, x_: float, y_: float):
        self.x = x_
        self.y = y_

    def to_test2(self) -> VectorForTest2:
        return VectorForTest2(self.x, self.y)

class VectorForTest2():
    x: float
    y: float

    def __init__(self, x_: float, y_: float):
        self.x = x_
        self.y = y_

    def add(self, other: Vector) -> VectorForTest2:
        self.x += other.x
        self.y += other.y
        return self

class Body():
    pos: Vector
    vel: Vector
    force: Vector
    mass: float

    def __init__(self, px: float, py: float, vx: float, vy: float, fx: float, fy: float, m: float):
        self.pos = Vector(px, py)
        self.vel = Vector(vx, vy)
        self.force = Vector(fx, fy)
        self.mass = m

    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos = Vector(((2.0 * DeviceAllocator.rand_uniform()) - 1.0), ((2.0 * DeviceAllocator.rand_uniform()) - 1.0))
        self.vel = Vector(0.0, 0.0)
        self.force = Vector(0.0, 0.0)
        self.mass = (((DeviceAllocator.rand_uniform() / 2.0) + 0.5) * kMaxMass)

    def compute_force(self):
        self.force.x = 0.0
        self.force.y = 0.0
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if (other is not self):
            d: Vector = self.pos.minus(other.pos)
            dist: float = d.dist_origin()
            f: float = (((kGravityConstant * self.mass) * other.mass) / ((dist * dist) + kDampeningFactor))
            __auto_v0: Vector = d.multiply(f)
            __auto_v1: Vector = __auto_v0.divide(dist)
            other.force.add(__auto_v1)

    def body_update(self):
        __auto_v0: Vector = self.force.multiply(kDt)
        __auto_v1: Vector = __auto_v0.divide(self.mass)
        self.vel.x += __auto_v1.x
        self.vel.y += __auto_v1.y
        __auto_v2: Vector = self.vel.multiply(kDt)
        self.pos.x += __auto_v2.x
        self.pos.y += __auto_v2.y
        if ((self.pos.x < (- 1)) or (self.pos.x > 1)):
            self.vel.x = (- self.vel.x)
        if ((self.pos.y < (- 1)) or (self.pos.y > 1)):
            self.vel.y = (- self.vel.y)

    def test_Expr_1(self):
        __auto_v0: Vector = self.force.multiply(kDt)
        __auto_v1: Vector = __auto_v0.divide(self.mass)
        self.vel.x += __auto_v1.x
        self.vel.y += __auto_v1.y

    def test_Expr_2(self):
        __auto_v0: Vector = self.force.scale(kDt)
        __auto_v1: Vector = __auto_v0.divide_by(self.mass)
        __auto_v2: Vector = __auto_v1.to_zero()
        self.vel.x -= __auto_v2.x
        self.vel.y -= __auto_v2.y

    def test_Expr_3(self):
        self.vel.x -= self.force.x
        self.vel.y -= self.force.y

    def test_Expr_4(self, other: Body):
        __auto_v0: Vector = other.vel.subtract(self.force)
        __auto_v0.x += self.force.x
        __auto_v0.y += self.force.y

    def test_annotation(self, other: Body):
        __auto_v0: VectorForTest1 = self.vel.to_test1()
        __auto_v1: VectorForTest2 = __auto_v0.to_test2()
        __auto_v1.x += other.vel.x
        __auto_v1.y += other.vel.y

    def test_Assign(self):
        __auto_v0: Vector = self.force.scale(kDt)
        __auto_v1: Vector = self.force.divide(self.mass)
        __auto_v2: Vector = __auto_v0.add(__auto_v1)
        a = self.vel.add(__auto_v2)

    def test_AnnAssign(self):
        __auto_v0: Vector = self.vel.add(self.force)
        a: Vector = __auto_v0.minus(self.vel)
        a.x -= self.force.x
        a.y -= self.force.y

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
