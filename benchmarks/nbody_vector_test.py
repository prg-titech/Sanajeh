from __future__ import annotations

import math
from sanajeh import DeviceAllocator

kSeed: int = 45  # device
kMaxMass: float = 1000.0  # device
kDt: float = 0.01  # device
kGravityConstant: float = 4e-6  # device
kDampeningFactor: float = 0.05  # device


class Vector:

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def add(self, other: Vector) -> Vector:
        self.x += other.x
        self.y += other.y
        return self

    def plus(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    def subtract(self, other: Vector) -> Vector:
        self.x -= other.x
        self.y -= other.y
        return self

    def minus(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    def scale(self, ratio: float) -> Vector:
        self.x *= ratio
        self.y *= ratio
        return self

    def multiply(self, multiplier: float) -> Vector:
        return Vector(self.x * multiplier, self.y * multiplier)

    def divide_by(self, divisor: float) -> Vector:
        self.x /= divisor
        self.y /= divisor
        return self

    def divide(self, divisor: float) -> Vector:
        return Vector(self.x / divisor, self.y / divisor)

    # Distance from origin
    def dist_origin(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

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
        return Vector(self.x - other.x, self.y - other.y)


class VectorForTest1:

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def to_test2(self) -> VectorForTest2:
        return VectorForTest2(self.x, self.y)

class VectorForTest2:

    def __init__(self, x_: float, y_: float):
        self.x: float = x_
        self.y: float = y_

    def add(self, other: Vector) -> VectorForTest2:
        self.x -= other.x
        self.y -= other.y
        return self

class Body:
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
        self.pos = Vector(2.0 * DeviceAllocator.rand_uniform() - 1.0,
                          2.0 * DeviceAllocator.rand_uniform() - 1.0)
        self.vel = Vector(0.0, 0.0)
        self.force = Vector(0.0, 0.0)
        self.mass = (DeviceAllocator.rand_uniform() / 2.0 + 0.5) * kMaxMass

    def compute_force(self):
        self.force.to_zero()
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if other is not self:
            d: Vector = self.pos.minus(other.pos)
            dist: float = d.dist_origin()
            f: float = kGravityConstant * self.mass * other.mass / (dist * dist + kDampeningFactor)
            other.force.add(d.multiply(f).divide(dist))

    def body_update(self):
        self.vel.add(self.force.multiply(kDt).divide(self.mass))
        # self.vel.add(self.force.scale(kDt).divide_by(self.mass))
        self.pos.add(self.vel.multiply(kDt))
        # self.pos.add(self.vel.scale(kDt))

        if self.pos.x < -1 or self.pos.x > 1:
            self.vel.x = -self.vel.x
        if self.pos.y < -1 or self.pos.y > 1:
            self.vel.y = -self.vel.y

    def test_Expr_1(self):
        self.vel.add(self.force.multiply(kDt).divide(self.mass))

    def test_Expr_2(self):
        self.vel.subtract(self.force.scale(kDt).divide_by(self.mass).to_zero())

    def test_Expr_3(self):
        self.vel.subtract(self.force)

    def test_Expr_4(self, other: Body):
        other.vel.subtract(self.force).add(self.force)

    def test_annotation(self, other: Body):
        self.vel.to_test1().to_test2().add(other.vel)

    def test_AnnAssign(self):
        a: Vector = self.vel.add(self.force).minus(self.vel)
        a.subtract(self.force)


def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)


def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
