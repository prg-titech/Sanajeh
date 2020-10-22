from __future__ import annotations

import math
from sanajeh import DeviceAllocator

kSeed: int = 45  # device
kMaxMass: float = 1000.0  # device
kDt: float = 0.01  # device
kGravityConstant: float = 4e-6  # device
kDampeningFactor: float = 0.05  # device


class Vector:
    x: float
    y: float

    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

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


def to_zero(v: Vector):
    v.x = 0.0
    v.y = 0.0
    return v


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
        to_zero(self.force)
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
        self.vel.minus(to_zero(self.force.scale(kDt).divide_by(self.mass)))

    def test_Expr_3(self):
        to_zero(self.force).add(self.force.scale(kDt).divide_by(self.mass)).minus(self.force.scale(kDt).divide_by(self.mass))

    def test_Assign(self):
        a = self.vel.add(self.force.scale(kDt).add(self.force.divide(self.mass)))

    def test_AnnAssign(self):
        a: Vector = self.vel.add(self.force).minus(self.vel)


def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)


def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
