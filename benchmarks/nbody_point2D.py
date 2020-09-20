from __future__ import annotations

import math
from sanajeh import DeviceAllocator


kSeed: int = 45  # device
kMaxMass: float = 1000.0  # device
kDt: float = 0.01  # device
kGravityConstant: float = 4e-6  # device
kDampeningFactor: float = 0.05  # device


class Point2D:
    x: float
    y: float

    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

    def dist(self, other: Point2D) -> float:
        dx: float = self.dist_x(other)
        dy: float = self.dist_y(other)
        return math.sqrt(dx * dx + dy * dy)

    def dist_x(self, other: Point2D) -> float:
        return self.x - other.x

    def dist_y(self, other: Point2D) -> float:
        return self.y - other.y

    def move(self, vx, vy, t):
        self.x += vx * t
        self.y += vy * t

    def is_out_x(self, r):
        if self.x < -r or self.x > r:
            return True
        else:
            return False

    def is_out_y(self, r):
        if self.y < -r or self.y > r:
            return True
        else:
            return False


class Body:
    pos: Point2D
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float

    def __init__(self, px: float, py: float, vx: float, vy: float, fx: float, fy: float, m: float):
        self.pos = Point2D(px, py)
        self.vel_x = vx
        self.vel_y = vy
        self.force_x = fx
        self.force_y = fy
        self.mass = m

    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos = Point2D(2.0 * DeviceAllocator.rand_uniform() - 1.0,
                           2.0 * DeviceAllocator.rand_uniform() - 1.0)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.mass = (DeviceAllocator.rand_uniform() / 2.0 + 0.5) * kMaxMass
        self.force_x = 0.0
        self.force_y = 0.0

    def compute_force(self):
        self.force_x = 0.0
        self.force_y = 0.0
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if other is not self:
            d = self.pos.dist(other.pos)
            f: float = kGravityConstant * self.mass * other.mass / (d * d + kDampeningFactor)
            other.force_x += f * self.pos.dist_x(other.pos) / d
            other.force_y += f * self.pos.dist_y(other.pos) / d

    def body_update(self):

        self.vel_x += self.force_x * kDt / self.mass
        self.vel_y += self.force_y * kDt / self.mass
        self.pos.move(self.vel_x, self.vel_y, kDt)
        if self.pos.is_out_x(1):
            self.vel_x = -self.vel_x

        if self.pos.is_out_y(1):
            self.vel_y = -self.vel_y


def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)


def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)


