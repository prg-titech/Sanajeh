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

class Body:
    target: Body
    pos_REF: Vector
    vel: Vector
    force: Vector
    mass: float

    def __init__(self):
        pass
    
    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos_REF = Vector(2.0 * DeviceAllocator.rand_uniform() - 1.0,
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


def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)


def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)

"""
Normalizer
# Split multiple access into different variables

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

class Body():
    target: Body
    pos: Vector
    vel: Vector
    force: Vector
    mass: float

    def __init__(self):
        pass

    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos = Vector(((2.0 * DeviceAllocator.rand_uniform()) - 1.0), ((2.0 * DeviceAllocator.rand_uniform()) - 1.0))
        self.vel = Vector(0.0, 0.0)
        self.force = Vector(0.0, 0.0)
        self.mass = (((DeviceAllocator.rand_uniform() / 2.0) + 0.5) * kMaxMass)

    def compute_force(self):
        self.force.to_zero()
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
        self.vel.add(__auto_v1)
        __auto_v2: Vector = self.vel.multiply(kDt)
        self.pos.add(__auto_v2)
        if ((self.pos.x < (- 1)) or (self.pos.x > 1)):
            self.vel.x = (- self.vel.x)
        if ((self.pos.y < (- 1)) or (self.pos.y > 1)):
            self.vel.y = (- self.vel.y)

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
"""

"""
Inliner

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

class Body():
    target: Body
    pos: Vector
    vel: Vector
    force: Vector
    mass: float

    def __init__(self):
        pass

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
            d: Vector = Vector((self.pos.x - other.pos.x), (self.pos.y - other.pos.y))
            dist: float = math.sqrt(((d.x * d.x) + (d.y * d.y)))
            f: float = (((kGravityConstant * self.mass) * other.mass) / ((dist * dist) + kDampeningFactor))
            __auto_v0: Vector = Vector((d.x * f), (d.y * f))
            __auto_v1: Vector = Vector((__auto_v0.x / dist), (__auto_v0.y / dist))
            other.force.x += __auto_v1.x
            other.force.y += __auto_v1.y

    def body_update(self):
        __auto_v0: Vector = Vector((self.force.x * kDt), (self.force.y * kDt))
        __auto_v1: Vector = Vector((__auto_v0.x / self.mass), (__auto_v0.y / self.mass))
        self.vel.x += __auto_v1.x
        self.vel.y += __auto_v1.y
        __auto_v2: Vector = Vector((self.vel.x * kDt), (self.vel.y * kDt))
        self.pos.x += __auto_v2.x
        self.pos.y += __auto_v2.y
        if ((self.pos.x < (- 1)) or (self.pos.x > 1)):
            self.vel.x = (- self.vel.x)
        if ((self.pos.y < (- 1)) or (self.pos.y > 1)):
            self.vel.y = (- self.vel.y)

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
"""

"""
Eliminator

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

class Body():
    target: Body
    pos: Vector
    vel: Vector
    force: Vector
    mass: float

    def __init__(self):
        pass

    def Body(self, idx: int):
        DeviceAllocator.rand_init(kSeed, idx, 0)
        self.pos.x: float = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.pos.y: float = ((2.0 * DeviceAllocator.rand_uniform()) - 1.0)
        self.vel.x: float = 0.0
        self.vel.y: float = 0.0
        self.force.x: float = 0.0
        self.force.y: float = 0.0
        self.mass = (((DeviceAllocator.rand_uniform() / 2.0) + 0.5) * kMaxMass)

    def compute_force(self):
        self.force.x = 0.0
        self.force.y = 0.0
        DeviceAllocator.device_do(Body, Body.apply_force, self)

    def apply_force(self, other: Body):
        if (other is not self):
            d.x: float = (self.pos.x - other.pos.x)
            d.y: float = (self.pos.y - other.pos.y)
            dist: float = math.sqrt(((d.x * d.x) + (d.y * d.y)))
            f: float = (((kGravityConstant * self.mass) * other.mass) / ((dist * dist) + kDampeningFactor))
            __auto_v0.x: float = (d.x * f)
            __auto_v0.y: float = (d.y * f)
            __auto_v1.x: float = (__auto_v0.x / dist)
            __auto_v1.y: float = (__auto_v0.y / dist)
            other.force.x += __auto_v1.x
            other.force.y += __auto_v1.y

    def body_update(self):
        __auto_v0.x: float = (self.force.x * kDt)
        __auto_v0.y: float = (self.force.y * kDt)
        __auto_v1.x: float = (__auto_v0.x / self.mass)
        __auto_v1.y: float = (__auto_v0.y / self.mass)
        self.vel.x += __auto_v1.x
        self.vel.y += __auto_v1.y
        __auto_v2.x: float = (self.vel.x * kDt)
        __auto_v2.y: float = (self.vel.y * kDt)
        self.pos.x += __auto_v2.x
        self.pos.y += __auto_v2.y
        if ((self.pos.x < (- 1)) or (self.pos.x > 1)):
            self.vel.x = (- self.vel.x)
        if ((self.pos.y < (- 1)) or (self.pos.y > 1)):
            self.vel.y = (- self.vel.y)

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
"""

"""
FieldSynthesizer

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

class Body():
    target: Body
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    force_x: float
    force_y: float
    mass: float

    def __init__(self):
        pass

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

def kernel_initialize_bodies():
    DeviceAllocator.device_class(Body)

def _update():
    DeviceAllocator.parallel_do(Body, Body.compute_force)
    DeviceAllocator.parallel_do(Body, Body.body_update)
"""