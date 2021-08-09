from __future__ import annotations
from sanajeh import DeviceAllocator

kSeed: int = 45
kMaxMass: float = 1000.0
kDt: float = 0.01
kGravityConstant: float = 4e-6
kDampeningFactor: float = 0.05
kMergeThreshold: float = 0.005
kTimeInterval: float = 0.05

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

class Body:
  merge_target: Body
  pos: Vector
  vel: Vector
  force: Vector
  mass: float
  has_incoming_merge: bool
  successful_merge: bool

  def __init__(self, merge_target_: Body, pos_x_: float, pos_y_: float, vel_x_: float, vel_y_: float,
      force_x_: float, force_y_: float, mass_: float, has_incoming_merge_: bool, successful_merge_: bool):
    self.merge_target = None
    self.pos = Vector(pos_x_, pos_y_)
    self.vel = Vector(vel_x_, vel_y_)
    self.force = Vector(0.0, 0.0)
    self.mass = mass_
    self.has_incoming_merge = False
    self.successful_merge = False

  def Body(self, idx: int):
    DeviceAllocator.rand_init(kSeed, idx, 0)
    self.merge_target = None
    self.pos = Vector(2.0 * DeviceAllocator.rand_uniform() - 1.0,
                      2.0 * DeviceAllocator.rand_uniform() - 1.0)
    self.vel = Vector((DeviceAllocator.rand_uniform() - 0.5) / 1000,
                      (DeviceAllocator.rand_uniform() - 0.5) / 1000)
    self.force = Vector(0.0, 0.0)
    self.mass = (DeviceAllocator.rand_uniform()/2 + 0.5) * kMaxMass
    self.has_incoming_merge = False
    self.successful_merge = False

  def compute_force(self):
    self.force.to_zero()
    DeviceAllocator.device_do(Body, Body.apply_force, self)

  def apply_force(self, other: Body):
    if other is not self:
      d: Vector = self.pos.minus(other.pos)
      dist: float = d.dist_origin()
      f: float = kGravityConstant * self.mass * other.mass / (dist * dist + kDampeningFactor)
      other.force.add(d.multiply(f).divide(dist))

  def update(self):
    self.vel.add(self.force.multiply(kTimeInterval).divide(self.mass))
    # self.vel.add(self.force.scale(kDt).divide_by(self.mass))
    self.pos.add(self.vel.multiply(kTimeInterval))
    # self.pos.add(self.vel.scale(kDt))

    if self.pos.x < -1 or self.pos.x > 1:
      self.vel.x = -self.vel.x
    if self.pos.y < -1 or self.pos.y > 1:
      self.vel.y = -self.vel.y

  def check_merge_into_this(self, other: Body):
    if not other.has_incoming_merge and self.mass > other.mass:
      d: Vector = self.pos.minus(other.pos)
      dist_square: float = d.dist_origin()
      if dist_square < kMergeThreshold*kMergeThreshold:
        self.merge_target = other
        other.has_incoming_merge = True

  def initialize_merge():
    self.merge_target = None
    self.has_incoming_merge = False
    self.successful_merge = False

  def prepare_merge():
    DeviceAllocator.device_do(Body, Body.check_merge_into_this, self)

  def update_merge():
    m: Body = self.merge_target
    if m is not None:
      if m.merge_target is not None:
        new_mass: float = self.mass + m.mass
        new_vel: Vector = self.vel.multiply(self.mass).plus(m.vel.multiply(m.mass)).divide(new_mass)
        m.mass = new_mass
        m.vel = new_vel
        m.pos = self.pos.add(m.pos).divide(2)
        self.successful_merge = True

  def delete_merged():
    if self.successful_merge:
      DeviceAllocator.destroy(self)

def kernel_initialize_bodies():
  DeviceAllocator.device_class(Body)