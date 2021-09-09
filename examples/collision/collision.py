from __future__ import annotations

import math
from sanajeh import DeviceAllocator

kSeed: int = 45  # device
kMaxMass: float = 1000.0  # device
kDt: float = 0.01  # device
kGravityConstant: float = 4e-6  # device
kDampeningFactor: float = 0.05  # device
kMergeThreshold: float = 0.005
kTimeInterval: float = 0.05

class Body:

  merge_target: Body
  pos_x: float
  pos_y: float
  vel_x: float
  vel_y: float
  force_x: float
  force_y: float
  mass: float
  has_incoming_merge: bool
  successful_merge: bool

  def __init__(self, tgt: Body, px: float, py: float, vx: float, vy: float, fx: float, fy: float,
      m: float, incoming: bool, success: bool):
    self.merge_target = tgt
    self.pos_x = px
    self.pos_y = py
    self.vel_x = vx
    self.vel_y = vy
    self.force_x = fx
    self.force_y = fy
    self.mass = m
    self.has_incoming_merge = incoming
    self.successful_merge = success

  def Body(self, idx: int):
    DeviceAllocator.rand_init(kSeed, idx, 0)
    self.merge_target = None
    self.pos_x = 2.0 * DeviceAllocator.rand_uniform() - 1.0
    self.pos_y = 2.0 * DeviceAllocator.rand_uniform() - 1.0
    self.vel_x = (DeviceAllocator.rand_uniform() - 0.5) / 1000
    self.vel_y = (DeviceAllocator.rand_uniform() - 0.5) / 1000
    self.mass = (DeviceAllocator.rand_uniform() / 2.0 + 0.5) * kMaxMass
    self.force_x = 0.0
    self.force_y = 0.0
    self.has_incoming_merge = False
    self.successful_merge = False

  def compute_force(self):
    self.force_x = 0.0
    self.force_y = 0.0
    DeviceAllocator.device_do(Body, Body.apply_force, self)

  def apply_force(self, other: Body):
    if other is not self:
      dx: float = self.pos_x - other.pos_x
      dy: float = self.pos_y - other.pos_y
      dist: float = math.sqrt(dx * dx + dy * dy)
      f: float = kGravityConstant * self.mass * other.mass / (dist * dist + kDampeningFactor)
      other.force_x += f * dx / dist
      other.force_y += f * dy / dist

  def body_update(self):
    self.vel_x += self.force_x * kTimeInterval / self.mass
    self.vel_y += self.force_y * kTimeInterval / self.mass
    self.pos_x += self.vel_x * kTimeInterval
    self.pos_y += self.vel_y * kTimeInterval
    if self.pos_x < -1 or self.pos_x > 1:
      self.vel_x = -self.vel_x
    if self.pos_y < -1 or self.pos_y > 1:
      self.vel_y = -self.vel_y

  def check_merge_into_this(self, other: Body):
    if not other.has_incoming_merge and self.mass > other.mass:
      dx: float = self.pos_x - other.pos_x
      dy: float = self.pos_y - other.pos_y
      dist_square: float = dx*dx + dy*dy
      if dist_square < kMergeThreshold*kMergeThreshold:
        self.merge_target = other
        other.has_incoming_merge = True

  def initialize_merge(self):
    self.merge_target = None
    self.has_incoming_merge = False
    self.successful_merge = False

  def prepare_merge(self):
    DeviceAllocator.device_do(Body, Body.check_merge_into_this, self)

  def update_merge(self):
    m: Body = self.merge_target
    if m is not None:
      if m.merge_target is not None:
        new_mass: float = self.mass + m.mass
        new_vel_x: float = (self.vel_x*self.mass + m.vel_x*m.mass) / new_mass
        new_vel_y: float = (self.vel_y*self.mass + m.vel_y*m.mass) / new_mass
        m.mass = new_mass
        m.vel_x = new_vel_x
        m.vel_y = new_vel_y
        m.pos_x = (self.pos_x + m.pos_x) / 2
        m.pos_y = (self.pos_y + m.pos_y) / 2

        self.successful_merge = True
  
  def delete_merged(self):
    if (self.successful_merge):
      DeviceAllocator.destroy(self)

def kernel_initialize_bodies():
  DeviceAllocator.device_class(Body)

def _update():
  DeviceAllocator.parallel_do(Body, Body.compute_force)
  DeviceAllocator.parallel_do(Body, Body.body_update)
  DeviceAllocator.parallel_do(Body, Body.initialize_merge)
  DeviceAllocator.parallel_do(Body, Body.prepare_merge)
  DeviceAllocator.parallel_do(Body, Body.update_merge)
  DeviceAllocator.parallel_do(Body, Body.delete_merged)