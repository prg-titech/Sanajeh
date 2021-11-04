from __future__ import annotations

import os, time, math, pygame, random
from sanajeh import DeviceAllocator

kSeed: int = 42
kMaxMass: float = 500
kDt: float = 0.01
kGravityConstant: float = 4e-6
kDampeningFactor: float = 0.05
kMergeThreshold: float = 0.1
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
  
  def __init__(self):
    self.merge_target_ref: Body = None
    self.pos: Vector = Vector(0,0)
    self.vel: Vector = Vector(0,0)
    self.force: Vector = Vector(0,0)
    self.mass: float = 0
    self.has_incoming_merge: bool = False
    self.successful_merge: bool = False

  def Body(self, idx: int):
    random.seed(idx)
    self.merge_target_ref = None
    self.pos = Vector(2.0 * random.uniform(0,1) - 1.0,
                      2.0 * random.uniform(0,1) - 1.0)
    self.vel = Vector((random.uniform(0,1) - 0.5) / 1000,
                      (random.uniform(0,1) - 0.5) / 1000)
    self.force = Vector(0.0, 0.0)
    self.mass = (random.uniform(0,1)/2 + 0.5) * kMaxMass
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

  def body_update(self):
    self.vel.add(self.force.multiply(kTimeInterval).divide(self.mass))
    self.pos.add(self.vel.multiply(kTimeInterval))

    if self.pos.x < -1 or self.pos.x > 1:
      self.vel.x = -self.vel.x
    if self.pos.y < -1 or self.pos.y > 1:
      self.vel.y = -self.vel.y

  def check_merge_into_this(self, other: Body):
    if not other.has_incoming_merge and self.mass > other.mass:
      d: Vector = self.pos.minus(other.pos)
      dist_square: float = d.dist_origin()
      if dist_square < kMergeThreshold*kMergeThreshold:
        self.merge_target_ref = other
        other.has_incoming_merge = True

  def initialize_merge(self):
    self.merge_target_ref = None
    self.has_incoming_merge = False
    self.successful_merge = False

  def prepare_merge(self):
    DeviceAllocator.device_do(Body, Body.check_merge_into_this, self)

  def update_merge(self):
    m: Body = self.merge_target_ref
    if m is not None:
      if m.merge_target_ref is not None:
        new_mass: float = self.mass + m.mass
        new_vel: Vector = self.vel.multiply(self.mass).plus(m.vel.multiply(m.mass)).divide(new_mass)
        m.mass = new_mass
        m.vel = new_vel
        m.pos = self.pos.add(m.pos).divide(2)
        self.successful_merge = True

  def delete_merged(self):
    if self.successful_merge:
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

def main(allocator, do_render):
  """
  Rendering settings
  """

  def render(b):
    size = pow(b.mass/5, 0.125)
    px = int((b.pos.x/2 + 0.5) * 500 - size/2)
    py = int((b.pos.y/2 + 0.5) * 500 - size/2)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), size)  

  if (do_render):
    os.environ["SDL_VIDEODRIVER"] = "windib"
    screen_width = 500
    screen_height = 500
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.flip()  

  num: int = 100
  iter: int = 5000

  allocator.initialize()
  initialize_time = time.perf_counter()

  allocator.parallel_new(Body, num)
  parallel_new_time = time.perf_counter()

  for x in range(iter):
    allocator.parallel_do(Body, Body.compute_force)
    allocator.parallel_do(Body, Body.body_update)
    allocator.parallel_do(Body, Body.initialize_merge)
    allocator.parallel_do(Body, Body.prepare_merge)
    allocator.parallel_do(Body, Body.update_merge)
    allocator.parallel_do(Body, Body.delete_merged)
    if (do_render):
      allocator.do_all(Body, render)
      pygame.display.flip()
      screen.fill((0, 0, 0)) 
  end_time = time.perf_counter()

  print("parallel new time(%-5d objects): %.dµs" % (num, ((parallel_new_time - initialize_time) * 1000000)))
  print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / iter))
  print("overall computation time(%-4d iterations): %dµs" % (iter, ((end_time - parallel_new_time) * 1000000)))