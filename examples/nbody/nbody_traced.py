from __future__ import annotations

import os, math, time
import pygame
from sanajeh import DeviceAllocator

kSeed: int = 45
kMaxMass: float = 1000.0 
kDt: float = 0.01
kGravityConstant: float = 4e-6
kDampeningFactor: float = 0.05 
 
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
    self.pos: Vector = Vector(0, 0)
    self.vel: Vector = Vector(0, 0)
    self.force: Vector = Vector(0, 0)
    self.mass: float = 0

  def Body(self, idx: int):
    DeviceAllocator.curand_init(kSeed, idx, 0)
    self.pos = Vector(2.0 * DeviceAllocator.curand_uniform() - 1.0,
                      2.0 * DeviceAllocator.curand_uniform() - 1.0)
    self.vel = Vector(0.0, 0.0)
    self.force = Vector(0.0, 0.0)
    self.mass = (DeviceAllocator.curand_uniform() / 2.0 + 0.5) * kMaxMass
  
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
    self.vel.add(self.force.multiply(kDt).divide(self.mass))
    self.pos.add(self.vel.multiply(kDt))

    if self.pos.x < -1 or self.pos.x > 1:
        self.vel.x = -self.vel.x
    if self.pos.y < -1 or self.pos.y > 1:
        self.vel.y = -self.vel.y

def main(allocator, do_render):
  """
  Rendering setting
  """
  def render(b):
    px = int((b.pos.x + 1) * 150)
    py = int((b.pos.y + 1) * 150)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), b.mass/10000*20)

  if (do_render):
    os.environ["SDL_VIDEODRIVER"] = "windib"
    screen_width = 300
    screen_height = 300
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.flip()  

  num: int = 10
  iter: int = 5000

  allocator.initialize()
  initialize_time = time.perf_counter()
  allocator.parallel_new(Body, num)
  parallel_new_time = time.perf_counter()

  for x in range(iter):
      allocator.parallel_do(Body, Body.compute_force)
      allocator.parallel_do(Body, Body.update)
      if (do_render):
        allocator.do_all(Body, render)
        pygame.display.flip()
        screen.fill((0, 0, 0))  
      end_time = time.perf_counter()

  print("parallel new time(%-5d objects): %.dµs" % (num, ((parallel_new_time - initialize_time) * 1000000)))
  print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / iter))
  print("overall computation time(%-4d iterations): %dµs" % (iter, ((end_time - parallel_new_time) * 1000000)))  


