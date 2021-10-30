from __future__ import annotations

import os, sys, time, pygame
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import DeviceAllocator

kSeed: int = 42
kSpawnThreshold: int = 4
kOptionSharkDie: bool = True
kOptionFishSpawn: bool = True
kOptionSharkSpawn: bool = True
kEnergyBoost: int = 4
kEnergyStart: int = 2
kSizeX: int = 100
kSizeY: int = 100
kNumIterations: int = 500

"""
Device definitions
"""

class Cell:
  def __init__(self):
    self.neighbors_: list[Cell] = [None]*4
    self.agent_ref: Agent = None
    self.neighbor_request_: list[bool] = [None]*5
    self.id: int = None
  
  def Cell(self, cell_id: int):
    DeviceAllocator.curand_init(kSeed, cell_id, 0)
    self.id = cell_id
    self.prepare()
    cells[cell_id] = self
  
  def setup(self):
    x: int = self.id % kSizeX
    y: int = self.id // kSizeX

    left: Cell = cells[y*kSizeX + x - 1] if x > 0 else cells[y*kSizeX + kSizeX - 1]
    right: Cell = cells[y*kSizeX + x + 1] if x < kSizeX - 1 else cells[y*kSizeX] 
    top: Cell = cells[(y - 1)*kSizeX + x] if y > 0 else cells[(kSizeY - 1)*kSizeX + x]
    bottom: Cell = cells[(y + 1)*kSizeX + x] if y < kSizeY - 1 else cells[x]

    cells[self.id].set_neighbors(left, top, right, bottom)

    agent_type: int = DeviceAllocator.curand() % 4
    if agent_type == 0:
      agent: Agent = DeviceAllocator.new(Fish, DeviceAllocator.curand())
      assert agent != None
      cells[self.id].enter(agent)
    elif agent_type == 1:
      agent: Agent = DeviceAllocator.new(Shark, DeviceAllocator.curand())
      assert agent != None
      cells[self.id].enter(agent)
    else:
      pass     

  def agent(self) -> Agent:
    return self.agent_ref

  def decide(self):
    if self.neighbor_request_[4]:
      self.agent_ref.set_new_position(self)
    else:
      candidates: list[int] = [None]*4
      num_candidates: int = 0

      i: int = 0
      while i < 4:
        if self.neighbor_request_[i]:
          candidates[num_candidates] = i
          num_candidates += 1
        i += 1

      if num_candidates > 0:
        selected_index: int = DeviceAllocator.curand() % num_candidates
        self.neighbors_[candidates[selected_index]].agent().set_new_position(self)
  
  def enter(self, agent: Agent):
    assert self.agent_ref == None
    assert agent != None

    self.agent_ref = agent
    agent.set_position(self)

  def has_fish(self) -> bool:
    return DeviceAllocator.type_cast(self.agent_ref, Fish)

  def has_shark(self) -> bool:
    return DeviceAllocator.type_cast(self.agent_ref, Shark)

  def is_free(self) -> bool:
    return self.agent_ref == None
  
  def kill(self):
    assert self.agent_ref != None
    DeviceAllocator.destroy(self.agent_ref)
    self.agent_ref = None
  
  def leave(self):
    assert self.agent_ref != None
    self.agent_ref = None
  
  def prepare(self):
    i: int = 0
    while i < 5:
      self.neighbor_request_[i] = False
      i += 1

  def set_neighbors(self, left: Cell, top: Cell, right: Cell, bottom: Cell):
    self.neighbors_[0] = left
    self.neighbors_[1] = top
    self.neighbors_[2] = right
    self.neighbors_[3] = bottom
  
  def request_random_fish_neighbor(self):
    # random_state?
    if not self.request_random_neighbor_has_fish(DeviceAllocator.random_state(self.agent_ref)):
      if not self.request_random_neighbor_is_free(DeviceAllocator.random_state(self.agent_ref)):
        self.neighbor_request_[4] = True

  def request_random_free_neighbor(self):
    if not self.request_random_neighbor_is_free(DeviceAllocator.random_state(self.agent_ref)):
      self.neighbor_request_[4] = True

  def request_random_neighbor_has_fish(self, random_state: DeviceAllocator.RandomState) -> bool:
    candidates: list[int] = [None]*4
    num_candidates: int = 0

    i: int = 0
    while i < 4:
      if self.neighbors_[i].has_fish():
        candidates[num_candidates] = i
        num_candidates += 1
      i += 1

    if num_candidates == 0:
      return False
    else:
      selected_index: int = DeviceAllocator.curand() % num_candidates
      selected: int = candidates[selected_index]
      neighbor_index: int = (selected + 2) % 4
      self.neighbors_[selected].neighbor_request_[neighbor_index] = True

      assert self.neighbors_[selected].neighbors_[neighbor_index] == self

      return True

  def request_random_neighbor_is_free(self, random_state: DeviceAllocator.RandomState) -> bool:
    candidates: list[int] = [None]*4
    num_candidates: int = 0
    
    i: int = 0
    while i < 4:
      if self.neighbors_[i].is_free():
        candidates[num_candidates] = i
        num_candidates += 1
      i += 1

    if num_candidates == 0:
      return False
    else:
      selected_index: int = DeviceAllocator.curand() % num_candidates
      selected: int = candidates[selected_index]
      neighbor_index: int = (selected + 2) % 4
      self.neighbors_[selected].neighbor_request_[neighbor_index] = True

      assert self.neighbors_[selected].neighbors_[neighbor_index] == self

      return True

class Agent:
  def __init__(self):
    self.position_ref: Cell = None
    self.new_position_ref: Cell = None
    self.kIsAbstract: bool = True
  
  def Agent(self, seed: int):
    DeviceAllocator.curand_init(seed, 0, 0)
  
  def position(self) -> Cell:
    return self.position_ref
  
  def set_new_position(self, new_pos: Cell):
    assert self.new_position_ref == self.position_ref
    
    self.new_position_ref = new_pos
  
  def set_position(self, cell: Cell):
    self.position_ref = cell

class Fish(Agent):
  def __init__(self):
    super().__init__()
    self.egg_timer_: int = None
    self.kIsAbstract: bool = False

  def Fish(self, seed: int):
    super().Agent(seed)
    self.egg_timer_ = seed % kSpawnThreshold

  def prepare(self):
    self.egg_timer_ += 1
    self.new_position_ref = self.position_ref

    assert self.position_ref != None
    self.position_ref.request_random_free_neighbor()

  def update(self):
    old_position: Cell = self.position_ref
    if old_position != self.new_position_ref:
      old_position.leave()
      self.new_position_ref.enter(self)

      if kOptionFishSpawn and self.egg_timer_ > kSpawnThreshold:
        new_fish: Fish = DeviceAllocator.new(Fish, DeviceAllocator.curand())
        assert new_fish != None
        old_position.enter(new_fish)
        self.egg_timer_ = 0

class Shark(Agent):
  def __init__(self):
    super().__init__()
    self.egg_timer_: int = None
    self.energy_: int = None
    self.kIsAbstract: bool = False
  
  def Shark(self, seed: int):
    super().Agent(seed)
    self.energy_ = kEnergyStart
    self.egg_timer_ = seed % kSpawnThreshold

  def prepare(self):
    self.egg_timer_ += 1
    self.energy_ -= 1

    assert self.position_ref != None
    if kOptionSharkDie and self.energy_ == 0:
      pass
    else:
      self.new_position_ref = self.position_ref
      self.position_ref.request_random_fish_neighbor()

  def update(self):
    if kOptionSharkDie and self.energy_ == 0:
      self.position_ref.kill()
    else:
      old_position: Cell = self.position_ref
      if old_position != self.new_position_ref:
        if self.new_position_ref.has_fish():
          self.energy_ += kEnergyBoost
          self.new_position_ref.kill()
        
        old_position.leave()
        self.new_position_ref.enter(self)

        if kOptionFishSpawn and self.egg_timer_ > kSpawnThreshold:
          new_shark: Shark = DeviceAllocator.new(Shark, DeviceAllocator.curand())
          assert new_shark != None
          old_position.enter(new_shark)
          self.egg_timer_ = 0

# cells: list = DeviceAllocator.array(kSizeX*kSizeY)

def main(allocator, do_render):

  # Initialize render  
  window = None
  screen = None
  pxarray = None

  def initialize_render():
    pygame.init()
    window = pygame.display.set_mode((kSizeX*6, kSizeY*6))
    screen = pygame.Surface((kSizeX, kSizeY))
    screen.fill((0,0,0))
    return (window, screen)

  def render():
    for i in range(kSizeX*kSizeY):
      x = i % kSizeX
      y = i // kSizeX
      if cells[i].has_fish():
        pxarray[x,y] = pygame.Color(0,255,0)
      elif cells[i].has_shark():
        pxarray[x,y] = pygame.Color(255,0,0)
      else:
        pxarray[x,y] = pygame.Color(255,255,255)  
    window.blit(pygame.transform.scale(screen, window.get_rect().size), (0,0))
    pygame.display.update()  
  
  total_time = time.perf_counter()

  allocator.initialize()
  initialize_time = time.perf_counter()
  allocator.parallel_new(Cell, kSizeX*kSizeY)
  parallel_new_time = time.perf_counter()
  
  allocator.parallel_do(Cell, Cell.setup)

  if do_render:
    os.environ["SDL_VIDEODRIVER"] = "windib"
    screen_width, screen_height = kSizeX, kSizeY
    scaling_factor = 6  
    window, screen = initialize_render()
    pxarray = pygame.PixelArray(screen)
    render()

  for i in range(kNumIterations):
    time_before = time.perf_counter()

    # step()    
    allocator.parallel_do(Cell, Cell.prepare)
    allocator.parallel_do(Fish, Fish.prepare)
    allocator.parallel_do(Cell, Cell.decide)
    allocator.parallel_do(Fish, Fish.update)

    allocator.parallel_do(Cell, Cell.prepare)
    allocator.parallel_do(Shark, Shark.prepare)
    allocator.parallel_do(Cell, Cell.decide)
    allocator.parallel_do(Shark, Shark.update)

    time_after = time.perf_counter()
    total_time += time_after - time_before

    # No rendering in the original dynasoar, so this doesn't require do_all
    if do_render:
      render()

  print(total_time)
