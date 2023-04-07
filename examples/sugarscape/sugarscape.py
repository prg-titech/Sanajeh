from __future__ import annotations

import os, sys, time, random, pygame

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/new-src")
from sanajeh import DeviceAllocator, device
from typing import cast

kSize: int = 100
kNumObjects: int = 262144
kSeed: int = 42

kProbMale: float = 0.12
kProbFemale: float = 0.15

kNumIterations: int = 100
kMaxVision: int = 2
kMaxAge: int = 100
kMaxEndowment: int = 200
kMaxMetabolism: int = 80
kSugarCapacity: int = 3500
kMaxSugarDiffusion: int = 60
kSugarDiffusionRate: float = 0.125
kMinMatingAge: int = 22
kMaxChildren: int = 8

cells: list[Cell] = DeviceAllocator.array(kSize*kSize)

class Cell:
  def __init__(self):
    self.agent_ref: Agent = None
    self.sugar_diffusion_: int = 0
    self.sugar_: int = 0
    self.sugar_capacity_: int = 0
    self.grow_rate_: int = 0
    self.cell_id_: int = None
    self.agent_type_: int = 0

  def Cell(self, cell_id: int):
    self.agent_ref = None
    self.sugar_ = 0
    self.sugar_capacity_ = 3500
    self.cell_id_ = cell_id

    random.seed(cell_id)
    r: float = random.uniform(0,1)
    if r <= 0.02:
      self.grow_rate_ = 50
    elif r <= 0.04:
      self.grow_rate_ = 0.5*50
    elif r <= 0.08:
      self.grow_rate_ = 0.25*50
    else:
      self.grow_rate_ = 0

    cells[cell_id] = self

  def prepare_diffuse(self):
    self.sugar_diffusion_ = kSugarDiffusionRate * self.sugar_
    max_diff: int = kMaxSugarDiffusion
    if self.sugar_diffusion_ > max_diff:
      self.sugar_diffusion_ = max_diff
    self.sugar_ -= self.sugar_diffusion_
  
  def update_diffuse(self):
    new_sugar: int = 0
    this_x: int = self.cell_id_ % kSize
    this_y: int = self.cell_id_ // kSize

    for dx in range(-kMaxVision, kMaxVision+1, 1):
      for dy in range(-kMaxVision, kMaxVision+1, 1):
        nx: int = this_x + dx
        ny: int = this_y + dy
        if (dx != 0 or dy != 0) and nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
          n_id: int = nx + ny*kSize
          n_cell: Cell = cells[n_id]

          new_sugar += 0.125 * n_cell.sugar_diffusion_

    self.sugar_ += new_sugar
  
  def cell_id(self) -> int:
    return self.cell_id_
  
  def random_float(self) -> float:
    return random.uniform(0,1)

  def random_int(self, a: int, b: int) -> int:
    return random.getrandbits(32) % (b - a) + a
  
  def decide_permission(self):
    selected_agent: Agent = None
    turn: int = 0
    this_x: int = self.cell_id_ % kSize
    this_y: int = self.cell_id_ // kSize

    for dx in range(-kMaxVision, kMaxVision+1, 1):
      for dy in range(-kMaxVision, kMaxVision+1, 1):
        nx: int = this_x + dx
        ny: int = this_y + dy
        if nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
          n_id: int = nx + ny*kSize
          n_cell: Cell = cells[n_id]
          n_agent: Agent = n_cell.agent()
          if n_agent != None and n_agent.cell_request() == self:
            turn += 1
            
            if self.random_float() <= 1.0/turn:
              selected_agent = n_agent
            else:
              assert turn > 1

    assert (turn == 0) == (selected_agent == None)

    if selected_agent != None:
      selected_agent.give_permission()
  
  def is_free(self) -> bool:
    return self.agent_ref == None
  
  def enter(self, agent: Agent):
    assert self.agent_ref == None
    assert agent != None
    self.agent_ref = agent
    if type(agent) == Male:
        self.agent_type_ = 1
    if type(agent) == Female:
        self.agent_type_ = 2
  
  def sugar(self) -> int:
    return self.sugar_
  
  def take_sugar(self, amount: int):
    self.sugar_ -= amount
  
  def grow_sugar(self):
    self.sugar_ += min(self.sugar_capacity_ - self.sugar_, self.grow_rate_)

  def agent(self) -> Agent:
    return self.agent_ref

  def setup(self):
    r: float = self.random_float()
    c_vision: int = kMaxVision//2 + self.random_int(0, kMaxVision//2)
    c_max_age: int = kMaxAge*2//3 + self.random_int(0, kMaxAge/3)
    c_endowment: int = kMaxEndowment//4 + self.random_int(0, kMaxEndowment*3/4)
    c_metabolism: int = kMaxMetabolism//3 + self.random_int(0, kMaxMetabolism*2/3)
    c_max_children: int = self.random_int(2, kMaxChildren)
    agent: Agent = None
    if r < kProbMale:
      self.agent_type_ = 1
      agent = DeviceAllocator.new(Male, self, c_vision, 0, c_max_age, c_endowment, c_metabolism)
    elif r < kProbMale + kProbFemale:
      self.agent_type_ = 2
      agent = DeviceAllocator.new(Female, self, c_vision, 0, c_max_age, c_endowment, c_metabolism, c_max_children)

    if agent != None:
      self.enter(agent)

  def leave(self):
    assert self.agent_ref != None
    self.agent_ref = None
    self.agent_type_ = 0

class Agent:
  kIsAbstract: bool = True

  def __init__(self):
    self.cell_ref: Cell = None
    self.cell_request_ref: Cell = None
    self.vision_: int = 0
    self.age_: int = 0
    self.max_age_: int = 0
    self.sugar_: int = 0
    self.metabolism_: int = 0
    self.endowment_: int = 0
    self.permission_: bool = None

  def Agent(self, cell: Cell, vision: int, age: int, max_age: int,
      endowment: int, metabolism: int):
    self.cell_ref = cell
    self.cell_request_ref = None
    self.vision_ = vision
    self.age_ = age
    self.max_age_ = max_age
    self.sugar_ = endowment
    self.endowment_ = endowment
    self.metabolism_ = metabolism
    self.permission_ = False
    assert cell != None
    random.seed(cell.random_int(0, kSize*kSize))
    
  def give_permission(self):
    self.permission_ = True
  
  def age_and_metabolize(self):
    dead: bool = False

    self.age_ = self.age_ + 1
    dead = self.age_ > self.max_age_

    self.sugar_ -= self.metabolism_
    dead = dead or self.sugar_ <= 0

    if dead:
      self.cell_ref.leave()
      DeviceAllocator.destroy(self)

  def prepare_move(self):
    assert self.cell_ref != None

    turn: int = 0
    target_cell: Cell = None
    target_sugar: int = 0

    this_x: int = self.cell_ref.cell_id() % kSize
    this_y: int = self.cell_ref.cell_id() // kSize

    for dx in range(-self.vision_, self.vision_+1, 1):
      for dy in range(-self.vision_, self.vision_+1, 1):
        nx: int = this_x + dx
        ny: int = this_y + dy
        if (dx != 0 or dy != 0) and nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
          n_id: int = nx + ny*kSize
          n_cell: Cell = cells[n_id]
          assert n_cell != None

          if n_cell.is_free():
            if n_cell.sugar() > target_sugar:
              target_cell = n_cell
              target_sugar = n_cell.sugar()
              turn = 1
            elif n_cell.sugar() == target_sugar:
              turn += 1
              if self.random_float() <= 1.0/turn:
                target_cell = n_cell
    
    self.cell_request_ref = target_cell
  
  def update_move(self):
    if self.permission_:
      assert self.cell_request_ref != None
      assert self.cell_request_ref.is_free()
      self.cell_ref.leave()
      self.cell_request_ref.enter(self)
      self.cell_ref = self.cell_request_ref

    self.harvest_sugar()

    self.cell_request_ref = None
    self.permission_ = False

  def harvest_sugar(self):
    amount: int = self.cell_ref.sugar()
    self.cell_ref.take_sugar(amount)
    self.sugar_ += amount

  def ready_to_mate(self) -> bool:
    return (self.sugar_ >= (self.endowment_ * 2 / 3)) and self.age_ >= kMinMatingAge

  def cell_request(self) -> Cell:
    return self.cell_request_ref

  def sugar(self) -> int:
    return self.sugar_

  def vision(self) -> int:
    return self.vision_

  def max_age(self) -> int:
    return self.max_age_

  def endowment(self) -> int:
    return self.endowment_

  def metabolism(self) -> int:
    return self.metabolism_

  def take_sugar(self, amount: int):
    self.sugar_ -= amount

  def random_float(self) -> float:
    return random.uniform(0,1)

class Male(Agent):
  kIsAbstract: bool = False

  def __init__(self):
    self.female_request_ref: Female = None
    self.proposal_accepted_: bool = None

  def Male(self, cell: Cell, vision: int, age: int, max_age: int,
      endowment: int, metabolism: int):
    super().Agent(cell, vision, age, max_age, endowment, metabolism)
    self.proposal_accepted_ = False
    self.female_request_ref = None

  def propose(self):
    if self.ready_to_mate():
      target_agent: Female = None
      target_sugar: int = -1

      this_x: int = self.cell_ref.cell_id() % kSize
      this_y: int = self.cell_ref.cell_id() // kSize

      for dx in range(-self.vision_, self.vision_+1, 1):
        for dy in range(-self.vision_, self.vision_+1, 1):
          nx: int = this_x + dx
          ny: int = this_y + dy
          if nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
            n_id: int = nx + ny*kSize
            n_cell: Cell = cells[n_id]

            if type(n_cell.agent()) == Female:
              n_female: Female = cast(Female, n_cell.agent())
              if n_female.ready_to_mate() and n_female.sugar() > target_sugar:
                target_agent = n_female
                target_sugar = n_female.sugar()

      assert (target_sugar == -1) == (target_agent == None)
      self.female_request_ref = target_agent

  def accept_proposal(self):
    self.proposal_accepted_ = True
  
  def female_request(self) -> Female:
    return self.female_request_ref

  def propose_offspring_target(self):
    if self.proposal_accepted_:
      assert self.female_request_ref != None

      target_cell: Cell = None
      turn: int = 0

      this_x: int = self.cell_ref.cell_id() % kSize
      this_y: int = self.cell_ref.cell_id() // kSize

      for dx in range(-self.vision_, self.vision_+1, 1):
        for dy in range(-self.vision_, self.vision_+1, 1):
          nx: int = this_x + dx
          ny: int = this_y + dy
          if (dx != 0 or dy != 0) and nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
            n_id: int = nx + ny*kSize
            n_cell: Cell = cells[n_id]

            if n_cell.is_free():
              turn += 1

              if self.random_float() <= 1.0/turn:
                target_cell = n_cell
      
      assert (turn == 0) == (target_cell == None)
      self.cell_request_ref = target_cell

  def mate(self):
    if self.proposal_accepted_ and self.permission_:
      assert self.female_request_ref != None
      assert self.cell_request_ref != None

      self.female_request_ref.increment_num_children()

      c_endowment: int = (self.endowment_ + self.female_request_ref.endowment()) // 2
      self.sugar_ -= self.endowment_ // 2
      self.female_request_ref.take_sugar(self.female_request_ref.endowment() // 2)

      c_vision: int = (self.vision_ + self.female_request_ref.vision()) // 2
      c_max_age: int = (self.max_age_ + self.female_request_ref.max_age()) // 2
      c_metabolism: int = (self.metabolism_ + self.female_request_ref.metabolism()) // 2

      child: Agent = None
      if self.random_float() <= 0.5:
        self.cell_request_ref.agent_type_ = 1
        child = DeviceAllocator.new(Male, self.cell_request_ref, c_vision, 0, c_max_age, c_endowment,
          c_metabolism)
      else:
        self.cell_request_ref.agent_type_ = 2
        child = DeviceAllocator.new(Female, self.cell_request_ref, c_vision, 0, c_max_age, c_endowment,
          c_metabolism, self.female_request_ref.max_children())
      
      assert self.cell_request_ref != None
      assert child != None
      assert self.cell_request_ref.is_free()
      self.cell_request_ref.enter(child)

    self.permission_ = False
    self.proposal_accepted_ = False
    self.female_request_ref = None
    self.cell_request_ref = None

class Female(Agent):
  kIsAbstract: bool = False

  def __init__(self):
    self.max_children_: int = 0
    self.num_children_: int = 0

  def Female(self, cell: Cell, vision: int, age: int, max_age: int,
      endowment: int, metabolism: int, max_children: int):
    super().Agent(cell, vision, age, max_age, endowment, metabolism)
    self.num_children_ = 0
    self.max_children_ = max_children

  def decide_proposal(self):
    if self.num_children_ < self.max_children_:
      selected_agent: Male = None
      selected_sugar: int = -1
      this_x: int = self.cell_ref.cell_id() % kSize
      this_y: int = self.cell_ref.cell_id() // kSize

      for dx in range(-kMaxVision, kMaxVision+1, 1):
        for dy in range(-kMaxVision, kMaxVision+1, 1):
          nx: int = this_x + dx
          ny: int = this_y + dy
          if nx >= 0 and nx < kSize and ny >= 0 and ny < kSize:
            n_id: int = nx + ny*kSize 
            n_cell: Cell = cells[n_id]

            if type(n_cell.agent()) == Male:
              n_male: Male = cast(Male, n_cell.agent())
              if n_male.female_request() == self and n_male.sugar() > selected_sugar:
                selected_agent = n_male
                selected_sugar = n_male.sugar()
    
      assert (selected_sugar == -1) == (selected_agent == None)

      if selected_agent != None:
        selected_agent.accept_proposal()

  def increment_num_children(self):
    self.num_children_ += 1

  def max_children(self) -> int:
    return self.max_children_

def main(allocator, do_render):

  def initialize_render():
    pygame.init()
    window = pygame.display.set_mode((kSize*6, kSize*6))
    screen = pygame.Surface((kSize, kSize))
    screen.fill((0,0,0))
    return (window, screen)

  def render(b):
    x = b.cell_id_ % kSize
    y = b.cell_id_ // kSize
    if b.agent_type_ == 1:
      pxarray[x,y] = pygame.Color(0,0,255)
    elif b.agent_type_ == 2:
      pxarray[x,y] = pygame.Color(255,0,0)
    else:
      sugar_level = b.sugar_ / kSugarCapacity
      sugar_level_int = min(int(sugar_level*255), 255)
      pxarray[x,y] = pygame.Color(sugar_level_int, sugar_level_int, 0)

  allocator.initialize()
  allocator.parallel_new(Cell, kSize*kSize)
  allocator.parallel_do(Cell, Cell.setup)

  if do_render:
    os.environ["SDL_VIDEODRIVER"] = "x11"
    screen_width, screen_height = kSize, kSize
    scaling_factor = 6
    window, screen = initialize_render()
    pxarray = pygame.PixelArray(screen)
    allocator.do_all(Cell, render)
    window.blit(pygame.transform.scale(screen, window.get_rect().size), (0,0))
    pygame.display.update()

  total_time = time.perf_counter()

  for i in range(kNumIterations):
    time_before = time.perf_counter()

    allocator.parallel_do(Cell, Cell.grow_sugar)
    allocator.parallel_do(Cell, Cell.prepare_diffuse)
    allocator.parallel_do(Cell, Cell.update_diffuse)

    allocator.parallel_do(Agent, Agent.age_and_metabolize)
    allocator.parallel_do(Agent, Agent.prepare_move)
    allocator.parallel_do(Cell, Cell.decide_permission)
    allocator.parallel_do(Agent, Agent.update_move)

    allocator.parallel_do(Male, Male.propose)
    allocator.parallel_do(Female, Female.decide_proposal)
    allocator.parallel_do(Male, Male.propose_offspring_target)
    allocator.parallel_do(Cell, Cell.decide_permission)
    allocator.parallel_do(Male, Male.mate)

    time_after = time.perf_counter()
    total_time += time_after - time_before

    if do_render:
      allocator.do_all(Cell, render)
      window.blit(pygame.transform.scale(screen, window.get_rect().size), (0,0))
      pygame.display.update() 

  print(total_time)
