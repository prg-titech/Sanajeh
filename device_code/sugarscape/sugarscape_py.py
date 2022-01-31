
from __future__ import annotations
import os, sys, time, pygame, random
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append((parentdir + '/src'))
from sanajeh import DeviceAllocator
kSize: int = 100
kSeed: int = 42
kProbMale: float = 0.12
kProbFemale: float = 0.15
kNumIterations: int = 10000
kMaxVision: int = 2
kMaxAge: int = 100
kMaxEndowment: int = 200
kMaxMetabolism: int = 80
kSugarCapacity: int = 3500
kMaxSugarDiffusion: int = 60
kSugarDiffusionRate: float = 0.125
kMinMatingAge: int = 22
kMaxChildren: int = 8
cells: list[Cell] = DeviceAllocator.array((kSize * kSize))

class SanajehBaseClass():

    def __init__(self):
        self.class_type: int = 0

    def SanajehBaseClass(self):
        pass

class Cell(SanajehBaseClass):

    def __init__(self):
        self.random_state_: DeviceAllocator.RandomState = None
        self.agent_ref: Agent = None
        self.sugar_diffusion_: int = 0
        self.sugar_: int = 0
        self.sugar_capacity_: int = 0
        self.grow_rate_: int = 0
        self.cell_id_: int = 0
        self.agent_type: int = 0
        self.atomic_request: int = 0

    def Cell(self, cell_id: int):
        self.agent_ref = None
        self.sugar_ = 0
        self.sugar_capacity_ = 3500
        self.cell_id_ = cell_id
        random.seed(cell_id)
        cells[cell_id] = self
        max_grow_rate: int = 50
        r: float = self.random_float()
        if (r <= 0.02):
            self.grow_rate_ = max_grow_rate
        elif (r <= 0.04):
            self.grow_rate_ = (0.5 * max_grow_rate)
        elif (r <= 0.08):
            self.grow_rate_ = (0.25 * max_grow_rate)
        else:
            self.grow_rate_ = 0

    def Setup(self):
        r: float = self.random_float()
        c_vision: int = ((kMaxVision // 2) + self.random_int(0, (kMaxVision // 2)))
        c_max_age: int = (((kMaxAge * 2) // 3) + self.random_int(0, (kMaxAge // 3)))
        c_endowment: int = ((kMaxEndowment // 4) + self.random_int(0, ((kMaxEndowment * 3) // 4)))
        c_metabolism: int = ((kMaxMetabolism // 3) + self.random_int(0, ((kMaxMetabolism * 2) // 3)))
        c_max_children: int = self.random_int(2, kMaxChildren)
        agent: Agent = None
        if (r < kProbMale):
            agent = DeviceAllocator.new(Male, self, c_vision, 0, c_max_age, c_endowment, c_metabolism)
        elif (r < (kProbMale + kProbFemale)):
            agent = DeviceAllocator.new(Female, self, c_vision, 0, c_max_age, c_endowment, c_metabolism, c_max_children)
        else:
            pass
        if (agent != None):
            self.enter(agent)

    def prepare_diffuse(self):
        self.sugar_diffusion_ = (kSugarDiffusionRate * self.sugar_)
        max_diff: int = kMaxSugarDiffusion
        if (self.sugar_diffusion_ > max_diff):
            self.sugar_diffusion_ = max_diff
        self.sugar_ -= self.sugar_diffusion_

    def update_diffuse(self):
        new_sugar: int = 0
        self_x: int = (self.cell_id_ % kSize)
        self_y: int = (self.cell_id_ // kSize)
        for dx in range((- kMaxVision), (kMaxVision + 1)):
            for dy in range((- kMaxVision), (kMaxVision + 1)):
                nx: int = (self_x + dx)
                ny: int = (self_y + dy)
                if (((dx != 0) or (dy != 0)) and (nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                    n_id: int = (nx + (ny * kSize))
                    n_cell: Cell = cells[n_id]
                    new_sugar += (0.125 * n_cell.sugar_diffusion_)
        self.sugar_ += new_sugar
        if (self.sugar_ > self.sugar_capacity_):
            self.sugar_ = self.sugar_capacity_

    def decide_permission(self):
        selected_agent: Agent = None
        turn: int = 0
        self_x: int = (self.cell_id_ % kSize)
        self_y: int = (self.cell_id_ // kSize)
        for dx in range((- kMaxVision), (kMaxVision + 1)):
            for dy in range((- kMaxVision), (kMaxVision + 1)):
                nx: int = (self_x + dx)
                ny: int = (self_y + dy)
                if ((nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                    n_id: int = (nx + (ny * kSize))
                    n_cell: Cell = cells[n_id]
                    n_agent: Agent = n_cell.agent()
                    if ((n_agent != None) and (n_agent.cell_request() == self)):
                        turn += 1
                        if (self.random_float() <= (1.0 / turn)):
                            selected_agent = n_agent
                        else:
                            assert (turn > 1)
        assert ((turn == 0) == (selected_agent == None))
        if (selected_agent != None):
            selected_agent.give_permission()

    def is_free(self) -> bool:
        return (self.agent_ref == None)

    def enter(self, agent: Agent):
        assert (self.agent_ref == None)
        assert (agent != None)
        self.agent_ref = agent
        if (type(self.agent_ref) == Male):
            self.agent_type = 1
        elif (type(self.agent_ref) == Female):
            self.agent_type = 2
        else:
            self.agent_type = 0

    def leave(self):
        assert (self.agent_ref != None)
        self.agent_ref = None
        self.agent_type = 0

    def sugar(self) -> int:
        return self.sugar_

    def take_sugar(self, amount: int):
        self.sugar_ -= amount

    def grow_sugar(self):
        self.sugar_ += min((self.sugar_capacity_ - self.sugar_), self.grow_rate_)

    def random_float(self) -> float:
        return ((random.getrandbits(32) % 100) * 0.01)

    def random_int(self, a: int, b: int) -> int:
        return (a + (random.getrandbits(32) % ((b - a) + 1)))

    def cell_id(self) -> int:
        return self.cell_id_

    def agent(self) -> Agent:
        return self.agent_ref

    def add_to_draw_array(self):
        pass

    def requestTicket(self):
        if (self.atomic_request < 200):
            DeviceAllocator.atomicAdd(self.atomic_request, 1)

class Agent(SanajehBaseClass):
    kIsAbstract: bool = True

    def __init__(self):
        self.random_state_: DeviceAllocator.RandomState = None
        self.cell_ref: Cell = None
        self.cell_request_ref: Cell = None
        self.vision_: int = 0
        self.age_: int = 0
        self.max_age_: int = 0
        self.sugar_: int = 0
        self.metabolism_: int = 0
        self.endowment_: int = 0
        self.permission_: bool = False

    def Agent(self, cell: Cell, vision: int, age: int, max_age: int, endowment: int, metabolism: int):
        self.cell_ref = cell
        self.cell_request_ref = None
        self.vision_ = vision
        self.age_ = age
        self.max_age_ = max_age
        self.sugar_ = endowment
        self.endowment_ = endowment
        self.metabolism_ = metabolism
        self.permission_ = False
        assert (cell != None)
        __auto_v0: int = cell.random_int(0, kSize)
        random.seed(__auto_v0)

    def prepare_move(self):
        assert (self.cell_ref != None)
        self.cell_ref.requestTicket()
        turn: int = 0
        target_cell: Cell = None
        target_sugar: int = 0
        self_x: int = (self.cell_ref.cell_id() % kSize)
        self_y: int = (self.cell_ref.cell_id() // kSize)
        for dx in range((- self.vision_), (self.vision_ + 1)):
            for dy in range((- self.vision_), (self.vision_ + 1)):
                nx: int = (self_x + dx)
                ny: int = (self_y + dy)
                if (((dx != 0) or (dy != 0)) and (nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                    n_id: int = (nx + (ny * kSize))
                    n_cell: Cell = cells[n_id]
                    assert (n_cell != None)
                    if n_cell.is_free():
                        if (n_cell.sugar() > target_sugar):
                            target_cell = n_cell
                            target_sugar = n_cell.sugar()
                            turn = 1
                        elif (n_cell.sugar() == target_sugar):
                            turn += 1
                            if (self.random_float() <= (1.0 / turn)):
                                target_cell = n_cell
        self.cell_request_ref = target_cell

    def update_move(self):
        if (self.permission_ == True):
            assert (self.cell_request_ref != None)
            assert self.cell_request_ref.is_free()
            self.cell_ref.leave()
            self.cell_request_ref.enter(self)
            self.cell_ref = self.cell_request_ref
        self.harvest_sugar()
        self.cell_request_ref = None
        self.permission_ = False

    def give_permission(self):
        self.permission_ = True

    def age_and_metabolize(self):
        dead: bool = False
        self.age_ += 1
        dead = (self.age_ > self.max_age_)
        self.sugar_ -= self.metabolism_
        dead = (dead or (self.sugar_ <= 0))
        if dead:
            self.cell_ref.leave()
            DeviceAllocator.destroy(self)

    def harvest_sugar(self):
        amount: int = self.cell_ref.sugar()
        self.cell_ref.take_sugar(amount)
        self.sugar_ += amount

    def ready_to_mate(self) -> bool:
        return ((self.sugar_ >= ((self.endowment_ * 2) / 3)) and (self.age_ >= kMinMatingAge))

    def cell_request(self) -> Cell:
        return self.cell_request_ref

    def sugar(self) -> int:
        return self.sugar_

    def endowment(self) -> int:
        return self.endowment_

    def vision(self) -> int:
        return self.vision_

    def max_age(self) -> int:
        return self.max_age_

    def metabolism(self) -> int:
        return self.metabolism_

    def take_sugar(self, amount: int):
        self.sugar_ = (+ amount)

    def random_float(self) -> float:
        return ((random.getrandbits(32) % 100) * 0.01)

class Male(Agent):
    kIsAbstract: bool = False

    def __init__(self):
        self.female_request_ref: Female = None
        self.proposal_accepted_: bool = False

    def Male(self, cell: Cell, vision: int, age: int, max_age: int, endowment: int, metabolism: int):
        super().Agent(cell, vision, age, max_age, endowment, metabolism)
        self.proposal_accepted_ = False
        self.female_request_ref = None

    def female_request(self) -> Female:
        return self.female_request_ref

    def accept_proposal(self):
        self.proposal_accepted_ = True

    def propose(self):
        if self.ready_to_mate():
            target_agent: Female = None
            target_sugar: int = (- 1)
            self_x: int = (self.cell_ref.cell_id() % kSize)
            self_y: int = (self.cell_ref.cell_id() // kSize)
            for dx in range((- self.vision_), (self.vision_ + 1)):
                for dy in range((- self.vision_), (self.vision_ + 1)):
                    nx: int = (self_x + dx)
                    ny: int = (self_y + dy)
                    if ((nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                        n_id: int = (nx + (ny * kSize))
                        n_cell: Cell = cells[n_id]
                        __auto_v0: Agent = n_cell.agent()
                        n_female: Female = DeviceAllocator.cast(Female, __auto_v0)
                        if ((type(n_female) == Female) and n_female.ready_to_mate()):
                            if (n_female.sugar() > target_sugar):
                                target_agent = n_female
                                target_sugar = n_female.sugar()
            assert ((target_sugar == (- 1)) == (target_agent == None))
            self.female_request_ref = target_agent

    def propose_offspring_target(self):
        if self.proposal_accepted_:
            assert (self.female_request_ref != None)
            target_cell: Cell = None
            turn: int = 0
            self_x: int = (self.cell_ref.cell_id() % kSize)
            self_y: int = (self.cell_ref.cell_id() // kSize)
            for dx in range((- self.vision_), (self.vision_ + 1)):
                for dy in range((- self.vision_), (self.vision_ + 1)):
                    nx: int = (self_x + dx)
                    ny: int = (self_y + dy)
                    if (((dx != 0) or (dy != 0)) and (nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                        n_id: int = (nx + (ny * kSize))
                        n_cell: Cell = cells[n_id]
                        if n_cell.is_free():
                            turn += 1
                            if (self.random_float() <= (1 / turn)):
                                target_cell = n_cell
            assert ((turn == 0) == (target_cell == None))
            self.cell_request_ref = target_cell

    def mate(self):
        if (self.proposal_accepted_ and self.permission_):
            assert (self.female_request_ref != None)
            assert (self.cell_request_ref != None)
            self.female_request_ref.increment_num_children()
            c_endowment: int = ((self.endowment_ + self.female_request_ref.endowment()) // 2)
            self.sugar_ -= (self.endowment_ // 2)
            self.female_request_ref.take_sugar((self.female_request_ref.endowment() // 2))
            c_vision: int = ((self.vision_ + self.female_request_ref.vision()) // 2)
            c_max_age: int = ((self.max_age_ + self.female_request_ref.max_age()) // 2)
            c_metabolism: int = ((self.metabolism_ + self.female_request_ref.metabolism()) // 2)
            child: Agent = None
            if (self.random_float() <= 0.5):
                child = DeviceAllocator.new(Male, self.cell_request_ref, c_vision, 0, c_max_age, c_endowment, c_metabolism)
            else:
                __auto_v0: int = self.female_request_ref.max_children()
                child = DeviceAllocator.new(Female, self.cell_request_ref, c_vision, 0, c_max_age, c_endowment, c_metabolism, __auto_v0)
            assert (self.cell_request_ref != None)
            assert (child != None)
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

    def Female(self, cell: Cell, vision: int, age: int, max_age: int, endowment: int, metabolism: int, max_children: int):
        super().Agent(cell, vision, age, max_age, endowment, metabolism)
        self.num_children_ = 0
        self.max_children_ = max_children

    def decide_proposal(self):
        if (self.num_children_ < self.max_children_):
            selected_agent: Male = None
            selected_sugar: int = (- 1)
            self_x: int = (self.cell_ref.cell_id() % kSize)
            self_y: int = (self.cell_ref.cell_id() // kSize)
            for dx in range((- kMaxVision), (kMaxVision + 1)):
                for dy in range((- kMaxVision), (kMaxVision + 1)):
                    nx: int = (self_x + dx)
                    ny: int = (self_y + dy)
                    if ((nx >= 0) and (nx < kSize) and (ny >= 0) and (ny < kSize)):
                        n_id: int = (nx + (ny * kSize))
                        n_cell: Cell = cells[n_id]
                        __auto_v0: Agent = n_cell.agent()
                        n_male: Male = DeviceAllocator.cast(Male, __auto_v0)
                        if (type(n_male) == Male):
                            if ((n_male.female_request() == self) and (n_male.sugar() > selected_sugar)):
                                selected_agent = n_male
                                selected_sugar = n_male.sugar()
            assert ((selected_sugar == (- 1)) == (selected_agent == None))
            if (selected_agent != None):
                selected_agent.accept_proposal()

    def increment_num_children(self):
        self.num_children_ += 1

    def max_children(self) -> int:
        return self.max_children_

def main(allocator, do_render):

    def initialize_render():
        pygame.init()
        window = pygame.display.set_mode(((kSize * 6), (kSize * 6)))
        screen = pygame.Surface((kSize, kSize))
        screen.fill((0, 0, 0))
        return (window, screen)

    def render(b):
        x = (b.cell_id_ % kSize)
        y = (b.cell_id_ // kSize)
        if (b.agent_type == 1):
            pxarray[(x, y)] = pygame.Color(0, 255, 0)
        elif (b.agent_type == 2):
            pxarray[(x, y)] = pygame.Color(255, 0, 0)
        else:
            sugarpct = (b.sugar_ / kSugarCapacity)
            clr = int((sugarpct * 254))
            pxarray[(x, y)] = pygame.Color(clr, clr, 0)
    allocator.initialize()
    allocator.parallel_new(Cell, (kSize * kSize))
    allocator.parallel_do(Cell, Cell.Setup)
    if do_render:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        (screen_width, screen_height) = (kSize, kSize)
        scaling_factor = 6
        (window, screen) = initialize_render()
        pxarray = pygame.PixelArray(screen)
        window.blit(pygame.transform.scale(screen, window.get_rect().size), (0, 0))
        pygame.display.update()
    total_time = 0
    for i in range(kNumIterations):
        starttime = time.perf_counter()
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
        endtime = time.perf_counter()
        total_time += (endtime - starttime)
        print('iterate', i)
        if do_render:
            screen.fill((0, 0, 0))
            allocator.do_all(Cell, render)
            window.blit(pygame.transform.scale(screen, window.get_rect().size), (0, 0))
            pygame.display.update()
    print(total_time)
