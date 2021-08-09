from __future__ import annotations

import math

from typing import List

from sanajeh import DeviceAllocator

SIZE_X: int = 100
SIZE_Y: int = 100

cells: List[Cell]
DeviceAllocator.array_size(cells, 1000)


class Cell:
  agent_: Agent

  def __init__(self, idx: int):
    self.agent_ = None
    cells[idx] = self
    # todo cell[], alive cell

  def agent(self) -> Agent:
    return self.agent_

  def is_empty(self) -> bool:
    return self.agent_ is None


class Agent:
  cell_id_: int
  action_: int  # 0 none 1 die 2 alive
  is_alive_: bool
  is_new_: bool

  def __init__(self, cid: int):
    self.cell_id_ = cid
    self.action_ = 0
    self.is_alive_ = False
    self.is_new_ = False

  def cell_id(self) -> int:
    return self.cell_id_

  def num_alive_neighbors(self) -> int:
    cell_x: int = self.cell_id_ % SIZE_X
    cell_y: int = self.cell_id_ / SIZE_Y
    result: int = 0
    dx: int = -1
    dy: int = -1
    while dx < 2:
      while dy < 2:
        nx: int = cell_x + dx
        ny: int = cell_y + dy

        if -1 < nx < SIZE_X and -1 < ny < SIZE_Y:
          if cells[ny * SIZE_X + nx].agent().is_alive_:
            result += 1
      dy += 1
    dx += 1
    return result


class Alive(Agent):
  def __init__(self, cid: int):
    super().__init__(cid)
    self.is_new_ = True
    self.is_alive_ = True

  def prepare(self):
    self.is_new_ = False
    alive_neighbors: int = self.num_alive_neighbors() - 1
    if alive_neighbors < 2 or alive_neighbors > 3:
      self.action_ = 1

  def update(self):
    cid: int = self.cell_id_

    if self.is_new_:
      self.create_candidates()
    elif self.action_ == 1:
      cells[cid].agent_ = DeviceAllocator.new(Candidate, cid)
      DeviceAllocator.destroy(self)

  def create_candidates(self):
    cell_x: int = self.cell_id_ % SIZE_X
    cell_y: int = self.cell_id_ / SIZE_Y
    dx: int = -1
    dy: int = -1
    while dx < 2:
      while dy < 2:
        nx: int = cell_x + dx
        ny: int = cell_y + dy
        if -1 < nx < SIZE_X and -1 < ny < SIZE_Y:
          if cells[ny * SIZE_X + nx].is_empty():
            self.maybe_create_candidate(nx, ny)
        dy += 1
      dx += 1

  def maybe_create_candidate(self, x: int, y: int):
    dx: int = -1
    dy: int = -1
    while dx < 2:
      while dy < 2:
        nx: int = x + dx
        ny: int = y + dy

        if -1 < nx < SIZE_X and -1 < ny < SIZE_Y:
          if cells[ny * SIZE_X + nx].agent().is_alive_:
            alive: Agent = cells[ny * SIZE_X + nx].agent()
            if alive.is_new_:
              if alive is self:
                cells[y * SIZE_X + x].agent_ = DeviceAllocator.new(Candidate, y * SIZE_X + x)
              return
      dy += 1
    dx += 1


class Candidate(Agent):

  def __init__(self, cid: int):
    super(self).__init__(cid)
    self.is_alive_ = False

  def prepare(self):
    alive_neighbors: int = self.num_alive_neighbors()
    if alive_neighbors == 3:
      self.action_ = 2
    elif alive_neighbors == 0:
      self.action_ = 1

  def update(self):
    cid: int = self.cell_id_
    if self.action_ == 2:
      cells[cid].agent_ = DeviceAllocator.new(Alive, cid)
      cells[cid].agent_.is_alive_ = True
      DeviceAllocator.destroy(self)


def kernel_initialize_bodies():
    DeviceAllocator.device_class(Cell, Agent, Candidate, Alive)