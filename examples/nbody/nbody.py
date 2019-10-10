from typing import List
import math
from configuration import *

# cuda_block_size = 256


# floatListをfloatのListとして使える
# floatList = List[float]

class Body:
    dev_Body_pos_x: List[float]
    # dev_Body_pos_x: floatList
    dev_Body_pos_y: List[float]
    dev_Body_vel_x: List[float]
    dev_Body_vel_y: List[float]
    dev_Body_mass: List[float]
    dev_Body_force_x: List[float]
    dev_Body_force_y: List[float]
    device_checksum: List[float]

    def __init__(self, idx: int, pos_x: float, pos_y: float, vel_x: float, vel_y: float, mass: float):
        self.dev_Body_pos_x[idx] = pos_x
        self.dev_Body_pos_y[idx] = pos_y
        self.dev_Body_vel_x[idx] = vel_x
        self.dev_Body_vel_y[idx] = vel_y
        self.dev_Body_mass[idx] = mass

    # device関数
    def body_compute_force(self, idx: int):
        self.dev_Body_force_x[idx] = 0.0
        self.dev_Body_force_y[idx] = 0.0
        i = 0
        while i < kNumBodies:
            i += 1
            if idx != i:
                dx: float = self.dev_Body_pos_x[i] - self.dev_Body_pos_x[idx]
                dy: float = self.dev_Body_pos_y[i] - self.dev_Body_pos_y[idx]
                dist: float = math.sqrt(dx * dx + dy * dy)
                f: float = (kGravityConstant * self.dev_Body_mass[idx] * self.dev_Body_mass[i] /
                            (dist * dist + kDampeningFactor)
                            )
                self.dev_Body_force_x[idx] += f * dx / dist
                self.dev_Body_force_y[idx] += f * dy / dist

    # device関数
    def body_update(self, idx: int):
        self.dev_Body_vel_x[idx] += self.dev_Body_force_x[idx] * kDt / self.dev_Body_mass[idx]
        self.dev_Body_vel_y[idx] += self.dev_Body_force_y[idx] * kDt / self.dev_Body_mass[idx]
        self.dev_Body_pos_x[idx] += self.dev_Body_vel_x[idx] * kDt
        self.dev_Body_pos_y[idx] += self.dev_Body_vel_y[idx] * kDt

        if self.dev_Body_pos_x[idx] < -1 or self.dev_Body_pos_x[idx] > 1:
            self.dev_Body_vel_x[idx] = -self.dev_Body_vel_x[idx]

        if self.dev_Body_pos_y[idx] < -1 or self.dev_Body_pos_y[idx] > 1:
            self.dev_Body_vel_y[idx] = -self.dev_Body_vel_y[idx]
