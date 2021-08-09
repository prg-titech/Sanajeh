# -*- coding: utf-8 -*-
import pygame

from benchmarks.nbody_vector import Body
import time
import sys
import random


obn = int(sys.argv[1])
itr = int(sys.argv[2])

bodies = []
# Compute on device
for i in range(obn):
    px_ = 2.0 * random.random() - 1.0
    py_ = 2.0 * random.random() - 1.0
    vx_ = 0.0
    vy_ = 0.0
    ms_ = (random.random() / 2.0 + 0.5) * 1000.0
    bodies.append(Body(px_, py_, vx_, vy_, 0.0, 0.0, ms_))

screen_width = 300
screen_height = 300
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.flip()


def render(b):
    px = int((b.pos.x + 1) * 150)
    py = int((b.pos.y + 1) * 150)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), 2)


def clear_screen():
    screen.fill((0, 0, 0))


# Compute on device
for x in range(itr):
    # p_do_start_time = time.perf_counter()
    for body in bodies:
        render(body)
    for j in range(obn):
        for k in range(obn):
            bodies[j].apply_force(bodies[k])
    for j in range(obn):
        bodies[j].body_update()
    pygame.display.flip()
    clear_screen()
    # p_do_end_time = time.perf_counter()
    # print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))

end_time = time.perf_counter()
