import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir + "/src")

from sanajeh_seq import PyAllocator
from nbody_seq import Body
import time

import pygame
os.environ["SDL_VIDEODRIVER"] = "windib"

screen_width = 500
screen_height = 500
pygame.display.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.flip()

def render(b):
  px = int((b.pos.x + 1) * 250)
  py = int((b.pos.y + 1) * 250)
  pygame.draw.circle(screen, (255, 255, 255), (px, py), 2)

def clear_screen():
  screen.fill((0, 0, 0))

allocator: PyAllocator = PyAllocator()
allocator.initialize()
initialize_time = time.perf_counter()

obn = int(sys.argv[1])
itr = int(sys.argv[2])
allocator.parallel_new(Body, obn)
parallel_new_time = time.perf_counter()

for x in range(itr):
  allocator.parallel_do(Body, Body.compute_force)
  allocator.parallel_do(Body, Body.body_update)
  allocator.do_all(Body, render)
  pygame.display.flip()
  clear_screen()  
  time.sleep(0.01)
end_time = time.perf_counter()

"""
print("parallel new time(%-5d objects): %.dµs" % (obn, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / itr))
print("overall computation time(%-4d iterations): %dµs" % (itr, ((end_time - parallel_new_time) * 1000000)))  
"""