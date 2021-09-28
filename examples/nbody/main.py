import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyAllocator
from nbody import Body
import time

"""
Options parser
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("number", help="number of bodies", type=int)
parser.add_argument("iter", help="number of iteration", type=int)
parser.add_argument("--cpu", help="process sequentially", action="store_true")
parser.add_argument("--r", help="rendering option", action="store_true")
args = parser.parse_args()

"""
Rendering setting
"""
import pygame
os.environ["SDL_VIDEODRIVER"] = "windib"

def render(b):
    px = int((b.pos.x + 1) * 150)
    py = int((b.pos.y + 1) * 150)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), b.mass/10000*20)

def clear_screen():
    screen.fill((0, 0, 0))

if (args.r):
  screen_width = 300
  screen_height = 300
  pygame.init()
  screen = pygame.display.set_mode((screen_width, screen_height))
  pygame.display.flip()  


"""
Main program
"""
allocator: PyAllocator = PyAllocator("examples/nbody/nbody.py", "nbody", args.cpu)

allocator.initialize()
initialize_time = time.perf_counter()

allocator.parallel_new(Body, args.number)
parallel_new_time = time.perf_counter()

for x in range(args.iter):
  allocator.parallel_do(Body, Body.compute_force)
  allocator.parallel_do(Body, Body.body_update)
  if (args.r):
    allocator.do_all(Body, render)
    pygame.display.flip()
    clear_screen()    
end_time = time.perf_counter()

"""
print("parallel new time(%-5d objects): %.dµs" % (args.number, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / args.iter))
print("overall computation time(%-4d iterations): %dµs" % (args.iter, ((end_time - parallel_new_time) * 1000000)))  
"""