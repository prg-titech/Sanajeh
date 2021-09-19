import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyAllocator
from nbody import Body
from nbody import Vector
import time

from expander import RuntimeExpander

"""
import pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

def render(b):
    px = int((b.pos.x + 1) * 150)
    py = int((b.pos.y + 1) * 150)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), b.mass/10000*20)

def clear_screen():
    screen.fill((0, 0, 0))

screen_width = 300
screen_height = 300
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.flip()  
"""

"""
compiler: PyCompiler = PyCompiler("examples/nbody/nbody.py", "nbody")
compiler.compile()
"""

expander: RuntimeExpander = RuntimeExpander()
expander.build_function(Body)
