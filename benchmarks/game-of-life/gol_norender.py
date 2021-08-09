import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir + '/src')

from sanajeh import PyAllocator
from nbody_vector import Body
import time
import sys

PyAllocator.initialize()
initialize_time = time.perf_counter()

obn = int(sys.argv[1])
itr = int(sys.argv[2])
PyAllocator.parallel_new(Cell, obn)
parallel_new_time = time.perf_counter()