import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir + "/src")

from sanajeh import PyAllocator
from nbody import Body
import time
import sys

# load shared library and initialize device classes on GPU
allocator: PyAllocator = PyAllocator("nbody")
allocator.initialize()
initialize_time = time.perf_counter()

# create objects on device
obn = int(sys.argv[1])
itr = int(sys.argv[2])
allocator.parallel_new(Body, obn)
parallel_new_time = time.perf_counter()

# compute on device
for x in range(itr):
  #p_do_start_time = time.perf_counter()
  allocator.parallel_do(Body, Body.compute_force)
  allocator.parallel_do(Body, Body.body_update)
  #p_do_end_time = time.perf_counter()
end_time = time.perf_counter()

print("parallel new time(%-5d objects): %.dµs" % (obn, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / itr))
print("overall computation time(%-4d iterations): %dµs" % (itr, ((end_time - parallel_new_time) * 1000000)))