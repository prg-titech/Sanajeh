import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir + "/src")

from sanajeh_seq import PyAllocator
from nbody_seq import Body
import time

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
end_time = time.perf_counter()

print("parallel new time(%-5d objects): %.dµs" % (obn, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / itr))
print("overall computation time(%-4d iterations): %dµs" % (itr, ((end_time - parallel_new_time) * 1000000)))  