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
args = parser.parse_args()

allocator: PyAllocator = PyAllocator("examples/nbody/nbody.py", "nbody", args.cpu)

allocator.initialize()
initialize_time = time.perf_counter()

allocator.parallel_new(Body, args.number)
parallel_new_time = time.perf_counter()

for x in range(args.iter):
  allocator.parallel_do(Body, Body.compute_force)
  allocator.parallel_do(Body, Body.body_update)
end_time = time.perf_counter()

print("parallel new time(%-5d objects): %.dµs" % (args.number, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / args.iter))
print("overall computation time(%-4d iterations): %dµs" % (args.iter, ((end_time - parallel_new_time) * 1000000)))  