# -*- coding: utf-8 -*-
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir + '/src')

from sanajeh import PyAllocator
from nbody import Body
import time
import sys

# Load shared library and initialize device classes on GPU
PyAllocator.initialize()
initialize_time = time.perf_counter()

# Create objects on device
obn = int(sys.argv[1])
itr = int(sys.argv[2])
PyAllocator.parallel_new(Body, obn)
parallel_new_time = time.perf_counter()

# Compute on device
for x in range(itr):
    p_do_start_time = time.perf_counter()
    PyAllocator.parallel_do(Body, Body.compute_force)
    PyAllocator.parallel_do(Body, Body.body_update)
    p_do_end_time = time.perf_counter()
    # print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))
end_time = time.perf_counter()

object_index = 0


def printAllFields(b):
    global object_index
    print("Object {}:".format(object_index))
    print("\tpos_x:{}".format(b.pos_x))
    print("\tpos_y:{}".format(b.pos_y))
    print("\tvel_x:{}".format(b.vel_x))
    print("\tvel_y:{}".format(b.vel_y))
    print("\tforce_x:{}".format(b.force_x))
    print("\tforce_y:{}".format(b.force_y))
    print("\tmass:{}\n".format(b.mass))
    object_index = object_index + 1


end_time2 = time.perf_counter()

# print("compile time(py2cpp): %dµs" % ((compile_time - start_time) * 1000000))
# print("compile time(nvcc): %dµs" % ((build_time - compile_time) * 1000000))
# print("initialize time: %dµs" % ((initialize_time - build_time) * 1000000))
print("parallel new time(%-5d objects): %.dµs" % (obn, ((parallel_new_time - initialize_time) * 1000000)))
print("average computation time: %dµs" % ((end_time - parallel_new_time) * 1000000 / itr))
print("overall computation time(%-4d iterations): %dµs" % (itr, ((end_time - parallel_new_time) * 1000000)))
# print("do_all time: %dµs" % ((end_time2 - end_time) * 1000000))
# print("overall time: %dµs" % ((end_time2 - start_time) * 1000000))