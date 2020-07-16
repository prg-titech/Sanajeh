# -*- coding: utf-8 -*-
import sanajeh
from sanajeh import PyAllocator, ffi, py_lib
import time
import sys

start_time = time.time()

# Compile python code to cpp code
PyAllocator.compile(source_path='./benchmarks/nbody.py')
compile_time = time.time()
# PyAllocator.printCppAndHpp()
# PyAllocator.printCdef()

# Compile cpp code to shared library
PyAllocator.build()
build_time = time.time()

# Load shared library and initialize device classes on GPU
PyAllocator.initialize()
initialize_time = time.time()

# Create objects on device
obn = int(sys.argv[1])
itr = int(sys.argv[2])
PyAllocator.parallel_new(py_lib.Body, obn)
parallel_new_time = time.time()

# Compute on device
for x in range(itr):
    p_do_start_time = time.time()
    PyAllocator.parallel_do(py_lib.Body, py_lib.Body.compute_force)
    PyAllocator.parallel_do(py_lib.Body, py_lib.Body.body_update)
    p_do_end_time = time.time()
    # print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))
end_time = time.time()

object_index = 0


@ffi.callback("void(float, float, float, float, float, float, float)")
def printAllFields(pos_x, pos_y, vel_x, vel_y, force_x, force_y, mass):
    global object_index
    print("Object {}:".format(object_index))
    print("\tpos_x:{}".format(pos_x))
    print("\tpos_y:{}".format(pos_y))
    print("\tvel_x:{}".format(vel_x))
    print("\tvel_y:{}".format(vel_y))
    print("\tforce_x:{}".format(force_x))
    print("\tforce_y:{}".format(force_y))
    print("\tmass:{}\n".format(mass))
    object_index = object_index + 1


PyAllocator.do_all(PyAllocator.py_lib.Body, printAllFields)

print("compile time(py2cpp): %.3fs" % (compile_time - start_time))
print("compile time(nvcc): %.3fs" % (build_time - compile_time))
print("initialize time: %.3fs" % (initialize_time - build_time))
print("parallel new time(%-5d objects): %.3fs" % (obn, parallel_new_time - initialize_time))
print("average computation time: %.3fs" % ((end_time - parallel_new_time) / itr))
print("overall computation time(%-4d iterations): %.3fs" % (itr, end_time - parallel_new_time))
print("overall time: %.3fs" % (end_time - start_time))
