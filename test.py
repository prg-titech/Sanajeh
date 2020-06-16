# -*- coding: utf-8 -*-

from sanajeh import PyAllocator
from benchmarks.nbody import Body
import time
import hashlib

start_time = time.time()

PyAllocator.compile(py_path='./benchmarks/nbody.py')
compile_time = time.time()
PyAllocator.printCppAndHpp()


# PyAllocator.build()
# build_time = time.time()
#
# PyAllocator.initialize()
# initialize_time = time.time()
#
# obn = 3000
# PyAllocator.parallel_new(Body, obn)
# parallel_new_time = time.time()
#
# for x in range(100):
#     p_do_start_time = time.time()
#     PyAllocator.parallel_do(Body, Body.compute_force)
#     PyAllocator.parallel_do(Body, Body.body_update)
#     p_do_end_time = time.time()
#     print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))
# end_time = time.time()
#
# print("\ncompile time: %.3fs" % (compile_time - start_time))
# print("initialize time: %.3fs" % (initialize_time - compile_time))
# print("parallel new time: %.3fs" % (parallel_new_time - initialize_time))
# print("overall computation time: %.3fs" % (end_time - parallel_new_time))
# print("overall time: %.3fs" % (end_time - start_time))
