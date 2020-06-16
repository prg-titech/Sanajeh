# -*- coding: utf-8 -*-

from sanajeh import __pyallocator__
from benchmarks.nbody import Body
import time

start_time = time.time()

__pyallocator__.compile(py_path='./benchmarks/nbody.py')
compile_time = time.time()
# __pyallocator__.printCppAndHpp()

__pyallocator__.initialize()
initialize_time = time.time()

__pyallocator__.parallel_new(Body, 3000)
parallel_new_time = time.time()

for x in range(100):
    p_do_start_time = time.time()
    __pyallocator__.parallel_do(Body, Body.compute_force)
    __pyallocator__.parallel_do(Body, Body.body_update)
    p_do_end_time = time.time()
    print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))
end_time = time.time()

print("compile time: %.3fs" % (compile_time - start_time))
print("initialize time: %.3fs" % (initialize_time - start_time))
print("parallel new time: %.3fs" % (parallel_new_time - initialize_time))
print("overall time: %.3fs" % (end_time - start_time))
