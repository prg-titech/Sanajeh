# -*- coding: utf-8 -*-

from sanajeh import __pyallocator__
from benchmarks.nbody import Body
import time

start_time = time.time()
__pyallocator__.initialize(path='./benchmarks/nbody.py')
# __pyallocator__.printCppAndHpp()
initialize_time = time.time()
print("initialize time: %.3fs" % (initialize_time - start_time))
__pyallocator__.parallel_new(Body, 3000)
p_new_time = time.time()
print("new time: %.3fs" % (p_new_time - initialize_time))
for x in range(100):
    p_do_start_time = time.time()
    __pyallocator__.parallel_do(Body, Body.compute_force)
    __pyallocator__.parallel_do(Body, Body.body_update)
    p_do_end_time = time.time()
    print("iteration%-3d time: %.3fs" % (x, p_do_end_time - p_do_start_time))
end_time = time.time()
print("overall time: %3.fs" % (end_time - start_time))
