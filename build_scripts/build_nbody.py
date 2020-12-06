# -*- coding: utf-8 -*-

from sanajeh import PyAllocator
import time

start_time = time.perf_counter()

# Compile python code to cpp code
PyAllocator.compile(py_path='./benchmarks/nbody_vector.py')
compile_time = time.perf_counter()
# PyAllocator.printCppAndHpp()
# PyAllocator.printCdef()

# Compile cpp code to shared library
PyAllocator.build()
build_time = time.perf_counter()