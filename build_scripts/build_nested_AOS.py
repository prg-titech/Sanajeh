# -*- coding: utf-8 -*-

from sanajeh import PyAllocator
import time

start_time = time.perf_counter()

# Compile python code to cpp code
# Since the nested AOS is written artificially the next line is not needed
# PyAllocator.compile(py_path='../benchmarks/nbody_vector.py')


compile_time = time.perf_counter()

# Compile cpp code to shared library
PyAllocator.build(cpp_path="../device_code/vector_AOS/vector_AOS.cu", so_path="../device_code/vector_AOS/vector_AOS.so")
build_time = time.perf_counter()