# -*- coding: utf-8 -*-
# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr

from typing import Callable
import cffi
import build

# Sanajeh package
import py2cpp
from config import CPP_FILE_PATH, HPP_FILE_PATH, SO_FILE_PATH


# Device side allocator
class PyAllocator:
    cpp_code: str
    hpp_code: str
    cdef_code: str
    lib = None

    def initialize(self, path):
        source = open(path, encoding="utf-8").read()
        codes = py2cpp.compile(source, CPP_FILE_PATH, HPP_FILE_PATH)
        self.cpp_code = codes[0]
        self.hpp_code = codes[1]
        self.cdef_code = codes[2]
        # Compile cpp source file to .so file
        build.run()
        # Initialize ffi module
        ffi = cffi.FFI()
        ffi.cdef(self.cdef_code)
        self.lib = ffi.dlopen(SO_FILE_PATH)
        if self.lib.AllocatorInitialize()==0:
            print("Successfully initialized the allocator through FFI.")

    # DEBUG propose
    def printCppAndHpp(self):
        print(self.cpp_code)
        print("--------------------------------")
        print(self.hpp_code)

    # dummy device_do
    def device_do(self, class_name, func: Callable, *args):
        pass

    def parallel_do(self, class_name, func: Callable, *args):
        # ffiでparallel_doを呼び出す
        pass

    def parallel_new(self, class_name, object_num):
        # ffiでparallel_newを呼び出す
        pass

    # dummy rand_init
    def rand_init(self, seed, sequence, offset):
        pass

    # dummy rand_uniform
    # (0,1]
    def rand_uniform(self):
        # curand_uniform
        pass


# identifier used in users' python code
__pyallocator__ = PyAllocator()








