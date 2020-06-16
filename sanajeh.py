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

    # compile python code to cpp code and .so file
    def compile(self, py_path):
        source = open(py_path, encoding="utf-8").read()
        codes = py2cpp.compile(source, CPP_FILE_PATH, HPP_FILE_PATH)
        self.cpp_code = codes[0]
        self.hpp_code = codes[1]
        self.cdef_code = codes[2]
        # Compile cpp source file to .so file
        build.run()

    # load the shared library and initialize the allocator on GPU
    def initialize(self, so_path=SO_FILE_PATH):
        # Initialize ffi module
        ffi = cffi.FFI()
        ffi.cdef(self.cdef_code)
        self.lib = ffi.dlopen(so_path)
        if self.lib.AllocatorInitialize() == 0:
            print("Successfully initialized the allocator through FFI.")

    # DEBUG propose
    def printCppAndHpp(self):
        print(self.cpp_code)
        print("--------------------------------")
        print(self.hpp_code)

    # dummy device_do
    def device_do(self, cls, func, *args):
        pass

    def parallel_do(self, cls, func, *args):
        object_class_name = cls.__name__
        func_str = func.__qualname__.split(".")
        # todo nested class exception
        func_class_name = func_str[0]
        func_name = func_str[1]
        # todo args
        if eval("self.lib.{}_{}_{}".format(object_class_name, func_class_name, func_name))() == 0:
            print("Successfully called parallel_do {} {} {}".format(object_class_name, func_class_name, func_name))

    def parallel_new(self, cls, object_num):
        object_class_name = cls.__name__
        if eval("self.lib.parallel_new_{}".format(object_class_name))(object_num) == 0:
            print("Successfully called parallel_new {} {}".format(object_class_name, object_num))

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
