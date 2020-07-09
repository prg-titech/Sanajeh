# -*- coding: utf-8 -*-
# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr

from typing import Callable
import sys
import cffi

# Sanajeh package
import py2cpp
import build
from config import CPP_FILE_PATH, HPP_FILE_PATH, SO_FILE_PATH


# Device side allocator
class DeviceAllocator:
    """
    Includes no implementations. The only purpose of this class is to provide special syntax for python device codes
    """

    # dummy device_do
    @staticmethod
    def device_do(cls, func, *args):
        pass

    # dummy for recognizing device classes
    @staticmethod
    def device_class(*cls):
        pass

    # dummy for recognizing parallel_do
    @staticmethod
    def parallel_do(cls, func, *args):
        pass

    # dummy rand_init
    @staticmethod
    def rand_init(seed, sequence, offset):
        pass

    # dummy rand_uniform
    # (0,1]
    @staticmethod
    def rand_uniform():
        pass


# Host side allocator
class PyAllocator:
    cpp_code: str = None
    hpp_code: str = None
    cpp_path: str = None
    hpp_path: str = None
    so_path: str = None
    cdef_code: str = None
    lib = None

    # compile python code to cpp code and .so file
    @staticmethod
    def compile(py_path, cpp_path=CPP_FILE_PATH, hpp_path=HPP_FILE_PATH):
        source = open(py_path, encoding="utf-8").read()
        PyAllocator.cpp_path = cpp_path
        PyAllocator.hpp_path = cpp_path
        codes = py2cpp.compile(source, cpp_path, hpp_path)
        PyAllocator.cpp_code = codes[0]
        PyAllocator.hpp_code = codes[1]
        PyAllocator.cdef_code = codes[2]

    @staticmethod
    def build(so_path=SO_FILE_PATH):
        """
        Compile cpp source file to .so file
        """
        PyAllocator.so_path = so_path
        if build.run(PyAllocator.cpp_path, so_path) != 0:
            print("Build failed!", file=sys.stderr)
            sys.exit(1)

    # load the shared library and initialize the allocator on GPU
    @staticmethod
    def initialize():
        """
        Initialize ffi module
        """
        ffi = cffi.FFI()
        ffi.cdef(PyAllocator.cdef_code)
        PyAllocator.lib = ffi.dlopen(PyAllocator.so_path)
        if PyAllocator.lib.AllocatorInitialize() == 0:
            pass
            # print("Successfully initialized the allocator through FFI.")
        else:
            print("Initialization failed!", file=sys.stderr)
            sys.exit(1)

    # DEBUG propose
    @staticmethod
    def printCppAndHpp():
        print(PyAllocator.cpp_code)
        print("--------------------------------")
        print(PyAllocator.hpp_code)

    # DEBUG propose
    @staticmethod
    def printCdef():
        print(PyAllocator.cdef_code)

    @staticmethod
    def parallel_do(cls, func, *args):
        """
        Parallelly run a function on all objects of a class.
        """
        object_class_name = cls.__name__
        func_str = func.__qualname__.split(".")
        # todo nested class exception
        func_class_name = func_str[0]
        func_name = func_str[1]
        # todo args
        if eval("PyAllocator.lib.{}_{}_{}".format(object_class_name, func_class_name, func_name))() == 0:
            pass
            # print("Successfully called parallel_do {} {} {}".format(object_class_name, func_class_name, func_name))
        else:
            print("Parallel_do expression failed!", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def parallel_new(cls, object_num):
        """
        Parallelly create objects of a class
        """
        object_class_name = cls.__name__
        if eval("PyAllocator.lib.parallel_new_{}".format(object_class_name))(object_num) == 0:
            pass
            # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
        else:
            print("Parallel_new expression failed!", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def do_all(cls, func):
        """
        Run a function which is used to received the fields on all object of a class.
        """
        class_name = cls.__name__
        if eval("PyAllocator.lib.{}_do_all".format(class_name))(func) == 0:
            pass
            # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
        else:
            print("Do_all expression failed!", file=sys.stderr)
            sys.exit(1)
