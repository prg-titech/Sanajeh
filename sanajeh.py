# -*- coding: utf-8 -*-
# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
import os
from typing import Callable
import sys
import cffi

# Sanajeh package
import py2cpp
from config import CPP_FILE_PATH, HPP_FILE_PATH, SO_FILE_PATH, CDEF_FILE_PATH

ffi = cffi.FFI()


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

    # dummy array_size
    # (0,1]
    @staticmethod
    def array_size(array, size):
        pass

    # dummy new
    @staticmethod
    def new(cls, *args):
        pass

# Host side allocator
class PyAllocator:
    cpp_code: str = ""
    hpp_code: str = ""
    cdef_code: str = ""
    cpp_path: str = CPP_FILE_PATH
    hpp_path: str = HPP_FILE_PATH
    so_path: str = SO_FILE_PATH
    lib = None

    # compile python code to cpp code and .so file
    @staticmethod
    def compile(py_path, cpp_path=CPP_FILE_PATH, hpp_path=HPP_FILE_PATH):
        source = open(py_path, encoding="utf-8").read()
        PyAllocator.cpp_path = cpp_path
        PyAllocator.hpp_path = hpp_path
        codes = py2cpp.compile(source)
        PyAllocator.cpp_code = codes[0]
        PyAllocator.hpp_code = codes[1]
        PyAllocator.cdef_code = codes[2]

    @staticmethod
    def build(cpp_path=CPP_FILE_PATH, so_path=SO_FILE_PATH):
        """
        Compile cpp source file to .so file
        """
        if os.system("./build.sh " + cpp_path + " -o " + so_path) != 0:
            print("Build failed!", file=sys.stderr)
            sys.exit(1)

    # load the shared library and initialize the allocator on GPU
    @staticmethod
    def initialize(cdef_path=CDEF_FILE_PATH, so_path=SO_FILE_PATH):
        """
        Initialize ffi module
        """
        PyAllocator.cpp_code = open(PyAllocator.cpp_path, mode='r').read()
        PyAllocator.hpp_code = open(PyAllocator.hpp_path, mode='r').read()
        PyAllocator.cdef_code = open(cdef_path, mode='r').read()

        ffi.cdef(PyAllocator.cdef_code)
        PyAllocator.lib = ffi.dlopen(so_path)
        if PyAllocator.lib.AllocatorInitialize() == 0:
            pass
            # print("Successfully initialized the allocator through FFI.")
        else:
            print("Initialization failed!", file=sys.stderr)
            sys.exit(1)

    # Free all of the memory on GPU
    @staticmethod
    def uninitialize():
        """
        Initialize ffi module
        """
        if PyAllocator.lib.AllocatorUninitialize() == 0:
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
        callback_types = "void({})".format(", ".join(cls.__dict__['__annotations__'].values()))
        fields = ", ".join(cls.__dict__['__annotations__'])
        lambda_for_create_host_objects = eval("lambda {}: func(cls({}))".format(fields, fields), locals())
        lambda_for_callback = ffi.callback(callback_types, lambda_for_create_host_objects)

        if eval("PyAllocator.lib.{}_do_all".format(class_name))(lambda_for_callback) == 0:
            pass
            # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
        else:
            print("Do_all expression failed!", file=sys.stderr)
            sys.exit(1)
