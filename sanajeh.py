# -*- coding: utf-8 -*-
# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr

from typing import Dict
from typing import Callable

# Sanajeh package
import py2cpp
from config import CPP_FILE_PATH, HPP_FILE_PATH


# Device side allocator
class PyAllocator:
    numObjects: int
    classDictionary: Dict

    def __init__(self):
        # self.numObjects = object_num
        self.classDictionary = {}
        # for i in class_names:
        #     self.classDictionary.setdefault(i, [])

    # def parallel_new(self, class_name, *args):
    #     # ffiでnew(device_allocator)を呼び出す
    #     ob = class_name(*args)
    #     self.classDictionary.setdefault(class_name.__name__, []).append(ob)
    #     return ob

    # can only be used in device code
    def device_do(self, class_name, func: Callable, *args):
        # ffiでdevice_doを呼び出す
        for i in self.classDictionary[class_name.__name__]:
            func(i, *args)

    def parallel_do(self, class_name, func: Callable, *args):
        # ffiでparallel_doを呼び出す
        for i in self.classDictionary[class_name.__name__]:
            func(i, *args)

    def parallel_new(self, class_name, object_num):
        # ffiでparallel_newを呼び出す
        pass

    def rand_init(self, seed, sequence, offset):
        pass

    # (0,1]
    def rand_uniform(self):
        # curand_uniform
        pass


__pyallocator__ = PyAllocator()
cpp_code = None
hpp_code = None


def initialize(path):
    global cpp_code
    global hpp_code
    source = open(path, encoding="utf-8").read()
    codes = py2cpp.compile(source, CPP_FILE_PATH, HPP_FILE_PATH)
    cpp_code = codes[0]
    hpp_code = codes[1]


# DEBUG propose
def printCppAndHpp():
    print(cpp_code)
    print("--------------------------------")
    print(hpp_code)
