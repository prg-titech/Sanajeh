# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
from typing import Dict
from typing import Callable
import ast

# Sanajeh package
from gencpp_ast import GenCppVisitor
from marker import Marker
import gencpp


# class A:
#     def __init__(self, a: int):
#         self.aa = a
#         print(self.aa)
#     pass
#
#     def add3(self):
#         self.aa += 3
#         print(self.aa)
#
#
# class B:
#     def __init__(self):
#         print("b")
#     pass


# Device side allocator
class PyAllocator:
    numObjects: int
    classDictionary: Dict

    def __init__(self):
        # self.numObjects = object_num
        self.classDictionary = {}
        # for i in class_names:
        #     self.classDictionary.setdefault(i, [])

    def new_(self, class_name, *args):
        # ffiでnew(device_allocator)を呼び出す
        ob = class_name(*args)
        self.classDictionary.setdefault(class_name.__name__, []).append(ob)
        return ob

    # can only be used in device code
    def device_do(self,  class_name, func: Callable, *args):
        # ffiでdevice_doを呼び出す
        for i in self.classDictionary[class_name.__name__]:
            func(i, *args)

    def parallel_do(self,  class_name, func: Callable, *args):
        # ffiでparallel_doを呼び出す
        for i in self.classDictionary[class_name.__name__]:
            func(i, *args)

    def parallel_new(self,  class_name, func: Callable, *args):
        # ffiでparallel_newを呼び出す
        pass


__pyallocator__ = PyAllocator()
cpp_code = None
hpp_code = None

def initialize(path):
    global cpp_code
    global hpp_code
    source = open(path, encoding="utf-8").read()
    # Generate python ast
    tree = ast.parse(source)
    # Mark device data on python ast
    rt = Marker.mark(tree)
    # Generate cpp ast from python ast
    gcv = GenCppVisitor(rt)
    cpp_node = gcv.visit(tree)
    # Generate cpp(hpp) code from cpp ast
    ctx = gencpp.BuildContext.create()
    cpp_code = cpp_node.buildCpp(ctx)
    hpp_code = cpp_node.buildHpp(ctx)
    cpp_path = './device_code/sanajeh_device_code.cu'
    hpp_path = './device_code/sanajeh_device_code.h'
    with open(cpp_path, mode='w') as cpp_file:
        cpp_file.write(cpp_code)
    with open(hpp_path, mode='w') as hpp_file:
        hpp_file.write(hpp_code)


# DEBUG propose
def printCppAndHpp():
    print(cpp_code)
    print("--------------------------------")
    print(hpp_code)


