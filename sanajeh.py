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


def initialize(path):
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    rt = Marker.mark(tree)
    gcv = GenCppVisitor(rt)
    cpp_node = gcv.visit(tree)
    ctx = gencpp.BuildContext.create()
    cpp_code = cpp_node.buildCpp(ctx)
    hpp_code = cpp_node.buildHpp(ctx)
    print(cpp_code)


