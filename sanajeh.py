# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
from typing import Dict
from typing import Callable
FILE_NAME = "sanajeh_device_code"


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


# # Host side allocator
# class PyAllocatorTHandle:
#     __device_allocator__: PyAllocatorT
#
#     def __init__(self, pat: PyAllocatorT):
#         self.__device_allocator__ = pat
#
#     def device_do(self,  class_name, func, *args):
#         return self.__device_allocator__.device_do(class_name, func, *args)
#
#     def parallel_do(self,  class_name, func, *args):
#         return self.__device_allocator__.parallel_do(class_name, func, *args)


__pyallocator__ = PyAllocator()



# if __name__ == '__main__':
#     p = PyAllocatorT(5, A, B)
#     print(p.classDictionary)
#     p.new_(A, 1)
#     p.new_(A, 2)
#     p.new_(B)
#     p.new_(A, 3)
#     p.device_do(A, A.add3)
#     print(p.classDictionary)
#     print(p.classDictionary[A])
#     print(p.classDictionary[A][0])
#     # print(type(a.device_do))
