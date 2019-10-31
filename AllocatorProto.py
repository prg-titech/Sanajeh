# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
from typing import List


class A:
    pass


# Device side allocator
class PyAllocatorT:
    numObjects: int
    classes: List[type]

    def __init__(self, object_num: int, *class_names: type):
        self.numObjects = object_num
        self.classes = class_names

    # # lambda_test = lambda n,x :n.x()
    # def device_do(self, func: classmethod):
    #
    #     pass


# Host side allocator
class PyAllocatorHandle:
    pass


if __name__ == '__main__':
    a = PyAllocatorT(5, A, A)
    print(type(a.device_do))
