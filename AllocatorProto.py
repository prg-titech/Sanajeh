# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
from typing import List
from typing import Callable
from typing import TypeVar

T = TypeVar("T")
F = TypeVar('F', int, float, complex)


class A:
    def test(self) -> None:
        print(1)
    pass


# Device side allocator
class PyAllocatorT:
    numObjects: int
    classes: List[type]

    def __init__(self, object_num: int, *class_names) -> None:
        self.numObjects = object_num
        self.classes = class_names

    def device_do(self, t: type, func: Callable[[], None]):
        # return func()
        pass


# Host side allocator
class PyAllocatorTHandle:
    py_alloc_t: PyAllocatorT

    def __init__(self, pat: PyAllocatorT) -> None:
        self.py_alloc_t = pat

    def parallel_do(self, t: type, func: Callable[[], None]):
        pass


if __name__ == '__main__':
    p = PyAllocatorT(5, A)
    a = A()
    # print(type(a.device_do))
    p.device_do(A, a.test)
