# This is a python version prototype of the AllocatorHandle(host side API) and AllocatorT(device side API) in DynaSOAr
from typing import List
from typing import Dict
from typing import Callable
from typing import TypeVar


class A:
    def __init__(self):
        print("a")
    pass


class B:
    def __init__(self):
        print("b")
    pass


# Device side allocator
class PyAllocatorT:
    # T = TypeVar("T")
    numObjects: int
    # classDictionary: Dict[T, List[T]]

    def __init__(self, object_num: int, *class_names: type) -> None:
        self.numObjects = object_num
        self.classDictionary = {}
        for i in class_names:
            self.classDictionary.setdefault(i, [])

    def new_(self, class_name: type) -> None:
        self.classDictionary.setdefault(class_name, []).append(class_name())

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
    p = PyAllocatorT(5, A, B)
    print(p.classDictionary)
    p.new_(A)
    p.new_(A)
    p.new_(B)
    p.new_(A)
    print(p.classDictionary)
    print(p.classDictionary[A])
    print(p.classDictionary[A][0])
    # print(type(a.device_do))
