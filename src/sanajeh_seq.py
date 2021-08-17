import random

# if we have different classes to instantiate then the allocator
# needs to keep track of objects of different classes (use dict?)
objects = None

class DeviceAllocator:

  @staticmethod
  def device_do(cls, func, *arg):
    global objects
    for obj in objects:
      getattr(obj, func.__name__)(*arg)

  @staticmethod
  def device_class(*cls):
    pass

  @staticmethod
  def parallel_do(cls, func, *args):
    pass  

  @staticmethod
  def rand_init(seed, sequence, offset):
    random.seed(sequence)

  @staticmethod
  def rand_uniform():
    return random.uniform(0,1)

  @staticmethod
  def array_size(array, size):
    pass

  # dummy new
  @staticmethod
  def new(cls, *args):
    pass

  # dummy destroy
  @staticmethod
  def destroy(obj):
    pass

class PyAllocator:

  def initialize(self):
    pass

  def uninitialize(self):
    pass

  def parallel_do(self, cls, func, *arg):
    global objects
    for obj in objects:
      getattr(obj, func.__name__)(*arg)

  def parallel_new(self, cls, object_num):
    global objects
    objects = [cls() for _ in range(object_num)]
    for i in range(object_num):
      getattr(objects[i], cls.__name__)(i)

  def do_all(self, cls, func):
    global objects
    for obj in objects:
      func(obj)