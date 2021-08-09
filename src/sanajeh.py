import os, sys
import py2cpp_

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

    # dummy destroy
    @staticmethod
    def destroy(obj):
        pass

class PyCompiler:
  file_path: str = ""
  file_name: str = ""
  dir_path: str = ""

  cpp_code: str = ""
  hpp_code: str = ""
  cdef_code: str = ""

  def __init__(self, pth: str, nme: str):
    self.file_path = pth
    self.file_name = nme
    self.dir_path = "device_code/{}".format(self.file_name)

  def compile(self):
    source = open(self.file_path, encoding="utf-8").read()
    codes = py2cpp_.compile(source, self.dir_path, self.file_name)
    self.cpp_code = codes[0]
    self.hpp_code = codes[1]
    self.cdef_code = codes[2]

  def build(self):
    """
    Compile cpp source file to .so file
    """
    so_path: str = "{}/{}.so".format(self.dir_path, self.file_name)
    if os.system("src/build.sh " + "{}/{}.cu".format(self.dir_path, self.file_name) + " -o " + so_path) != 0:
      print("Build failed!", file=sys.stderr)
      sys.exit(1)

  def printCppAndHpp(self):
    print(self.cpp_code)
    print("--------------------------------")
    print(self.hpp_code)

  def printCdef(self):
    print(self.cdef_code)

import cffi
ffi = cffi.FFI()

from typing import Callable

class PyAllocator:
  file_name: str = ""
  cpp_code: str = ""
  hpp_code: str = ""
  cdef_code: str = ""
  lib = None

  def __init__(self, name: str):
    self.file_name = name

  # load the shared library and initialize the allocator on GPU
  def initialize(self):
    """
    Initialize ffi module
    """
    self.cpp_code = open("device_code/{}/{}.cu".format(self.file_name, self.file_name), mode="r").read()
    self.hpp_code = open("device_code/{}/{}.h".format(self.file_name, self.file_name), mode="r").read()
    self.cdef_code = open("device_code/{}/{}.cdef".format(self.file_name, self.file_name), mode="r").read()

    ffi.cdef(self.cdef_code)
    self.lib = ffi.dlopen("device_code/{}/{}.so".format(self.file_name, self.file_name))
    if self.lib.AllocatorInitialize() == 0:
      pass
      #print("Successfully initialized the allocator through FFI.")
    else:
      print("Initialization failed!", file=sys.stderr)
      sys.exit(1)

  # Free all of the memory on GPU
  def uninitialize():
    """
    Initialize ffi module
    """
    if self.lib.AllocatorUninitialize() == 0:
      pass
      # print("Successfully initialized the allocator through FFI.")
    else:
      print("Initialization failed!", file=sys.stderr)
      sys.exit(1)

  def parallel_do(self, cls, func, *args):
    """
    Parallelly run a function on all objects of a class.
    """
    object_class_name = cls.__name__
    func_str = func.__qualname__.split(".")
    # todo nested class exception
    func_class_name = func_str[0]
    func_name = func_str[1]
    # todo args
    if eval("self.lib.{}_{}_{}".format(object_class_name, func_class_name, func_name))() == 0:
      pass
      # print("Successfully called parallel_do {} {} {}".format(object_class_name, func_class_name, func_name))
    else:
      print("Parallel_do expression failed!", file=sys.stderr)
      sys.exit(1)

  def parallel_new(self, cls, object_num):
    """
    Parallelly create objects of a class
    """
    object_class_name = cls.__name__
    if eval("self.lib.parallel_new_{}".format(object_class_name))(object_num) == 0:
      pass
      # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
    else:
      print("Parallel_new expression failed!", file=sys.stderr)
      sys.exit(1)

  def do_all(cls, func):
    """
    Run a function which is used to received the fields on all object of a class.
    """
    class_name = cls.__name__
    callback_types = "void({})".format(", ".join(cls.__dict__['__annotations__'].values()))
    fields = ", ".join(cls.__dict__['__annotations__'])
    lambda_for_create_host_objects = eval("lambda {}: func(cls({}))".format(fields, fields), locals())
    lambda_for_callback = ffi.callback(callback_types, lambda_for_create_host_objects)

    if eval("self.lib.{}_do_all".format(class_name))(lambda_for_callback) == 0:
      pass
      # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
    else:
      print("Do_all expression failed!", file=sys.stderr)
      sys.exit(1)