import os, sys
import py2cpp
import cffi
import random
from typing import Callable

ffi = cffi.FFI()

cpu_flag = False
objects = None  

class PyCompiler:
  file_path: str = ""
  file_name: str = ""
  dir_path: str = ""

  #cpp_code: str = ""
  #hpp_code: str = ""
  #cdef_code: str = ""

  def __init__(self, pth: str, nme: str):
    self.file_path = pth
    self.file_name = nme
    self.dir_path = "device_code/{}".format(self.file_name)

  def compile(self):
    source = open(self.file_path, encoding="utf-8").read()
    codes = py2cpp.compile(source, self.dir_path, self.file_name)
    #self.cpp_code = codes[0]
    #self.hpp_code = codes[1]
    #self.cdef_code = codes[2]

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

# Device side allocator
class DeviceAllocator:
  @staticmethod
  def device_do(cls, func, *args):
    if cpu_flag:
      for obj in objects:
        getattr(obj, func.__name__)(*args)
    else:
      pass

  @staticmethod
  def device_class(*cls):
    pass

  @staticmethod
  def parallel_do(cls, func, *args):
    pass

  @staticmethod
  def rand_init(seed, sequence, offset):
    if cpu_flag:
      random.seed(sequence)
    else:
      pass

  # (0,1]
  @staticmethod
  def rand_uniform():
    if cpu_flag:
      return random.uniform(0,1)
    else:
      pass

  @staticmethod
  def array_size(array, size):
    pass

  @staticmethod
  def new(cls, *args):
    pass

  @staticmethod
  def destroy(obj):
    pass

class PyAllocator:
  file_path: str = ""
  file_name: str = ""
  cpp_code: str = ""
  hpp_code: str = ""
  cdef_code: str = ""
  lib = None

  def __init__(self, path: str, name: str, flag: bool):
    self.file_name = name
    self.file_path = path
    global cpu_flag    
    cpu_flag = flag

  # load the shared library and initialize the allocator on GPU
  def initialize(self):
    if cpu_flag:
      pass
    else:
      """
      Compilation before initializing ffi
      """
      compiler: PyCompiler = PyCompiler(self.file_path, self.file_name)
      compiler.compile()
      compiler.build()

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
    if cpu_flag:
      pass
    else:
      """
      Initialize ffi module
      """
      if self.lib.AllocatorUninitialize() == 0:
        pass
        # print("Successfully uninitialized the allocator through FFI.")
      else:
        print("Initialization failed!", file=sys.stderr)
        sys.exit(1)

  def parallel_do(self, cls, func, *args):
    if cpu_flag:
      for obj in objects:
        getattr(obj, func.__name__)(*args)
    else:
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
    if cpu_flag:
      global objects
      objects = [cls.__new__(cls) for _ in range(object_num)]
      for i in range(object_num):
        getattr(objects[i], cls.__name__)(i)
    else:
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
  
  def do_all(self, cls, func):
    if cpu_flag:
      for obj in objects:
        func(obj)
    else:
      def expand(name):
        field_map = {}
        module = name.__dict__["__module__"]
        if "__annotations__" in name.__dict__.keys():
          for field, ftype in name.__dict__["__annotations__"].items():
            if ftype in ["int", "float", "bool"]: 
              field_map[field] = ftype
            else:
              """
              Expand the nested classes on the assumptions that the field type
              is defined in the same module as the parent class.
              """
              for nested_field, nested_ftype in expand(getattr(__import__(module), ftype)).items():
                field_map[field + "_" + nested_field] = nested_ftype
        return field_map        
      """
      Run a function which is used to received the fields on all object of a class.
      """
      class_name = cls.__name__
      #callback_types = "void({})".format(", ".join(cls.__dict__['__annotations__'].values()))
      #fields = ", ".join(cls.__dict__['__annotations__'])      
      callback_types = "void({})".format(", ".join(expand(cls).values()))
      fields = ", ".join(expand(cls))
      lambda_for_create_host_objects = eval("lambda {}: func(cls({}))".format(fields, fields), locals())
      lambda_for_callback = ffi.callback(callback_types, lambda_for_create_host_objects)

      if eval("self.lib.{}_do_all".format(class_name))(lambda_for_callback) == 0:
        pass
        # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
      else:
        print("Do_all expression failed!", file=sys.stderr)
        sys.exit(1)