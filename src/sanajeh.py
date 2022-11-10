import os, sys
import py2cpp
import cffi
import random
from typing import Callable
from expander import RuntimeExpander

ffi = cffi.FFI()

cpu_flag = False
objects = {}  

# Device side allocator
class DeviceAllocator:

    class RandomState:
        def __init__(self):
            pass

    # Can be removed later
    """
    @staticmethod
    def curand_init(seed, sequence, offset):
        if cpu_flag:
        random.seed(seed + sequence)
        else:
        pass      

    @staticmethod
    def curand():
        if cpu_flag:
        return random.getrandbits(32)
        else:
        pass

    @staticmethod
    def curand_uniform():
        if cpu_flag:
        return random.uniform(0,1)
        else:
        pass
    
    def random_state(ref):
        pass
    """

    @staticmethod
    def new(cls, *args):
        if cpu_flag:
            new_object = cls()
            getattr(new_object, cls.__name__)(*args)
            global objects
            if cls in objects:
                objects[cls].append(new_object)
            else:
                objects[cls] = [new_object]
            return new_object
        else:
            pass
  
    @staticmethod
    def destroy(obj):
        if cpu_flag:
            if type(obj) in objects:
                objects[type(obj)].remove(obj)
                del obj
        else:
            pass
  
    """
    @staticmethod
    def device_class(*cls):
        pass
    """

    @staticmethod
    def device_do(cls, func, *args):
        if cpu_flag:
            for obj in objects[cls]:
                getattr(obj, func.__name__)(*args)
        else:
            pass

    @staticmethod
    def parallel_do(cls, func, *args):
        pass   

    @staticmethod
    def array(size):
        return [None] * size

class SeqAllocator:
    def __init__(self):
        global cpu_flag
        cpu_flag = True
        pass
    
    def initialize(self):
        pass
    
    def uninitialize(self):
        pass
  
    def parallel_do(self, cls, func, *args):
        if cls in objects:
            objects_to_check = objects[cls][:len(objects[cls])]
            for cls_object in objects_to_check:
                getattr(cls_object, func.__name__)(*args)

    def parallel_new(self, cls, object_num):
        cls_objects = [cls() for _ in range(object_num)]
        for i in range(object_num):
            getattr(cls_objects[i], cls.__name__)(i)
        global objects
        if cls in objects:
            for new_object in cls_objects:
                objects[cls].append(new_object)
        else:
            objects[cls] = cls_objects

    def do_all(self, cls, func):
        for obj in objects[cls]:
            func(obj)

class PyCompiler:
  
    def __init__(self, path: str):
        self.file_path = path
        self.file_name, _ = os.path.splitext(os.path.basename(path))
        self.dir_path = "device_code/{}".format(self.file_name)

    def compile(self, emit_py, emit_cpp, emit_hpp, emit_cdef):
        source = open(self.file_path, encoding="utf-8").read()
        py, cpp, hpp, cdef = py2cpp.compile(source, self.dir_path, self.file_name)

        if emit_py:
            print(py)
            return
        elif emit_cpp:
            print(cpp)
            return
        elif emit_hpp:
            print(hpp)
            return
        elif emit_cdef:
            print(cdef)
            return
                
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)
        compile_path: str = os.path.join(self.dir_path, self.file_name)
        with open(compile_path + ".cu", mode="w") as cpp_file:
            cpp_file.write(cpp)
        with open(compile_path + ".h", mode="w") as hpp_file:
            hpp_file.write(hpp)
        with open(compile_path + ".cdef", mode="w") as cdef_file:
            cdef_file.write(cdef)
        with open(compile_path + "_py.py", mode="w") as py_file:
            py_file.write(py)
    
        so_path: str = "{}/{}.so".format(self.dir_path, self.file_name)
        if os.system("src/build.sh " + "{}/{}.cu".format(self.dir_path, self.file_name) + " -o " + so_path) != 0:
            print("Build failed!", file=sys.stderr)
            sys.exit(1)   

class PyAllocator:
    file_path: str = ""
    file_name: str = ""
    cpp_code: str = ""
    hpp_code: str = ""
    cdef_code: str = ""
    lib = None

    expander: RuntimeExpander = RuntimeExpander()

    def __init__(self, path: str, name: str):
        self.file_name = name
        self.file_path = path
        self.py_code = ""

    # load the shared library and initialize the allocator on GPU
    def initialize(self):
        """
        Initialize ffi module
        """
        self.py_code = open("device_code/{}/{}_py.py".format(self.file_name, self.file_name), mode="r").read()
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
            # print("Successfully uninitialized the allocator through FFI.")
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

    def do_all(self, cls, func):
        name = cls.__name__
        if name not in self.expander.built.keys():
            self.expander.build_function(cls, self.py_code)
        callback_types = "void({})".format(", ".join(self.expander.flattened[name].values()))
        fields = ", ".join(self.expander.flattened[name])
        lambda_for_create_host_objects = eval("lambda {}: func(cls.__rebuild_{}({}))".format(fields, name, fields), locals())
        lambda_for_callback = ffi.callback(callback_types, lambda_for_create_host_objects)
        if eval("self.lib.{}_do_all".format(name))(lambda_for_callback) == 0:
            pass
        else:
            print("Do_all expression failed!", file=sys.stderr)
            sys.exit(1)   