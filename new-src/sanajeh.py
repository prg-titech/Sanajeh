import os, sys, ast
import astunparse
import cffi
import random

import call_graph as cg
from inheritance import ClassInspectorMain
from transformer import Normalizer, Inliner, Eliminator, FieldSynthesizer
from py2cpp import Preprocessor, CppVisitor, HppVisitor, INDENT
from expander import RuntimeExpander

ffi = cffi.FFI()

cpu_flag = False
objects = {}  

def device(cls):
    return cls

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
        classes = [c for c in objects]
        for cls_check in classes:
            if subtype(cls_check, cls):
                objects_to_check = objects[cls_check][:len(objects[cls_check])]
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

    def compile(self, emit_py, emit_cpp, emit_hpp, emit_cdef, verbose):
        source = open(self.file_path, encoding="utf-8").read()
        if not os.path.isdir("device_code"):
            os.mkdir("device_code")
        py, cpp, hpp, cdef = compile(source, self.dir_path, self.file_name, verbose)

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
        if os.system("new-src/build.sh " + "{}/{}.cu".format(self.dir_path, self.file_name) + " -o " + so_path) != 0:
            print("Build failed!", file=sys.stderr)
            sys.exit(1)

class PyAllocator:
    
    file_name: str = ""
    py_code: str = ""
    cpp_code: str = ""
    hpp_code: str = ""
    cdef_code: str = ""
    lib = None
    expander: RuntimeExpander = RuntimeExpander()

    def __init__(self, name: str):
        self.file_name = name

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
            # print("Successfully initialized the allocator through FFI.")
            pass
        else:
            # print("Initialization failed!", file=sys.stderr)
            sys.exit(1)

    def uninitialize():
        """
        Initialize ffi module
        """
        if self.lib.AllocatorUninitialize() == 0:
            # print("Successfully uninitialized the allocator through FFI.")
            pass
        else:
            # print("Initialization failed!", file=sys.stderr)
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
            # print("Successfully called parallel_do {} {} {}".format(object_class_name, func_class_name, func_name))
            pass
        else:
            # print("Parallel_do expression failed!", file=sys.stderr)
            sys.exit(1)

    def parallel_new(self, cls, object_num):
        """
        Parallelly create objects of a class
        """
        object_class_name = cls.__name__
        if eval("self.lib.parallel_new_{}".format(object_class_name))(object_num) == 0:
            # print("Successfully called parallel_new {} {}".format(object_class_name, object_num))
            pass
        else:
            # print("Parallel_new expression failed!", file=sys.stderr)
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
            # print("Do_all expression failed!", file=sys.stderr)
            sys.exit(1)   

def subtype(cls_check, cls):
    if cls_check == cls:
        return True
    for super_cls_check in cls_check.__bases__:
        if subtype(super_cls_check, cls):
            return True
    return False

def compile(source_code, dir_path, file_name, verbose):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name

    classinspector = ClassInspectorMain(source_code, verbose)
    classinspector.Start()

    # Generate python ast
    py_ast = classinspector.ast_code

    # Generate python call graph and mark device data
    cgv = cg.CallGraphVisitor()
    cgv.visit(py_ast)
    mdv = cg.MarkDeviceVisitor()
    mdv.visit(cgv.root)

    # Transformation passes
    normalizer = Normalizer(mdv.root)
    ast.fix_missing_locations(normalizer.visit(py_ast))
    inliner = Inliner(normalizer.root)
    ast.fix_missing_locations(inliner.visit(py_ast))
    eliminator = Eliminator(inliner.root)
    ast.fix_missing_locations(eliminator.visit(py_ast))
    synthesizer = FieldSynthesizer(eliminator.root)
    ast.fix_missing_locations(synthesizer.visit(py_ast))

    # Rebuild the call graph after transformation
    recgv = cg.CallGraphVisitor()
    recgv.visit(py_ast)
    remdv = cg.MarkDeviceVisitor()
    remdv.visit(recgv.root)

    # Preprocessor (find device class in python code and compile parallel_do expressions into c++ ones)
    pp = Preprocessor(remdv.root)
    pp.visit(py_ast)
    cv = CppVisitor(pp.root)
    hv = HppVisitor(pp.root)

    cpp_include_expr = '#include "{}.h"\n\n'.format(FILE_NAME)
    allocator_declaration = "AllocatorHandle<AllocatorT>* allocator_handle;\n" \
                            "__device__ AllocatorT* device_allocator;\n"
    init_cpp = ['\n\nextern "C" int AllocatorInitialize(){\n',
                INDENT + "allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);\n",
                INDENT + "AllocatorT* dev_ptr = allocator_handle->device_pointer();\n",
                INDENT + "cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);\n",
                pp.build_global_device_variables_init(),
                INDENT + "return 0;\n" 
                "}"
                ]
    unit_cpp = ['\n\nextern "C" int AllocatorUninitialize(){\n',
                pp.build_global_device_variables_unit(),
                INDENT +
                "return 0;\n"
                "}"
                ]

    cpp_code = cpp_include_expr \
                + allocator_declaration \
                + cv.visit(py_ast) \
                + pp.build_do_all_cpp() \
                + pp.build_parallel_do_cpp() \
                + pp.build_parallel_new_cpp() \
                + "".join(init_cpp) \
                + "".join(unit_cpp)

    endif_expr = "\n#endif"
    precompile_expr = "#ifndef SANAJEH_DEVICE_CODE_H" \
                      "\n#define SANAJEH_DEVICE_CODE_H" \
                      "\n#define KNUMOBJECTS 64*64*64*64"
    hpp_include_expr = '\n\n#include <curand_kernel.h>\n#include "dynasoar.h"'
    init_hpp = '\nextern "C" int AllocatorInitialize();\n'
    unit_hpp = 'extern "C" int AllocatorUninitialize();\n'

    hpp_code = precompile_expr \
                + hpp_include_expr \
                + hv.visit(py_ast) \
                + pp.build_do_all_hpp() \
                + pp.build_parallel_do_hpp() \
                + pp.build_parallel_new_hpp() \
                + init_hpp \
                + unit_hpp \
                + endif_expr

    init_cdef = '\nint AllocatorInitialize();'
    unit_cdef = '\nint AllocatorUninitialize();'

    cdef_code = pp.build_parallel_do_cdef() \
                + pp.build_parallel_new_cdef() \
                + init_cdef \
                + pp.build_do_all_cdef() \
                + unit_cdef

    new_py_ast = astunparse.unparse(py_ast)

    return new_py_ast, cpp_code, hpp_code, cdef_code