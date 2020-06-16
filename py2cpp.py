# -*- coding: utf-8 -*-
# Mark all device functions

import ast

import type_converter
from config import INDENT, FILE_NAME
from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode
from gencpp_ast import GenCppVisitor
import gencpp


# Generate python call graph
class GenPyCallGraphVistor(ast.NodeVisitor):
    __root = CallGraph('root', None)
    __node_path = [__root]
    __current_node = None
    __variables = {}

    def __init__(self):
        self.__pp = Preprocessor(self.__root)

    @property
    def root(self):
        return self.__root

    # JUST FOR DEBUG
    @property
    def node_path(self):
        return self.__node_path

    # JUST FOR DEBUG
    @property
    def variables(self):
        return self.__variables

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        super(GenPyCallGraphVistor, self).visit(node)

    # todo other py files
    # def visit_Module(self, node):
    #     self.generic_visit(node)

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        if type(self.__current_node) is not CallGraph:
            print("Error, doesn't support nested classes")
            assert False
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name, None)
        if class_node is not None:
            # Program shouldn't come to here, which means a class is defined twice
            print("The class {} is defined twice.".format(class_name))
            assert False
        class_node = ClassNode(node.name, None)
        self.__current_node.declared_classes.add(class_node)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name
        if type(self.__current_node) is not CallGraph and type(self.__current_node) is not ClassNode:
            print("Error, doesn't support nested functions")
            assert False
        func_node = self.__current_node.GetFunctionNode(func_name, None)
        if func_node is not None:
            # Program shouldn't come to here, which means a function is defined twice
            print("The function {} is defined twice.".format(func_name))
            assert False
        func_node = FunctionNode(func_name, None)
        self.__current_node.declared_functions.add(func_node)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Add arguments to the environment
    def visit_arguments(self, node):
        if type(self.__current_node) is not FunctionNode:
            print('Unexpected node "{}"'.format(self.__current_node.name))
            assert False
        for arg in node.args:
            if arg.arg == "self":
                continue
            self.__variables.setdefault(self.__current_node.id, []).append(arg.arg)

    # Add global variables to the environment
    def visit_Global(self, node):
        for global_variable in node.names:
            self.__variables.setdefault(self.__current_node.id, []).append(global_variable)
            var_node = self.__root.GetVariableNode(global_variable, None, None)
            if var_node is None:
                print("The global variable {} is not existed.".format(global_variable))
                assert False
            self.__current_node.called_variables.add(var_node)

    # Create nodes for variables without type annotation
    def visit_Assign(self, node):
        for var in node.targets:
            var_name = None
            # todo id_name
            id_name = None

            if type(var) is ast.Attribute:
                var_name = var.attr
                # print(var_name, var.value.id)
                # todo Attribute variables(self should refer to the class not in the current block),
                # todo haven't thought about other occasions
                if var.value.id == 'self':
                    pass
            elif type(var) is ast.Name:
                var_name = var.id
                self.__variables.setdefault(self.__current_node.id, [])
                # print(self.__variables)
                if var_name not in self.__variables[self.__current_node.id]:
                    var_node = VariableNode(var_name, id_name, None)
                    self.__current_node.declared_variables.add(var_node)
                    self.__variables[self.__current_node.id].append(var_name)

        self.generic_visit(node)

    # Create nodes for variables with type annotation
    def visit_AnnAssign(self, node):
        var = node.target
        ann = node.annotation.id

        var_name = None
        # todo id_name
        id_name = None

        if type(var) is ast.Attribute:
            var_name = var.attr
            # print(var_name, var.value.id)
            if var.value.id == 'self':
                pass
            # todo Attribute variables(self should refer to the class not in the current block),
            # todo haven't thought about other occasions
        elif type(var) is ast.Name:
            var_name = var.id
            self.__variables.setdefault(self.__current_node.id, [])
            # print(self.__variables)
            if var_name not in self.__variables[self.__current_node.id]:
                var_node = VariableNode(var_name, id_name, ann)
                self.__current_node.declared_variables.add(var_node)
                self.__variables[self.__current_node.id].append(var_name)
            else:
                var_node = self.__current_node.GetVariableNode(var_name, id_name, ann)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.__variables.setdefault(self.__current_node.id, [])
        if node.id in self.__variables[self.__current_node.id]:
            return
        for annotate_location_node in self.__node_path[-2::-1]:
            self.__variables.setdefault(annotate_location_node.id, [])
            if node.id in self.__variables[annotate_location_node.id]:
                var_node = annotate_location_node.GetVariableNode(node.id, None, None)
                if var_node is None:
                    print('Unexpected error, can not find variable "{}"', node.id)
                    assert False
                self.__current_node.called_variables.add(var_node)
                break

    # mark all device data in the CallGraph
    def mark_device_data(self, node):
        self.__pp.visit(node)
        if not self.__pp.has_device_data:
            return False
        self.__root.MarkDeviceDataByClassName(self.__pp.classes)
        return True

    def build_parallel_do_cpp(self):
        return '\n' + '\n\n'.join(self.__pp.cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_do_codes)

    def build_parallel_new_cpp(self):
        return '\n' + '\n\n'.join(self.__pp.cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_new_codes)


# Find device class in python code and compile parallel_do expressions into c++ ones
class Preprocessor(ast.NodeVisitor):
    __root: CallGraph
    __classes = []
    __node_path = []
    has_device_data = False
    __is_root = True  # the flag of whether visiting the root node of python ast
    __node_root = None  # the root node of python ast
    __cpp_parallel_do_codes = []
    __hpp_parallel_do_codes = []
    __cdef_parallel_do_codes = []
    __cpp_parallel_new_codes = []
    __hpp_parallel_new_codes = []
    __cdef_parallel_new_codes = []

    @property
    def cpp_parallel_do_codes(self):
        return self.__cpp_parallel_do_codes

    @property
    def hpp_parallel_do_codes(self):
        return self.__hpp_parallel_do_codes

    @property
    def cdef_parallel_do_codes(self):
        return self.__cdef_parallel_do_codes

    @property
    def cpp_parallel_new_codes(self):
        return self.__cpp_parallel_new_codes

    @property
    def hpp_parallel_new_codes(self):
        return self.__hpp_parallel_new_codes

    @property
    def cdef_parallel_new_codes(self):
        return self.__cdef_parallel_new_codes

    # Collect information of parallel_new and build codes for that function in c++
    class ParallelNewAnalyzer:
        def __init__(self, class_name):
            self.__class_name = class_name  # The class of the object

        def buildCpp(self):
            parallel_new_expr = INDENT + "allocator_handle->parallel_new<{}>(object_num);\n".format(self.__class_name)
            return_expr = INDENT + "return 0;"
            return 'extern "C" int parallel_new_{}(int object_num){{\n'.format(self.__class_name) \
                   + parallel_new_expr \
                   + return_expr \
                   + "\n}"

        def buildHpp(self):
            return 'extern "C" int parallel_new_{}(int object_num);'.format(self.__class_name)

        def buildCdef(self):
            return 'int parallel_new_{}(int object_num);'.format(self.__class_name)

    # Collect information of those functions used in the parallel_do function, and build codes for that function in c++
    class ParallelDoAnalyzer(ast.NodeVisitor):
        def __init__(self, rt, class_name, func_class_name, func_name):
            self.__root = rt
            self.__node_path = [rt]
            self.__current_node = None
            self.__object_class_name = class_name  # The class of the object
            self.__func_class_name = func_class_name  # The class of the function executed
            self.__func_name = func_name
            self.__args = {}

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            super(Preprocessor.ParallelDoAnalyzer, self).visit(node)

        def visit_ClassDef(self, node):
            if node.name != self.__func_class_name:
                return
            class_name = node.name
            class_node = self.__current_node.GetClassNode(class_name, None)
            if class_node is None:
                # Program shouldn't come to here, which means the class does not exist
                print("The class {} is not exist.".format(class_name))
                assert False
            self.__node_path.append(class_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_FunctionDef(self, node):
            func_name = node.name
            func_node = self.__current_node.GetFunctionNode(func_name, None)
            if func_node is None:
                # Program shouldn't come to here, which means the function does not exist
                print("The function {} does not exist.".format(func_name))
                assert False
            if func_name != self.__func_name or self.__current_node.name != self.__func_class_name:
                return
            for arg_ in node.args.args:
                if arg_.arg == 'self':
                    continue
                self.__args[arg_.arg] = arg_.annotation.id

        def buildCpp(self):
            arg_strs = []
            for arg_ in self.__args:
                arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))
            parallel_do_expr = INDENT + "allocator_handle->parallel_do<{}, &{}::{}>({});\n".format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ", ".join(self.__args)
            )
            return_expr = INDENT + "return 0;"

            return 'extern "C" int {}_{}_{}({}){{\n'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ", ".join(arg_strs)) \
                   + parallel_do_expr \
                   + return_expr \
                   + "\n}"

        def buildHpp(self):
            arg_strs = []
            for arg_ in self.__args:
                arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))

            return 'extern "C" int {}_{}_{}({});'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ",".join(arg_strs)
            )

        def buildCdef(self):
            arg_strs = []
            for arg_ in self.__args:
                arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))

            return 'int {}_{}_{}({});'.format(
                self.__object_class_name,
                self.__func_class_name,
                self.__func_name,
                ",".join(arg_strs)
            )

    def __init__(self, rt: CallGraph):
        self.__root = rt
        self.__node_path.append(rt)
        self.__current_node = None

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        super(Preprocessor, self).visit(node)

    @property
    def classes(self):
        return self.__classes

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} is not exist.".format(class_name))
            assert False
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.__current_node.GetFunctionNode(func_name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function does not exist
            print("The function {} does not exist.".format(func_name))
            assert False
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Analyze function calling relationships
    def visit_Call(self, node):
        # Find device classes
        if type(node.func) is ast.Attribute and node.func.value.id == "__pyallocator__":
            if node.func.attr == 'parallel_new':
                self.has_device_data = True
                if node.args[0].id not in self.__classes:
                    self.__classes.append(node.args[0].id)
                pna = self.ParallelNewAnalyzer(node.args[0].id)
                self.__cpp_parallel_new_codes.append(pna.buildCpp())
                self.__hpp_parallel_new_codes.append(pna.buildHpp())
                self.__cdef_parallel_new_codes.append(pna.buildCdef())
            elif node.func.attr == 'parallel_do':
                pda = self.ParallelDoAnalyzer(self.__root,
                                              node.args[0].id,
                                              node.args[1].value.id,
                                              node.args[1].attr)
                pda.visit(self.__node_root)
                self.__cpp_parallel_do_codes.append(pda.buildCpp())
                self.__hpp_parallel_do_codes.append(pda.buildHpp())
                self.__cdef_parallel_do_codes.append(pda.buildCdef())

        func_name = None
        # todo id_name maybe class name
        id_name = None
        call_node = None

        # ignore call other functions in the same class
        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            id_name = node.func.value.id
            if id_name == 'self':
                return
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        for parent_node in self.__node_path[::-1]:
            # todo id_name
            if type(parent_node) is FunctionNode:
                continue
            call_node = parent_node.GetFunctionNode(func_name, id_name)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionNode(func_name, id_name)
            self.__root.library_functions.add(call_node)
        self.__current_node.called_functions.add(call_node)
        self.generic_visit(node)


def compile(source_code, cpp_path, hpp_path):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
        cpp_path:       path for the compiled c++ source file
        hpp_path:       path for the compiled c++ header file
    """
    # Generate python ast
    py_ast = ast.parse(source_code)
    # Generate python call graph
    gpcgv = GenPyCallGraphVistor()
    gpcgv.visit(py_ast)
    # Mark all device data on the call graph
    if not gpcgv.mark_device_data(py_ast):
        print("No device data found")
        assert False
    # Generate cpp ast from python ast
    gcv = GenCppVisitor(gpcgv.root)
    cpp_node = gcv.visit(py_ast)
    # Generate cpp(hpp) code from cpp ast
    ctx = gencpp.BuildContext.create()
    # Expression needed for DynaSOAr API
    cpp_include_expr = '#include "{}.h"\n\n'.format(FILE_NAME)
    allocator_declaration = "AllocatorHandle<AllocatorT>* allocator_handle;\n" \
                            "__device__ AllocatorT* device_allocator;\n"
    precompile_expr = "#ifndef SANAJEH_DEVICE_CODE_H" \
                      "\n#define SANAJEH_DEVICE_CODE_H" \
                      "\n#define KNUMOBJECTS 64*64*64*64"
    hpp_include_expr = '\n\n#include <curand_kernel.h>\n#include "dynasoar.h"'
    init_cpp = ['\n\nextern "C" int AllocatorInitialize(){\n',
                INDENT +
                "allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);\n",
                INDENT +
                "AllocatorT* dev_ptr = allocator_handle->device_pointer();\n",
                INDENT +
                "cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);\n",
                INDENT +
                "return 0;\n"
                "}"
                ]
    init_hpp = '\nextern "C" int AllocatorInitialize();\n'
    init_cdef = '\nint AllocatorInitialize();\n'
    endif_expr = "\n#endif"

    # Source code
    cpp_code = cpp_include_expr \
               + allocator_declaration \
               + cpp_node.buildCpp(ctx) \
               + gpcgv.build_parallel_do_cpp() \
               + gpcgv.build_parallel_new_cpp()\
               + "".join(init_cpp)
    # Header code
    hpp_code = precompile_expr \
               + hpp_include_expr \
               + cpp_node.buildHpp(ctx) \
               + gpcgv.build_parallel_do_hpp() \
               + gpcgv.build_parallel_new_hpp() \
               + init_hpp \
               + endif_expr

    # Codes for cffi cdef() function
    cdef_code = gpcgv.build_parallel_do_cdef() + gpcgv.build_parallel_new_cdef() + init_cdef

    with open(cpp_path, mode='w') as cpp_file:
        cpp_file.write(cpp_code)
    with open(hpp_path, mode='w') as hpp_file:
        hpp_file.write(hpp_code)
    return cpp_code, hpp_code, cdef_code