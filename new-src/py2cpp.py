# -*- coding: utf-8 -*-

import ast, os, sys
import hashlib
import six

import build_cpp as cpp
import call_graph as cg
from transformer import Normalizer, Inliner, Eliminator, FieldSynthesizer

import astunparse

INDENT: str = "\t"

class ParallelNewBuilder:
    def __init__(self, class_name):
        self.class_name = class_name
    
    def buildCpp(self):
        parallel_new_expr = INDENT + "allocator_handle->parallel_new<{}>(object_num);\n".format(self.class_name)
        return_expr = INDENT + "return 0;"
        return 'extern "C" int parallel_new_{}(int object_num){{\n'.format(self.class_name) \
                + parallel_new_expr \
                + return_expr \
                + "\n}"

    def buildHpp(self):
        return 'extern "C" int parallel_new_{}(int object_num);'.format(self.class_name)

    def buildCdef(self):
        return 'int parallel_new_{}(int object_num);'.format(self.class_name) 

class DoAllBuilder():
    def __init__(self, class_node):
        self.class_node = class_node

    def buildCpp(self):
        fields_str = ""
        field_types_str = ""
        for i in range(len(self.class_node.declared_fields)):
            field = self.class_node.declared_fields[i]
            field_type = field.type
            if type(field_type) not in [cg.IntNode, cg.FloatNode, cg.BoolNode]:
                # fields_str += "(int) this->{}, ".format(field)                    
                fields_str += "0".format(field.name)
                field_types_str += "{}".format("int")
            else:
                fields_str += "this->{}".format(field.name)
                field_types_str += "{}".format(field_type.name)
            if i != len(self.class_node.declared_fields) - 1:
                fields_str += ", "
                field_types_str += ", "
        func_exprs = ['\n' +
                        'void {}::_do(void (*pf)({})){{\n'.format(self.class_node.name, field_types_str) +
                        INDENT +
                        'pf({});\n'.format(fields_str) +
                        '}',
                        '\n' +
                        'extern "C" int {}_do_all(void (*pf)({})){{\n'.format(self.class_node.name, field_types_str) +
                        INDENT +
                        'allocator_handle->template device_do<{}>(&{}::_do, pf);\n '.format(self.class_node.name,
                                                                                            self.class_node.name) +
                        INDENT + 'return 0;\n' +
                        '}']
        return "\n".join(func_exprs)
        
    def buildHpp(self):
        return 'extern "C" {}\n'.format(self.buildCdef())
    
    def buildCdef(self):
        field_types_str = ""
        for i in range(len(self.class_node.declared_fields)):
            field = self.class_node.declared_fields[i]
            field_type = field.type
            if type(field_type) not in [cg.IntNode, cg.FloatNode, cg.BoolNode]:
                field_type = cg.IntNode()
            if i != len(self.class_node.declared_fields) - 1:
                field_types_str += "{}, ".format(field_type.name)
            else:
                field_types_str += "{}".format(field_type.name)
        return 'int {}_do_all(void (*pf)({}));'.format(self.class_node.name, field_types_str)

class ParallelDoBuilder():

    def __init__(self, func_node):
        self.func_node = func_node
        self.args = []
        for arg in func_node.arguments:
            if arg.name != "self":
                self.args.append(arg)

    def buildCpp(self):
        arg_strs = []
        args = []
        for arg in self.args:
            args.append(arg.name)
            arg_strs.append("{} {}".format(arg.type.name, arg.name))
        parallel_do_expr = INDENT + "allocator_handle->parallel_do<{}, &{}::{}>({});\n".format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ", ".join(args)
        )
        return_expr = INDENT + "return 0;"
        return 'extern "C" int {}_{}_{}({}){{\n'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ", ".join(arg_strs)) \
                + parallel_do_expr \
                + return_expr \
                + "\n}"
    
    def buildHpp(self):
        arg_strs = []
        for arg in self.args:
            arg_strs.append("{} {}".format(arg.type.name, arg.name))
        return 'extern "C" int {}_{}_{}({});'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ",".join(arg_strs)
        )
    
    def buildCdef(self):
        arg_strs = []
        for arg in self.args:
            args_strs.append("{} {}".format(arg.type.name. arg.name))
        return 'int {}_{}_{}({});'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ",".join(arg_strs)
        )

class Preprocessor(ast.NodeVisitor):
    
    def __init__(self, root: cg.RootNode):
        self.stack = [root]
        self.ast_root = None
        self.classes = []
        self.cpp_parallel_new_codes = []
        self.hpp_parallel_new_codes = []
        self.cdef_parallel_new_codes = []
        self.cpp_do_all_codes = []
        self.hpp_do_all_codes = []
        self.cdef_do_all_codes = []
        self.parallel_do_hashtable = []
        self.cpp_parallel_do_codes = []
        self.hpp_parallel_do_codes = []
        self.cdef_parallel_do_codes = []
        self.global_device_variables = {}

    @property
    def root(self):
        return self.stack[0]

    def __gen_Hash(self, lst):
        """
        Helper function, generate same hash value for tuple with same strings
            Used to prevent generate mutiple code for a same parallel_do function
        """
        m = hashlib.md5()
        for elem in lst:
            m.update(elem.encode('utf-8'))
        return m.hexdigest()

    def build_parallel_do_cpp(self):
        return '\n\n' + '\n\n'.join(self.cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n'.join(self.hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.cdef_parallel_do_codes)

    def build_do_all_cpp(self):
        return '\n\n'.join(self.cpp_do_all_codes)

    def build_do_all_hpp(self):
        return '\n\n' + '\n'.join(self.hpp_do_all_codes)

    def build_do_all_cdef(self):
        return '\n' + '\n'.join(self.cdef_do_all_codes)

    def build_parallel_new_cpp(self):
        return '\n\n' + '\n\n'.join(self.cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.cdef_parallel_new_codes)

    def visit_Module(self, node):
        self.ast_root = node
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.stack[-1].get_ClassNode(class_name)
        self.stack.append(class_node)
        self.generic_visit(node)
        self.stack.pop()
    
    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.stack[-1].get_FunctionNode(func_name, self.stack[-1].name)
        self.stack.append(func_node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node):
        # Find device classes through device code
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name \
                and node.func.value.id == "DeviceAllocator":
            if node.func.attr == "device_class":
                for class_arg in node.args:
                    class_name = class_arg.id
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                        pnb = ParallelNewBuilder(class_name)
                        self.cpp_parallel_new_codes.append(pnb.buildCpp())
                        self.hpp_parallel_new_codes.append(pnb.buildHpp())
                        self.cdef_parallel_new_codes.append(pnb.buildCdef())
                        class_node = self.root.get_ClassNode(class_name)
                        dab = DoAllBuilder(class_node)
                        self.cpp_do_all_codes.append(dab.buildCpp())
                        self.hpp_do_all_codes.append(dab.buildHpp())
                        self.cdef_do_all_codes.append(dab.buildCdef())
            elif node.func.attr == "parallel_do":
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.parallel_do_hashtable:
                    self.parallel_do_hashtable.append(hval)
                    func_node = self.root.get_FunctionNode(node.args[1].attr, node.args[1].value.id)
                    pdb = ParallelDoBuilder(func_node)
                    self.cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.cdef_parallel_do_codes.append(pdb.buildCdef())
            elif node.func.attr == "array_size":
                self.global_device_variables[str(node.args[0].id)] = node.args[1].n
        # Find device classes through host code
        if type(node.func) is ast.Attribute \
                and type(node.func.value) is ast.Name \
                and (node.func.value.id == "allocator" or node.func.value.id == "PyAllocator"):
            if node.func.attr == "parallel_new":
                class_name = node.args[0].id
                if class_name not in self.classes:
                    self.classes.append(class_name)
                    pnb = ParallelNewBuilder(class_name)
                    self.cpp_parallel_new_codes.append(pnb.buildCpp())
                    self.hpp_parallel_new_codes.append(pnb.buildHpp())
                    self.cdef_parallel_new_codes.append(pnb.buildCdef())
                    class_node = self.root.get_ClassNode(class_name)
                    dab = DoAllBuilder(class_node)
                    self.__cpp_do_all_codes.append(dab.buildCpp())
                    self.__hpp_do_all_codes.append(dab.buildHpp())
                    self.__cdef_do_all_codes.append(dab.buildCdef())
            elif node.func.attr == "parallel_do":
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.parallel_do_hashtable:
                    self.parallel_do_hashtable.append(hval)
                    func_node = self.root.get_FunctionNode(node.args[1].attr, node.args[1].value.id)
                    pdb = ParallelDoBuilder(func_node)
                    self.cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.cdef_parallel_do_codes.append(pdb.buildCdef())
        self.generic_visit(node)

def compile(source_code, dir_path, file_name):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name

    # Generate python ast
    py_ast = ast.parse(source_code)

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
    cv = cpp.CppVisitor(pp.root)
    cv.visit(py_ast)
    hv = cpp.HppVisitor(pp.root)

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

    print(hpp_code)

    new_py_ast = astunparse.unparse(py_ast)

    return new_py_ast, None, None, None
    
