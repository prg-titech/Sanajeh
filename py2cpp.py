# -*- coding: utf-8 -*-
# Mark all device functions

import ast
import copy
import hashlib
import sys

import astunparse

import type_converter
from config import INDENT, FILE_NAME, CDEF_FILE_PATH, CPP_FILE_PATH, HPP_FILE_PATH, PY_FILE_PATH
from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode
from gen_cppast import GenCppAstVisitor
import build_cpp


# Generate python call graph
class GenPyCallGraphVisitor(ast.NodeVisitor):

    def __init__(self):
        self.__root = CallGraph('root')
        self.__pp = Preprocessor(self.__root)
        self.__node_path = [self.__root]
        self.__current_node = None
        self.__variables = {}

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
        super(GenPyCallGraphVisitor, self).visit(node)

    # todo other py files
    # def visit_Module(self, node):
    #     self.generic_visit(node)

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        if type(self.__current_node) is not CallGraph:
            print("Doesn't support nested classes", file=sys.stderr)
            sys.exit(1)
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is not None:
            # Program shouldn't come to here, which means a class is defined twice
            print("The class {} is defined twice.".format(class_name), file=sys.stderr)
            sys.exit(1)
        class_node = ClassNode(node.name)
        self.__current_node.declared_classes.add(class_node)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name
        if type(self.__current_node) is not CallGraph and type(self.__current_node) is not ClassNode:
            print("Doesn't support nested functions", file=sys.stderr)
            sys.exit(1)
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is not None:
            # Program shouldn't come to here, which means a function is defined twice
            print("The function {} is defined twice.".format(func_name), file=sys.stderr)
            sys.exit(1)
        ret_type = None
        if type(self.__current_node) is ClassNode and func_name == "__init__":
            ret_type = self.__current_node.name
        elif hasattr(node.returns, "id"):
            ret_type = node.returns.id
        func_node = FunctionNode(func_name, self.__current_node.name, ret_type)
        self.__current_node.declared_functions.add(func_node)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Add arguments to the environment
    def visit_arguments(self, node):
        if type(self.__current_node) is not FunctionNode:
            print('Unexpected node "{}"'.format(self.__current_node.name), file=sys.stderr)
            sys.exit(1)
        for arg in node.args:
            annotation = None
            if arg.arg == "self":
                continue
            if hasattr(arg, "annotation"):
                annotation = arg.annotation.id
            var_node = VariableNode(arg.arg, annotation)
            self.__current_node.arguments.add(var_node)
            self.__variables.setdefault(self.__current_node.id, []).append(arg.arg)

    # Add global variables to the environment
    def visit_Global(self, node):
        for global_variable in node.names:
            self.__variables.setdefault(self.__current_node.id, []).append(global_variable)
            var_node = self.__root.GetVariableNode(global_variable, None)
            if var_node is None:
                print("The global variable {} is not existed.".format(global_variable), file=sys.stderr)
                sys.exit(1)
            self.__current_node.called_variables.add(var_node)

    # Create nodes for variables without type annotation
    def visit_Assign(self, node):
        for var in node.targets:
            var_name = None

            if type(var) is ast.Attribute:
                pass
                # var_name = var.attr
                # print(var_name, var.value.id)
                # # todo Attribute variables(self should refer to the class not in the current block),
                # # todo haven't thought about other occasions
                # if var.value.id == 'self':
                #     pass
            elif type(var) is ast.Name:
                var_name = var.id
                self.__variables.setdefault(self.__current_node.id, [])
                # print(self.__variables)
                if var_name not in self.__variables[self.__current_node.id]:
                    var_node = VariableNode(var_name, None)
                    self.__current_node.declared_variables.add(var_node)
                    self.__variables[self.__current_node.id].append(var_name)

        self.generic_visit(node)

    # Create nodes for variables with type annotation
    def visit_AnnAssign(self, node):
        var = node.target
        ann = None
        e_ann = None
        if type(node.annotation) is ast.Subscript:
            ann = node.annotation.value.id
            e_ann = node.annotation.slice.value.id
        else:
            ann = node.annotation.id

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
                var_node = VariableNode(var_name, ann, e_ann)
                self.__current_node.declared_variables.add(var_node)
                self.__variables[self.__current_node.id].append(var_name)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.__variables.setdefault(self.__current_node.id, [])
        if node.id in self.__variables[self.__current_node.id]:
            return
        for annotate_location_node in self.__node_path[-2::-1]:
            self.__variables.setdefault(annotate_location_node.id, [])
            if node.id in self.__variables[annotate_location_node.id]:
                var_node = annotate_location_node.GetVariableNode(node.id, None)
                if var_node is None:
                    print('Unexpected error, can not find variable "{}"', node.id, file=sys.stderr)
                    sys.exit(1)
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
        return '\n\n' + '\n\n'.join(self.__pp.cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n'.join(self.__pp.hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_do_codes)

    def build_do_all_cpp(self):
        return '\n\n'.join(self.__pp.cpp_do_all_codes)

    def build_do_all_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_do_all_codes)

    def build_do_all_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_do_all_codes)

    def build_parallel_new_cpp(self):
        return '\n\n' + '\n\n'.join(self.__pp.cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.__pp.hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.__pp.cdef_parallel_new_codes)

    def build_global_device_variables_init(self):
        ret = []
        for var in self.__pp.global_device_variables:
            var_node = self.__root.GetVariableNode(var, None)
            e_type = type_converter.convert(var_node.e_type)
            n = self.__pp.global_device_variables[var]
            # Cell ** host_cells;
            # cudaMalloc( & host_cells, sizeof(Cell *) * dataset.x * dataset.y);
            # cudaMemcpyToSymbol(cells, & host_cells, sizeof(Cell **), 0,
            # cudaMemcpyHostToDevice);
            ret.append(INDENT + "{}* host_{};\n".format(e_type, var) + \
               INDENT + "cudaMalloc(&host_{}, sizeof({})*{});\n".format(var, e_type, n) + \
               INDENT + "cudaMemcpyToSymbol({}, &host_{}, sizeof({}*), 0, cudaMemcpyHostToDevice);\n"\
                   .format(var, var, e_type))
        return "\n".join(ret)

    def build_global_device_variables_unit(self):
        ret = []
        for var in self.__pp.global_device_variables:
            ret.append(INDENT + "cudaFree(host_{});\n".format(var))
        return "\n".join(ret)


# Find device class in python code and compile parallel_do expressions into c++ ones
class Preprocessor(ast.NodeVisitor):

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

    @property
    def cpp_do_all_codes(self):
        return self.__cpp_do_all_codes

    @property
    def hpp_do_all_codes(self):
        return self.__hpp_do_all_codes

    @property
    def cdef_do_all_codes(self):
        return self.__cdef_do_all_codes

    # Build codes for parallel_new in c++
    class ParallelNewBuilder:
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
    class ParallelDoBuilder(ast.NodeVisitor):
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
            super(Preprocessor.ParallelDoBuilder, self).visit(node)

        def visit_ClassDef(self, node):
            if node.name != self.__func_class_name:
                return
            class_name = node.name
            class_node = self.__current_node.GetClassNode(class_name)
            if class_node is None:
                # Program shouldn't come to here, which means the class does not exist
                print("The class {} is not exist.".format(class_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(class_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_FunctionDef(self, node):
            func_name = node.name
            func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
            if func_node is None:
                # Program shouldn't come to here, which means the function does not exist
                print("The function {} does not exist.".format(func_name), file=sys.stderr)
                sys.exit(1)
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

    # Collect information of class fields and build codes do_all functions in c++
    class DoAllBuilder(ast.NodeVisitor):
        def __init__(self, rt, class_name):
            self.__root = rt
            self.__node_path = [rt]
            self.__current_node = None
            self.__class_name = class_name
            self.__field = {}

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            super(Preprocessor.DoAllBuilder, self).visit(node)

        def visit_ClassDef(self, node):
            if node.name != self.__class_name:
                return
            class_name = node.name
            class_node = self.__current_node.GetClassNode(class_name)
            if class_node is None:
                # Program shouldn't come to here, which means the class does not exist
                print("The class {} is not exist.".format(class_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(class_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_FunctionDef(self, node):
            func_name = node.name
            func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
            if func_node is None:
                # Program shouldn't come to here, which means the function does not exist
                print("The function {} does not exist.".format(func_name), file=sys.stderr)
                sys.exit(1)
            self.__node_path.append(func_node)
            self.generic_visit(node)
            self.__node_path.pop()

        def visit_AnnAssign(self, node):
            if type(self.__current_node) is ClassNode:
                var = node.target
                anno = node.annotation
                self.__field[var.id] = type_converter.convert(anno.id)

        def buildCpp(self):
            fields_str = ""
            field_types_str = ""
            for i, field in enumerate(self.__field):
                if i != len(self.__field) - 1:
                    fields_str += "this->{}, ".format(field)
                    field_types_str += "{}, ".format(self.__field[field])
                else:
                    fields_str += "this->{}".format(field)
                    field_types_str += "{}".format(self.__field[field])
            func_exprs = ['\n' +
                          'void {}::_do(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                          INDENT +
                          'pf({});\n'.format(fields_str) +
                          '}',
                          '\n' +
                          'extern "C" int {}_do_all(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                          INDENT +
                          'allocator_handle->template device_do<{}>(&{}::_do, pf);\n '.format(self.__class_name,
                                                                                              self.__class_name) +
                          INDENT + 'return 0;\n' +
                          '}']
            return "\n".join(func_exprs)

        def buildHpp(self):
            return 'extern "C" {}\n'.format(self.buildCdef())

        def buildCdef(self):
            field_types_str = ""
            for i, field in enumerate(self.__field):
                if i != len(self.__field) - 1:
                    field_types_str += "{}, ".format(self.__field[field])
                else:
                    field_types_str += "{}".format(self.__field[field])
            return 'int {}_do_all(void (*pf)({}));'.format(self.__class_name, field_types_str)

    def __init__(self, rt: CallGraph):
        self.__classes = []

        self.has_device_data = False
        self.__is_root = True  # the flag of whether visiting the root node of python ast
        self.__node_root = None  # the root node of python ast
        self.__cpp_parallel_do_codes = []
        self.__hpp_parallel_do_codes = []
        self.__cdef_parallel_do_codes = []
        self.__cpp_parallel_new_codes = []
        self.__hpp_parallel_new_codes = []
        self.__cdef_parallel_new_codes = []
        self.__cpp_do_all_codes = []
        self.__hpp_do_all_codes = []
        self.__cdef_do_all_codes = []
        self.__parallel_do_hashtable = []
        self.__root = rt
        self.__node_path = [rt]
        self.__current_node = None
        self.global_device_variables = {}

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        super(Preprocessor, self).visit(node)

    @property
    def classes(self):
        return self.__classes

    def __gen_Hash(self, lst):
        """
        Helper function, generate same hash value for tuple with same strings
            Used to prevent generate mutiple code for a same parallel_do function
        """
        m = hashlib.md5()
        for elem in lst:
            m.update(elem.encode('utf-8'))
        return m.hexdigest()

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} is not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is None:
            # Program shouldn't come to here, which means the function does not exist
            print("The function {} does not exist.".format(func_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Analyze function calling relationships
    def visit_Call(self, node):
        # Find device classes through device code
        if type(node.func) is ast.Attribute and \
                hasattr(node.func.value, "id") and \
                node.func.value.id == "DeviceAllocator":
            if node.func.attr == 'device_class':
                self.has_device_data = True
                for cls in node.args:
                    if cls.id not in self.__classes:
                        self.__classes.append(cls.id)
                        pnb = self.ParallelNewBuilder(cls.id)
                        self.__cpp_parallel_new_codes.append(pnb.buildCpp())
                        self.__hpp_parallel_new_codes.append(pnb.buildHpp())
                        self.__cdef_parallel_new_codes.append(pnb.buildCdef())
                        dab = self.DoAllBuilder(self.__root, node.args[0].id)
                        dab.visit(self.__node_root)
                        self.__cpp_do_all_codes.append(dab.buildCpp())
                        self.__hpp_do_all_codes.append(dab.buildHpp())
                        self.__cdef_do_all_codes.append(dab.buildCdef())
            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    pdb = self.ParallelDoBuilder(self.__root,
                                                 node.args[0].id,
                                                 node.args[1].value.id,
                                                 node.args[1].attr)
                    pdb.visit(self.__node_root)
                    self.__cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.__hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.__cdef_parallel_do_codes.append(pdb.buildCdef())
            elif node.func.attr == 'array_size':
                self.global_device_variables[str(node.args[0].id)] = node.args[1].n

        # Find device classes through host code
        if type(node.func) is ast.Attribute \
                and hasattr(node.func.value, "id") and \
                node.func.value.id == "PyAllocator":
            if node.func.attr == 'parallel_new':
                self.has_device_data = True
                if node.args[0].id not in self.__classes:
                    self.__classes.append(node.args[0].id)
                    pnb = self.ParallelNewBuilder(node.args[0].id)
                    self.__cpp_parallel_new_codes.append(pnb.buildCpp())
                    self.__hpp_parallel_new_codes.append(pnb.buildHpp())
                    self.__cdef_parallel_new_codes.append(pnb.buildCdef())
                    dab = self.DoAllBuilder(self.__root, node.args[0].id)
                    dab.visit(self.__node_root)
                    self.__cpp_do_all_codes.append(dab.buildCpp())
                    self.__hpp_do_all_codes.append(dab.buildHpp())
                    self.__cdef_do_all_codes.append(dab.buildCdef())

            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    pdb = self.ParallelDoBuilder(self.__root,
                                                 node.args[0].id,
                                                 node.args[1].value.id,
                                                 node.args[1].attr)
                    pdb.visit(self.__node_root)
                    self.__cpp_parallel_do_codes.append(pdb.buildCpp())
                    self.__hpp_parallel_do_codes.append(pdb.buildHpp())
                    self.__cdef_parallel_do_codes.append(pdb.buildCdef())

        func_name = None
        call_node = None
        var_type = None

        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            if type(node.func.value) is ast.Attribute:
                if hasattr(node.func.value.value, "id") \
                        and node.func.value.value.id == "self" \
                        and type(self.__node_path[-2]) is ClassNode:
                    for var in self.__node_path[-2].declared_variables:
                        if var.name == node.func.value.attr:
                            var_type = var.v_type
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        for parent_node in self.__node_path[::-1]:
            if type(parent_node) is FunctionNode:
                continue
            call_node = parent_node.GetFunctionNode(func_name, var_type)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionNode(func_name, var_type, None)
            self.__root.library_functions.add(call_node)
        self.__current_node.called_functions.add(call_node)
        self.generic_visit(node)

class DeviceCodeVisitor(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.root: CallGraph = root
        self.node_path = [self.root]

    def visit_Module(self, node):
        for x in node.body:
            if type(x) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            self.visit(node.args)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node


class Searcher(DeviceCodeVisitor):
    """Find all classes that are used for fields or variables types in device codes"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.dev_cls = set()  # device classes
        self.sdef_cls = set()  # classes that are used for fields or variables types

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.dev_cls.add(name)
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in ("str", "float", "int"):
                self.sdef_cls.add(ann)
        return node

    def visit_arg(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in ("str", "float", "int", None):
                self.sdef_cls.add(ann)
        return node


class Normalizer(DeviceCodeVisitor):
    """Use variables to rewrite nested expressions"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.v_counter = 0  # used to count the auto generated variables
        self.has_auto_variables = False
        self.last_annotation = None

    def visit_FunctionDef(self, node):
        self.v_counter = 0  # counter needs to be reset in every function
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Expr(self, node):
        ret = []
        self.has_auto_variables = False
        self.last_annotation = "None"
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.Expr(value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        self.has_auto_variables = False
        self.last_annotation = "None"
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.Assign(targets=node.targets, value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        ret = []
        self.has_auto_variables = False
        self.last_annotation = "None"
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Call(self, node):
        ret = []
        # if self.has_auto_variables:
        if hasattr(node.func, "value") and type(node.func.value) == ast.Name and self.has_auto_variables:
            if type(self.node_path[-1]) == FunctionNode:
                var_type = self.node_path[-1].GetVariableType(node.func.value.id)
                if var_type is None:
                    class_node = self.node_path[0].GetClassNode(self.last_annotation)
                    for x in class_node.declared_functions:
                        if x.name == node.func.attr:
                            self.last_annotation = x.ret_type
                            break
                else:
                    for x in self.node_path[-1].called_functions:
                        if x.name == node.func.attr and x.c_name == var_type:
                            self.last_annotation = x.ret_type
                            break

        elif hasattr(node.func, "value") and type(node.func.value) == ast.Attribute and self.has_auto_variables:
            if node.func.value.value.id == "self":
                var_type = None
                for x in self.node_path[-2].declared_variables:
                    if x.name == node.func.value.attr:
                        var_type = x.v_type
                        break
                for x in self.node_path[-1].called_functions:
                    if x.name == node.func.attr and x.c_name == var_type:
                        self.last_annotation = x.ret_type
                        break
            else:
                var_type = None
                caller_type = self.node_path[-1].GetVariableType(node.func.value.value.id)
                var_class = self.node_path[0].GetClassNode(caller_type)
                for x in var_class.declared_variables:
                    if x.name == node.func.value.attr:
                        var_type = x.v_type
                        break
                caller_class = self.node_path[0].GetClassNode(var_type)
                for x in caller_class.declared_functions:
                    if x.name == node.func.attr:
                        self.last_annotation = x.ret_type
                        break

        # self.f.A().B()...
        if hasattr(node.func, "value") and type(node.func.value) == ast.Call:
            self.has_auto_variables = True
            assign_nodes = self.visit(node.func.value)
            new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
            self.visit(assign_nodes[-1])
            ret.extend(assign_nodes[:-1])
            if type(self.node_path[-2]) == ClassNode:
                pass
            else:
                print("Invalid data structure.", file=sys.stderr)
                sys.exit(1)
            ret.append(ast.AnnAssign(target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store()),
                                     value=assign_nodes[-1],
                                     simple=1,
                                     annotation=ast.Name(id=self.last_annotation, ctx=ast.Load())
                                     ))
            self.v_counter += 1
            node.func.value = new_var_node

        for x in range(len(node.args)):
            # self.f.A(B())
            if type(node.args[x]) == ast.Call:
                self.has_auto_variables = True
                assign_nodes = self.visit(node.args[x])
                new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
                ret.extend(assign_nodes[:-1])
                self.visit(assign_nodes[-1])
                ret.append(ast.AnnAssign(target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store()),
                                         value=assign_nodes[-1],
                                         simple=1,
                                         annotation=ast.Name(id=self.last_annotation, ctx=ast.Load())
                                         ))
                self.v_counter += 1
                node.args[x] = new_var_node
        ret.append(node)
        return ret


class FunctionBodyGenerator(ast.NodeTransformer):
    """Gernerate new ast nodes which are used by the Inliner to inline functions"""

    def __init__(self, node, caller_type):
        self.func_name = None
        self.caller = None
        self.args = []
        self.func_args = []
        self.new_ast_nodes = []
        self.node = copy.deepcopy(node)
        self.caller_type = caller_type

    def visit_Module(self, node):
        for x in node.body:
            if type(x) == ast.ClassDef:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        if node.name != self.caller_type:
            return node
        else:
            node.body = [self.visit(x) for x in node.body]
        return node

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            #  check if argument number is correct
            if len(node.args.args) - 1 != len(self.args):
                print("Invalid argument number.", file=sys.stderr)
                sys.exit(1)
            for arg in node.args.args:
                if arg.arg != "self":
                    self.func_args.append(arg.arg)
            for x in node.body:
                self.new_ast_nodes.append(self.visit(x))
        return node

    def visit_Name(self, node):
        if node.id == "self":
            return self.caller
        for i in range(len(self.func_args)):
            if node.id == self.func_args[i]:
                return self.args[i]
        return node

    def GetTransformedNodes(self, caller, func_name, args):
        """Traverse the ast node given, find the target function and return the transformed implementation"""
        self.caller = copy.deepcopy(caller)
        self.func_name = func_name
        self.args = copy.deepcopy(args)
        self.visit(self.node)


class Inliner(DeviceCodeVisitor):
    """Replace function callings with specific implementation"""

    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Expr(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    continue
                ret.append(v)
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.Assign(targets=node.targets, value=v.value)
                ret.append(v)
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v.value)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Call(self, node):
        if type(node.func) == ast.Attribute:
            if type(node.func.value) == ast.Attribute:
                if type(node.func.value.value) == ast.Name:
                    if node.func.value.value.id == "self" and type(self.node_path[-2]) is ClassNode:
                        caller_type = None
                        for x in self.node_path[-2].declared_variables:
                            if x.name == node.func.value.attr:
                                caller_type = x.v_type
                        if caller_type not in self.sdef_cls:
                            return node
                        func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                        func_body_gen.GetTransformedNodes(node.func.value,
                                                          node.func.attr,
                                                          node.args)
                        if len(func_body_gen.new_ast_nodes) != 0:
                            return func_body_gen.new_ast_nodes
                    else:
                        var_type = None
                        caller_type = self.node_path[-1].GetVariableType(node.func.value.value.id)
                        var_class = self.node_path[0].GetClassNode(caller_type)
                        for x in var_class.declared_variables:
                            if x.name == node.func.value.attr:
                                var_type = x.v_type
                                break
                        if var_type not in self.sdef_cls:
                            return node
                        func_body_gen = FunctionBodyGenerator(self.node, var_type)
                        func_body_gen.GetTransformedNodes(node.func.value,
                                                          node.func.attr,
                                                          node.args)
                        if len(func_body_gen.new_ast_nodes) != 0:
                            return func_body_gen.new_ast_nodes
            elif type(node.func.value) == ast.Name:
                caller_type = self.node_path[-1].GetVariableType(node.func.value.id)
                if caller_type not in self.sdef_cls:
                    return node
                func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                func_body_gen.GetTransformedNodes(node.func.value,
                                                  node.func.attr,
                                                  node.args)
                if len(func_body_gen.new_ast_nodes) != 0:
                    return func_body_gen.new_ast_nodes
        return node


class Eliminator(DeviceCodeVisitor):
    """Inlining constructor functions"""

    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls
        self.var_dict = {}

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.var_dict = {}
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                elif node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if hasattr(node.annotation, "id") and node.annotation.id in self.sdef_cls and node.value is not None:
            if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                if node.value.func.id in self.sdef_cls:
                    func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                    func_body_gen.GetTransformedNodes(node.target,
                                                      "__init__",
                                                      node.value.args)
                    if len(func_body_gen.new_ast_nodes) != 0:
                        return func_body_gen.new_ast_nodes
            else:
                if type(node.value) is ast.Name:
                    self.var_dict[node.target.id] = self.var_dict[node.value.id]
                else:
                    self.var_dict[node.target.id] = node.value
                return None
        return node

    def visit_Assign(self, node):
        ret = []
        self.generic_visit(node)
        for field in node.targets:
            if type(field) == ast.Attribute and type(self.node_path[-2]) == ClassNode:
                var_type = None
                for x in self.node_path[-2].declared_variables:
                    if x.name == field.attr:
                        var_type = x.v_type
                        break
                if node.value is not None and var_type in self.sdef_cls:
                    if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                        if node.value.func.id in self.sdef_cls:
                            func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                            func_body_gen.GetTransformedNodes(field,
                                                              "__init__",
                                                              node.value.args)
                            if len(func_body_gen.new_ast_nodes) != 0:
                                ret.extend(func_body_gen.new_ast_nodes)
        if len(ret) != 0:
            return ret
        return node

    def visit_Name(self, node):
        if node.id in self.var_dict:
            return self.var_dict[node.id]
        return node


class FieldSynthesizer(DeviceCodeVisitor):
    """Synthesize new fields from the references to the nested objects"""

    def __init__(self, root: CallGraph, sdef_cls):
        super().__init__(root)
        self.sdef_cls = sdef_cls
        self.field_dict = {}

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.field_dict = {}
            self.node_path.append(class_node)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            i = 0
            for field in self.field_dict:
                node.body.insert(i, ast.AnnAssign(target=ast.Name(id=field, ctx=ast.Store()),
                                                  annotation=ast.Name(id=self.field_dict[field], ctx=ast.Load()),
                                                  simple=1,
                                                  value=None
                                                  )
                                 )
                i += 1
            self.node_path.pop()
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                elif node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Attribute(self, node):
        ctx = None
        if type(node.ctx) == ast.Load:
            ctx = ast.Load()
        elif type(node.ctx) == ast.Store:
            ctx = ast.Store()
        if type(node.value) == ast.Name:
            if self.node_path[-1].GetVariableType(node.value.id) in self.sdef_cls:
                return ast.Name(id=node.value.id + "_" + node.attr, ctx=ctx)
        elif type(node.value) == ast.Attribute and type(self.node_path[-2]) == ClassNode:
            var_type = None
            for x in self.node_path[-2].declared_variables:
                if x.name == node.value.attr:
                    var_type = x.v_type
                    break
            if var_type in self.sdef_cls:
                return ast.Attribute(attr=node.value.attr + "_" + node.attr, ctx=ctx, value=node.value.value)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if hasattr(node.annotation, "id") and node.annotation.id in self.sdef_cls:
            if node.value is None:
                return None
        if type(node.target) == ast.Attribute and node.target.value.id == "self":
            self.field_dict[node.target.attr] = node.annotation.id
            return ast.Assign(targets=[node.target], value=node.value)
        node.simple = 1
        return node


def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    gpcgv1 = GenPyCallGraphVisitor()
    gpcgv1.visit(node)
    if not gpcgv1.mark_device_data(node):
        print("No device data found")
        sys.exit(1)
    call_graph = gpcgv1.root
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_cls:
            scr.sdef_cls.remove(cls)
    ast.fix_missing_locations(Inliner(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(Eliminator(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(FieldSynthesizer(call_graph, scr.sdef_cls).visit(node))


def compile(source_code):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Generate python ast
    py_ast = ast.parse(source_code)
    # Generate python call graph
    gpcgv = GenPyCallGraphVisitor()
    gpcgv.visit(py_ast)
    # Mark all device data on the call graph
    if not gpcgv.mark_device_data(py_ast):
        print("No device data found")
        sys.exit(1)
    # Transform nested-objects to basic fields
    transform(py_ast, gpcgv.root)
    new_py_code = astunparse.unparse(py_ast)
    new_py_ast = ast.parse(new_py_code)
    gpcgv1 = GenPyCallGraphVisitor()
    gpcgv1.visit(new_py_ast)
    # Mark all device data on the call graph
    if not gpcgv1.mark_device_data(new_py_ast):
        print("No device data found")
        sys.exit(1)
    # Generate cpp ast from python ast
    gcv = GenCppAstVisitor(gpcgv1.root)
    cpp_node = gcv.visit(new_py_ast)
    # Generate cpp(hpp) code from cpp ast
    ctx = build_cpp.BuildContext.create()
    # Expression needed for DynaSOAr API
    cpp_include_expr = '#include "{}.h"\n\n'.format(FILE_NAME)
    allocator_declaration = "AllocatorHandle<AllocatorT>* allocator_handle;\n" \
                            "__device__ AllocatorT* device_allocator;\n"
    precompile_expr = "#ifndef SANAJEH_DEVICE_CODE_H" \
                      "\n#define SANAJEH_DEVICE_CODE_H" \
                      "\n#define KNUMOBJECTS 64*64*64*64"
    hpp_include_expr = '\n\n#include <curand_kernel.h>\n#include "dynasoar.h"'
    b = gpcgv1.build_global_device_variables_unit()
    init_cpp = ['\n\nextern "C" int AllocatorInitialize(){\n',
                INDENT +
                "allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);\n",
                INDENT +
                "AllocatorT* dev_ptr = allocator_handle->device_pointer();\n",
                INDENT +
                "cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);\n",
                gpcgv1.build_global_device_variables_init(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    init_hpp = '\nextern "C" int AllocatorInitialize();\n'
    init_cdef = '\nint AllocatorInitialize();'
    unit_cpp = ['\n\nextern "C" int AllocatorUninitialize(){\n',
                gpcgv1.build_global_device_variables_unit(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    unit_hpp = '\nextern "C" int AllocatorUninitialize();\n'
    unit_cdef = '\nint AllocatorUninitialize();'
    endif_expr = "\n#endif"

    # Source code
    cpp_code = cpp_include_expr \
               + allocator_declaration \
               + cpp_node.buildCpp(ctx) \
               + gpcgv1.build_do_all_cpp() \
               + gpcgv1.build_parallel_do_cpp() \
               + gpcgv1.build_parallel_new_cpp() \
               + "".join(init_cpp) \
               + "".join(unit_cpp)
    # Header code
    hpp_code = precompile_expr \
               + hpp_include_expr \
               + cpp_node.buildHpp(ctx) \
               + gpcgv1.build_do_all_hpp() \
               + gpcgv1.build_parallel_do_hpp() \
               + gpcgv1.build_parallel_new_hpp() \
               + init_hpp \
               + unit_hpp \
               + endif_expr

    # Codes for cffi cdef() function
    cdef_code = gpcgv1.build_parallel_do_cdef() \
                + gpcgv1.build_parallel_new_cdef() \
                + init_cdef + gpcgv1.build_do_all_cdef() \
                + unit_cdef
    with open(CPP_FILE_PATH, mode='w') as cpp_file:
        cpp_file.write(cpp_code)
    with open(HPP_FILE_PATH, mode='w') as hpp_file:
        hpp_file.write(hpp_code)
    with open(CDEF_FILE_PATH, mode='w') as cdef_file:
        cdef_file.write(cdef_code)
    with open(PY_FILE_PATH, mode='w') as py_file:
        py_file.write(new_py_code)

    return cpp_code, hpp_code, cdef_code
