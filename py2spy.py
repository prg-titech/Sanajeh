# -*- coding: utf-8 -*-
# Compile python code to device python code
import ast
import sys

from call_graph import CallGraph


class PyDeviceCodeTransformer(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.__root: CallGraph = root
        self.__node_path = [self.__root]
        self.__current_node = None
        self.__is_root = True  # the flag of whether visiting the root node of python ast
        self.__node_root = None  # the root node of python ast
        # Extra code to add in python code
        self.code = """
@staticmethod
def parallel_new(object_num, lib):
    lib.parallel_new(object_num)
    
@staticmethod
def do_all(lib, func):
    lib.do_all(func)
"""

    class ClassNodeTransformer(ast.NodeTransformer):

        def __init__(self, class_name):
            self.class_name = class_name

        def visit_FunctionDef(self, node):
            if node.name == "parallel_new":
                node.name = "parallel_new_{}".format(self.class_name)
            elif node.name == "do_all":
                node.name = "{}_do_all".format(self.class_name)
            self.generic_visit(node)
            return node

        def visit_Attribute(self, node):
            if node.attr == "parallel_new":
                node.attr = "parallel_new_{}".format(self.class_name)
            elif node.attr == "do_all":
                node.attr = "{}_do_all".format(self.class_name)
            self.generic_visit(node)
            return node

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        return super(PyDeviceCodeTransformer, self).visit(node)

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.__current_node.GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        if class_node.is_device:
            extra_ast = ast.parse(self.code)
            cnt = self.ClassNodeTransformer(class_node.name)
            cnt.visit(extra_ast)
            # ast.fix_missing_locations(tree)
            for ast_node in extra_ast.body:
                node.body.append(ast_node)
        self.__node_path.append(class_node)
        return node
