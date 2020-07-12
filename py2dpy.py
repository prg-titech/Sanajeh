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

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        ret = super(PyDeviceCodeTransformer, self).visit(node)

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.__current_node.GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        if class_node.is_device:
            pass
            # todo add parallel_new, do_all.
            # todo how to deal with parallel_do?
        self.__classes.append(name)
        self.__node_path.append(class_node)


