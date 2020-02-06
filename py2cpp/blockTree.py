# -*- coding: utf-8 -*-
# Mark all device functions

from typing import List
from typing import Set


# Block tree Node
class BlockTreeNode:

    def __init__(self, node_type, function_name, class_name, identifier_name):
        # define the type of the node, 0:class, 1:function, 2:variable
        self.nd_type = node_type
        self.f_name: str = function_name  # function name
        self.c_name: str = class_name  # class name
        self.i_name: str = identifier_name  # identifier name

        # if the function is called in a class but not in any of the functions in that class,
        # the class name will be added here for marking purpose
        self.called_c_name = set()
        # classes declared in this block
        self.declared_classes: List[BlockTreeNode] = []
        # functions declared in this block
        self.declared_functions: List[BlockTreeNode] = []
        # variables declared in this block
        self.declared_variables: List[str] = []
        self.called_functions: Set[BlockTreeNode] = set()  # functions called in this block
        self.is_device = False  # if it is an __device__ block

    # Find the class 'class_name' by BFS
    def GetClassNode(self, class_name, identifier_name):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_classes:
                if x.c_name == class_name and x.i_name == identifier_name:
                    return x
                q.append(x)
            q.pop(0)
        return None

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, class_name, identifier_name):
        # find the function in the classes defined in this block
        for x in self.declared_classes:
            if x.c_name == class_name:
                func_node = x.GetFunctionNode(function_name, class_name, identifier_name)
                if func_node is not None:
                    return func_node
        # find the function in the functions defined in this block
        for x in self.declared_functions:
            if x.f_name == function_name and x.c_name == class_name and x.i_name == identifier_name:
                return x

    # Mark all the functions called by this function node to device function
    def RecursiveMark(self):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            self.is_device = True
            print('Marking, function: {}, class: {}, identifier: {}'.format(nd.f_name, nd.c_name, nd.i_name))
            for x in nd.called_functions:
                q.append(x)
            q.pop(0)
        return None

    # Query whether the function is a device function
    def IsDeviceFunction(self, function_name, class_name, identifier_name):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_functions:
                if x.f_name == function_name and x.c_name == class_name and x.i_name == identifier_name:
                    return x.is_device
                q.append(x)
            q.pop(0)
        return False


# A tree which represents the calling adn declaring relationships
class EnvironmentBlockTree:
    # global_classes
    global_classes: List[BlockTreeNode] = []
    # global_functions
    declared_functions: List[BlockTreeNode] = []
    # global variables
    declared_variables: List[str] = []
    called_functions: Set[BlockTreeNode] = set()  # functions called in this block
