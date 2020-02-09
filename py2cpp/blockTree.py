# -*- coding: utf-8 -*-
# Mark all device functions

from typing import Set


# Block tree Node
class BlockTreeNode:

    def __init__(self, node_name, identifier_name):
        self.name: str = node_name  # node name
        self.i_name: str = identifier_name  # identifier name
        self.declared_variables: Set[VariableTreeNode] = set()  # variables declared in this block
        self.called_functions: Set[FunctionTreeNode] = set()  # functions called in this block
        self.is_device = False  # if it is an __device__ node

    # Mark this node and all functions called in this block to device function
    def RecursiveMarkByCall(self):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            self.is_device = True
            print('Marking {}, identifier: {}'.format(nd.name, nd.i_name))
            for x in nd.called_functions:
                q.append(x)
            q.pop(0)
        return None


# Tree node for class
class ClassTreeNode(BlockTreeNode):

    def __init__(self, node_name, identifier_name):
        super(ClassTreeNode, self).__init__(node_name, identifier_name)

        # classes declared in this class
        self.declared_classes: Set[ClassTreeNode] = set()
        # functions declared in this class
        self.declared_functions: Set[FunctionTreeNode] = set()

    # Find the class 'class_name' by BFS
    def GetClassNode(self, class_name, identifier_name):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_classes:
                if x.name == class_name and x.i_name == identifier_name:
                    return x
                q.append(x)
            q.pop(0)
        return None

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, class_name, identifier_name):
        # find the function in the classes defined in this block
        # todo: when class_name is None, identifier_name maybe a class name
        for x in self.declared_classes:
            if x.name == class_name:
                func_node = x.GetFunctionNode(function_name, class_name, identifier_name)
                if func_node is not None:
                    return func_node
        # find the function in the functions defined in this block
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x

    def IsDeviceFunction(self, function_name, class_name, identifier_name):
        for x in self.declared_classes:
            print(" '{}.{} {}'".format(class_name, function_name, x.name))
            if x.name == class_name:
                result = x.IsDeviceFunction(function_name, class_name, identifier_name)
                if result is not None:  
                    return result
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x.is_device
        return None


# Tree node for function
class FunctionTreeNode(BlockTreeNode):
    pass


# Tree node for variable
class VariableTreeNode(BlockTreeNode):
    def __init__(self, node_name, identifier_name):
        super(VariableTreeNode, self).__init__(node_name, identifier_name)
        self.v_type: str = ""
    # todo
    pass


# A tree which represents the calling adn declaring relationships
class BlockTree:
    declared_classes: Set[ClassTreeNode] = set()    # global classes
    declared_functions: Set[FunctionTreeNode] = set()    # global functions
    declared_variables: Set[VariableTreeNode] = set()    # global variables
    called_functions: Set[FunctionTreeNode] = set()  # functions called globally(shouldn't be device function)

    # Find the class 'class_name'
    def GetClassNode(self, class_name, identifier_name):
        for x in self.declared_classes:
            if x.name == class_name and x.i_name == identifier_name:
                return x
            # nested class
            cln = x.GetClassNode(class_name, identifier_name)
            if cln is not None:
                return cln
        return None

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, class_name, identifier_name):
        # find the function in the classes defined in the global block
        for x in self.declared_classes:
            if x.name == class_name:
                func_node = x.GetFunctionNode(function_name, class_name, identifier_name)
                if func_node is not None:
                    return func_node
        # find the function in the functions defined in the global block
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x
        return None

    # Mark all functions that needs to be allocated in the allocator
    def MarkFunctionsByClassName(self, class_names):
        for cln in class_names:
            # do not support just mark a child class which nest in another class
            for cls in self.declared_classes:
                if cls.name == cln:
                    # Mark all called functions in that class cls (not in the functions of cls)
                    cls.RecursiveMarkByCall()
                    # Mark all called functions in that class cls (in the functions of cls)
                    for func in cls.declared_functions:
                        func.RecursiveMarkByCall()

    # Query whether the function is a device function
    def IsDeviceFunction(self, function_name, class_name, identifier_name):
        for x in self.declared_classes:
            if x.name == class_name:
                result = x.IsDeviceFunction(function_name, class_name, identifier_name)
                if result is not None:
                    return result
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x.is_device
        print("Undeclared function '{}.{}'".format(class_name, function_name))
