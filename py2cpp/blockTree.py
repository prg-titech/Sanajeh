# -*- coding: utf-8 -*-
# A tree which represents the calling and declaring relationships
from abc import abstractmethod
from typing import Set


# Block tree Node
class BlockTreeNode:

    node_count = 0

    def __init__(self, node_name, identifier_name):
        self.name: str = node_name  # node name
        self.declared_variables: Set[VariableTreeNode] = set()  # variables declared in this block
        self.called_functions: Set[FunctionTreeNode] = set()  # functions called in this block
        self.called_variables: Set[VariableTreeNode] = set()  # variables called in this block
        self.is_device = False  # if it is an __device__ node
        self.i_name: str = identifier_name  # identifier name
        self.id = BlockTreeNode.node_count  # node id
        BlockTreeNode.node_count += 1

    # Mark this node
    def MarkDeviceData(self):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            if nd.is_device:
                q.pop(0)
                continue
            nd.is_device = True
            # if type(nd) is ClassTreeNode:
            #     print("Class {}".format(nd.name))
            # if type(nd) is FunctionTreeNode:
            #     print("Function {}".format(nd.name))
            for x in nd.called_variables:
                x.MarkDeviceData()

            for x in nd.declared_variables:
                x.MarkDeviceData()

            for x in nd.called_functions:
                if x.is_device:
                    return
                q.append(x)
            q.pop(0)


# Tree node for class
class ClassTreeNode(BlockTreeNode):

    def __init__(self, node_name, identifier_name):
        super(ClassTreeNode, self).__init__(node_name, identifier_name)

        # functions declared in this class
        self.declared_functions: Set[FunctionTreeNode] = set()

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, identifier_name):
        # todo identifier_name maybe a class name
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x

    def GetVariableNode(self, variable_name, identifier_name, variable_type):
        # find the variable in the class block
        for x in self.declared_variables:
            if x.name == variable_name and x.i_name == identifier_name:
                if x.v_type == variable_type:
                    return x
                else:
                    print("Variable '{}' has type '{}', not '{}'".format(
                        variable_name, x.v_type, variable_type
                    ))
                    assert False

        return None

    def IsDeviceFunction(self, function_name, identifier_name):
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x.is_device
        return None


# Tree node for function
class FunctionTreeNode(BlockTreeNode):
    def __init__(self, node_name, identifier_name):
        super(FunctionTreeNode, self).__init__(node_name, identifier_name)

    def GetVariableNode(self, variable_name, identifier_name, variable_type):
        # find the variable in this function
        for x in self.declared_variables:
            if x.name == variable_name and x.i_name == identifier_name:
                if x.v_type == variable_type:
                    return x
                else:
                    print("Variable '{}' has type '{}', not '{}'".format(
                        variable_name, x.v_type, variable_type
                    ))
                    assert False

        return None


# Tree node for variable
class VariableTreeNode(BlockTreeNode):
    def __init__(self, node_name, identifier_name, var_type):
        super(VariableTreeNode, self).__init__(node_name, identifier_name)
        self.v_type: str = var_type  # type of the variable, "None" for untyped variables

    def MarkDeviceData(self):
        self.is_device = True
        # print("Variable {}".format(self.name))


# A tree which represents the calling and declaring relationships
class BlockTreeRoot(BlockTreeNode):
    declared_classes: Set[ClassTreeNode] = set()  # global classes
    declared_functions: Set[FunctionTreeNode] = set()  # global functions
    declared_variables: Set[VariableTreeNode] = set()  # global variables
    library_functions: Set[FunctionTreeNode] = set()  # library functions
    called_functions: Set[FunctionTreeNode] = set()  # functions called globally(shouldn't be device function)
    called_variables: Set[VariableTreeNode] = set()  # variables called globally

    def __init__(self, node_name, identifier_name):
        super(BlockTreeRoot, self).__init__(node_name, identifier_name)

    # Find the class 'class_name'
    def GetClassNode(self, class_name, identifier_name):
        for x in self.declared_classes:
            if x.name == class_name and x.i_name == identifier_name:
                return x
        return None

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, identifier_name):
        # find the function in the functions defined in the global block
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x
        for x in self.library_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x
        return None

    def GetVariableNode(self, variable_name, identifier_name, variable_type):
        # find the variable in the global block
        for x in self.declared_variables:
            if x.name == variable_name and x.i_name == identifier_name:
                if x.v_type == variable_type or variable_type is None:
                    return x
                else:
                    print("Variable '{}' has type '{}', not '{}'".format(
                        variable_name, x.v_type, variable_type
                    ))
                    assert False

        return None

    # Mark all functions that needs to be allocated in the allocator
    def MarkDeviceDataByClassName(self, class_names):
        for cln in class_names:
            # do not support just mark a child class which nest in another class
            for cls in self.declared_classes:
                if cls.name == cln:

                    # Mark all called functions in class cls
                    cls.MarkDeviceData()

                    # Mark all called functions in the functions of cls
                    for func in cls.declared_functions:
                        func.MarkDeviceData()

                    # Mark all variables in that class cls (in the functions of cls)
                    for var in cls.declared_variables:
                        var.MarkDeviceData()

    # Query whether the function is a device function
    def IsDeviceFunction(self, function_name, class_name, identifier_name):
        for x in self.declared_classes:
            if x.name == class_name:
                result = x.IsDeviceFunction(function_name, identifier_name)
                if result is not None:
                    return result
        for x in self.declared_functions:
            if x.name == function_name and x.i_name == identifier_name:
                return x.is_device
        print("Undeclared function '{}.{}'".format(class_name, function_name))
