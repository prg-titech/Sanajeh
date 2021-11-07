# -*- coding: utf-8 -*-
# Define Python call graph nodes' data structure

from typing import Set


# Block tree Node
class CallGraphNode:
    node_count = 0

    def __init__(self, node_name):
        self.name: str = node_name  # node name
        self.declared_variables: Set[VariableNode] = set()  # variables declared in this block
        self.called_functions: Set[FunctionNode] = set()  # functions called in this block
        self.called_variables: Set[VariableNode] = set()  # variables called in this block
        self.is_device = False  # if it is an __device__ node
        self.id = CallGraphNode.node_count  # node id
        CallGraphNode.node_count += 1

    # Mark this node
    def MarkDeviceData(self):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            if nd.is_device:
                q.pop(0)
                continue
            nd.is_device = True
            # if type(nd) is ClassNode:
            #     print("Class {}".format(nd.name))
            # if type(nd) is FunctionNode:
            #     print("Function {}".format(nd.name))
            for var_node in nd.called_variables:
                var_node.MarkDeviceData()

            for var_node in nd.declared_variables:
                var_node.MarkDeviceData()

            for func_node in nd.called_functions:
                if func_node.is_device:
                    return
                q.append(func_node)
            q.pop(0)

    def GetVariableNode(self, variable_name, variable_type):
        # find the variable in the global block
        for var_node in self.declared_variables:
            if var_node.name == variable_name:
                if var_node.v_type == variable_type or variable_type is None:
                    return var_node
                else:
                    print("Variable '{}' has type '{}', not '{}'".format(
                        variable_name, var_node.v_type, variable_type
                    ))
                    assert False

        return None

class TypeNode(CallGraphNode):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.declared_functions: Set[FunctionNode] = set()
        self.declared_fields: Set[VariableNode] = set()
        self.super_class: str = ""

    def GetFieldNode(self, field_name):
        for field in self.declared_fields:
            if field.name == field_name:
                return field
        return None

    # Find the function 'function_name' by recursion
    def GetFunctionNode(self, function_name, class_name):
        if class_name is None:
            for func_node in self.declared_functions:
                if func_node.name == function_name:
                    return func_node
        for func_node in self.declared_functions:
            if func_node.name == function_name and func_node.c_name == class_name:
                return func_node
        return None

class ListTypeNode(TypeNode):

    def __init__(self, type_node):
        super().__init__(type_node.name)
        self.type_node = type_node

class RefTypeNode(TypeNode):

    def __init__(self, type_node):
        super().__init__(type_node.name)
        self.type_node = type_node

# Tree node for class
class ClassNode(TypeNode):

    def __init__(self, node_name, super_class):
        super().__init__(node_name)
        # functions declared in this class
        # self.declared_functions: Set[FunctionNode] = set()
        # self.declared_fields: Set[VariableNode] = set()
        self.expanded_fields: dict = {}
        self.super_class: str = super_class
        self.has_random_state: bool = False

    def IsDeviceFunction(self, function_name, class_name):
        for func_node in self.declared_functions:
            if func_node.name == function_name and func_node.c_name == class_name:
                return func_node.is_device
        return None

# Tree node for function
class FunctionNode(CallGraphNode):
    def __init__(self, node_name, class_name, return_type):
        super().__init__(node_name)
        # arguments of the function
        self.arguments: Set[VariableNode] = set()
        self.ret_type = return_type
        self.c_name = class_name
        self.declared_functions: Set[FunctionNode] = set()

    def GetFunctionNode(self, function_name, class_name):
        if class_name is None:
            for function_node in self.declared_functions:
                if function_node.name == function_name:
                    return function_node
        for function_node in self.declared_functions:
            if function_node.name == function_name and function_node.c_name == class_name:
                return function_node
        return None

    def GetVariableType(self, variable_name):
        for var_node in self.declared_variables:
            if var_node.name == variable_name:
                return var_node.v_type
        # find the variable in the arguments:
        else:
            for arg in self.arguments:
                if arg.name == variable_name:
                    return arg.v_type

# Tree node for variable
class VariableNode(CallGraphNode):
    def __init__(self, node_name, var_type, element_type=None):
        super().__init__(node_name)
        self.v_type: str = var_type  # type of the variable, "None" for untyped variables
        self.e_type: str = element_type  # type of the element, only for arrays
        self.is_device = True if node_name == "kSeed" else False

    def MarkDeviceData(self):
        self.is_device = True
        # print("Variable {}".format(self.name))

    def MarkDeviceField(self, call_graph):
        self.is_device = True
        field_class = None
        if self.v_type == "list" and self.e_type not in ["int", "bool", "float", "RandomState"]:
            field_class = call_graph.GetClassNode(self.e_type)
        elif self.v_type not in ["int", "bool", "float", "RandomState"] \
        and self.name.split("_")[-1] == "ref":
            field_class = call_graph.GetClassNode(self.v_type)

        if field_class is not None and not field_class.is_device:
            call_graph.MarkDeviceDataByClassName([field_class.name])

# A tree which represents the calling and declaring relationships
class CallGraph(CallGraphNode):

    def __init__(self, node_name):
        super().__init__(node_name)
        # self.declared_classes: Set[ClassNode] = set()  # global classes
        self.declared_classes: Set[ClassNode] = set()

        random_class: ClassNode = ClassNode("random", None)
        random_class.declared_functions.add(FunctionNode("getrandbits", "random", "int"))
        random_class.declared_functions.add(FunctionNode("uniform", "random", "float"))
        random_class.declared_functions.add(FunctionNode("seed", "random", None))
        self.declared_classes.add(random_class)

        da_class: ClassNode = ClassNode("DeviceAllocator", None)
        da_class.declared_functions.add(FunctionNode("new", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("destroy", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("device_do", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("parallel_do", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("array", "DeviceAllocator", None))
        self.declared_classes.add(da_class)

        self.declared_types: Set[TypeNode] = set()
        self.declared_types.add(TypeNode("int"))
        self.declared_types.add(TypeNode("bool"))
        self.declared_types.add(TypeNode("float"))

        self.declared_functions: Set[FunctionNode] = set()  # global functions
        self.declared_variables: Set[VariableNode] = set()  # global variables
        self.library_functions: Set[FunctionNode] = set()  # library functions
        self.called_functions: Set[FunctionNode] = set()  # functions called globally(shouldn't be device function)
        self.called_variables: Set[VariableNode] = set()  # variables called globally

    def GetTypeNode(self, type_name):
        for class_node in self.declared_classes:
            if class_node.name == type_name:
                return class_node
        for type_node in self.declared_types:
            if type_node.name == type_name:
                return type_node
        return None
    
    # Find the class 'class_name'
    def GetClassNode(self, class_name):
        for class_node in self.declared_classes:
            if class_node.name == class_name and type(class_node) is ClassNode:
                return class_node
        return None

    def GetFunctionNode(self, function_name, class_name):
        for class_node in self.declared_classes:
            ret = class_node.GetFunctionNode(function_name, class_name)
            if ret is not None:
                return ret
        # find the function in the functions defined in the global block
        for function_node in self.declared_functions:
            if function_node.name == function_name:
                return function_node
        for function_node in self.library_functions:
            if function_node.name == function_name:
                return function_node
        return None

    # Mark all functions that needs to be allocated in the allocator
    def MarkDeviceDataByClassName(self, class_names):
        for cln in class_names:
            # do not support just mark a child class which nest in another class
            children = []
            for cls in self.declared_classes:
                if cls.super_class == cln and cls.name not in class_names \
                and not cls.is_device:
                    children.append(cls.name)

                if cls.name == cln:
                    # Mark all called functions in class cls
                    cls.MarkDeviceData()

                    for field in cls.declared_fields:
                        field.MarkDeviceField(self)

                    # Mark all called functions in the functions of cls
                    for func in cls.declared_functions:
                        func.MarkDeviceData()

                    # Mark all variables in that class cls (in the functions of cls)
                    for var in cls.declared_variables:
                        var.MarkDeviceData()
            
            if children:
                self.MarkDeviceDataByClassName(children)

    # Query whether the function is a device function
    def IsDeviceFunction(self, function_name, class_name):
        for class_node in self.declared_classes:
            if class_node.name == class_name:
                result = class_node.IsDeviceFunction(function_name, class_name)
                if result is not None:
                    return result
        for function_node in self.declared_functions:
            if function_node.name == function_name:
                return function_node.is_device
        print("Undeclared function '{}.{}'".format(class_name, function_name))
