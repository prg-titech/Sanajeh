# -*- coding: utf-8 -*-
# Mark all device functions

import ast
from blockTree import BlockTreeRoot, ClassTreeNode, FunctionTreeNode, VariableTreeNode


# Generate python function/variable tree
class GenPyTreeVisitor(ast.NodeVisitor):
    __root = BlockTreeRoot('root', None)
    __node_path = []
    __variable_environment = {}

    @property
    def root(self):
        return self.__root

    # JUST FOR DEBUG
    @property
    def node_path(self):
        return self.__node_path

    # JUST FOR DEBUG
    @property
    def variable_environment(self):
        return self.__variable_environment

    def visit_Module(self, node):
        self.__node_path.append(self.__root)
        self.generic_visit(node)

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        current_node = self.__node_path[-1]
        if type(current_node) is not BlockTreeRoot:
            print("Error, doesn't support nested classes")
            assert False
        class_name = node.name
        class_node = current_node.GetClassNode(class_name, None)
        if class_node is not None:
            # Program shouldn't come to here, which means a class is defined twice
            print("The class {} is defined twice.".format(class_name))
            assert False
        class_node = ClassTreeNode(node.name, None)
        current_node.declared_classes.add(class_node)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        current_node = self.__node_path[-1]
        func_name = node.name
        if type(current_node) is not BlockTreeRoot and type(current_node) is not ClassTreeNode:
            print("Error, doesn't support nested functions")
            assert False
        func_node = current_node.GetFunctionNode(func_name, None)
        if func_node is not None:
            # Program shouldn't come to here, which means a function is defined twice
            print("The function {} is defined twice.".format(func_name))
            assert False
        func_node = FunctionTreeNode(func_name, None)
        current_node.declared_functions.add(func_node)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Add arguments to the environment
    def visit_arguments(self, node):
        current_node = self.__node_path[-1]
        if type(current_node) is not FunctionTreeNode:
            print('Unexpected node "{}"'.format(current_node.name))
            assert False
        for arg in node.args:
            self.__variable_environment.setdefault(current_node.name, []).append(arg.arg)

    # Add global variables to the environment
    def visit_Global(self, node):
        current_node = self.__node_path[-1]
        for global_variable in node.names:
            self.__variable_environment.setdefault(current_node.name, []).append(global_variable)
            var_node = self.__root.GetVariableNode(global_variable, None, None)
            if var_node is None:
                print("The global variable {} is not existed.".format(global_variable))
                assert False
            current_node.called_variables.add(var_node)

    # Create node for variables without type annotation
    def visit_Assign(self, node):
        current_node = self.__node_path[-1]
        for var in node.targets:
            var_name = None
            # todo id_name
            id_name = None

            if type(var) is ast.Attribute:
                var_name = var.attr
                # print(var_name, var.value.id)
                # todo Attribute variables(self should refer to the class not in the current block),
                # todo haven't thought about other ocaasions
                if var.value.id == 'self':
                    pass
            elif type(var) is ast.Name:
                var_name = var.id
                self.__variable_environment.setdefault(current_node.name, [])
                # print(self.__variable_environment)
                if var_name not in self.__variable_environment[current_node.name]:
                    var_node = VariableTreeNode(var_name, id_name, None)
                    current_node.declared_variables.add(var_node)
                    self.__variable_environment[current_node.name].append(var_name)

        self.generic_visit(node)

    # Create node for variables with type annotation
    def visit_AnnAssign(self, node):
        current_node = self.__node_path[-1]
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
            # todo haven't thought about other ocaasions
        elif type(var) is ast.Name:
            var_name = var.id
            self.__variable_environment.setdefault(current_node.name, [])
            # print(self.__variable_environment)
            if var_name not in self.__variable_environment[current_node.name]:
                var_node = VariableTreeNode(var_name, id_name, ann)
                current_node.declared_variables.add(var_node)
                self.__variable_environment[current_node.name].append(var_name)
            else:
                var_node = current_node.GetVariableNode(var_name, id_name, ann)
        self.generic_visit(node)

    def visit_Name(self, node):
        current_node = self.__node_path[-1]
        self.__variable_environment.setdefault(current_node.name, [])
        if node.id == 'self' and node.id in  self.__variable_environment[current_node.name]:
            return
        for annotate_location_node in self.__node_path[-2::-1]:
            self.__variable_environment.setdefault(annotate_location_node.name, [])
            if node.id in self.__variable_environment[annotate_location_node.name]:
                var_node = annotate_location_node.GetVariableNode(node.id, None, None)
                if var_node is None:
                    print('Unexpected error, can not find variable "{}"', node.id)
                    assert False
                current_node.called_variables.add(var_node)
                break

    # Mark all device data in ast 'node'
    def MarkDeviceData(self, node):
        dds = DeviceDataSearcher(self.__root)
        dds.visit(node)
        # mark all device data in the BlockTreeRoot
        self.__root.MarkDeviceDataByClassName(dds.classes)
        # edit the abstract syntax tree
        nm = ASTMarker(self.__root)
        nm.visit(node)


# Analyze function calling relationships and find the device classes
class DeviceDataSearcher(ast.NodeVisitor):
    __root: BlockTreeRoot
    __classes = []
    __node_path = []

    def __init__(self, rt: BlockTreeRoot):
        self.__root = rt
        self.__node_path.append(rt)

    @property
    def classes(self):
        return self.__classes

    def visit_ClassDef(self, node):
        current_node = self.__node_path[-1]
        class_name = node.name
        class_node = current_node.GetClassNode(class_name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not exist
            print("The class {} is not exist.".format(class_name))
            assert False
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        current_node = self.__node_path[-1]
        func_name = node.name
        func_node = current_node.GetFunctionNode(func_name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the class is not exist
            print("The function {} not exist.".format(func_name))
            assert False
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Analyze function calling relationships
    def visit_Call(self, node):
        current_node = self.__node_path[-1]
        # Find device classes
        if type(node.func) is ast.Attribute:
            if node.func.attr == 'new_':
                if node.args[0].id not in self.classes:
                    self.classes.append(node.args[0].id)

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
            if type(parent_node) is FunctionTreeNode:
                continue
            call_node = parent_node.GetFunctionNode(func_name, id_name)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionTreeNode(func_name, id_name)
            self.__root.library_functions.add(call_node)
        current_node.called_functions.add(call_node)
        self.generic_visit(node)


class ASTMarker(ast.NodeVisitor):
    __node_path = []

    def __init__(self, rt: BlockTreeRoot):
        self.__root: BlockTreeRoot = rt
        self.__node_path.append(rt)

    # todo not only function
    def visit_FunctionDef(self, node):
        pn = node.parent
        func_name = node.name
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef):
            pn = pn.parent
        class_name = None
        if hasattr(pn, "name"):
            class_name = pn.name
        if self.__root.IsDeviceFunction(func_name, class_name, None):
            node.is_device_f = True
        else:
            node.is_device_f = False


# Mark the tree and return a marked BlockTreeRoot
class Marker:

    @staticmethod
    def mark(tree):
        # Let declared_functions nodes know their parents
        for parent_node in ast.walk(tree):
            for child in ast.iter_child_nodes(parent_node):
                child.parent = parent_node
        gptv = GenPyTreeVisitor()
        gptv.visit(tree)
        gptv.MarkDeviceData(tree)
        return gptv.root
