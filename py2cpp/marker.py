# -*- coding: utf-8 -*-
# Mark all device functions

import ast
from blockTree import BlockTree, ClassTreeNode, FunctionTreeNode, VariableTreeNode


# Generate python function/variable tree
class GenPyTreeVisitor(ast.NodeVisitor):
    __root = BlockTree()

    @property
    def root(self):
        return self.__root

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        pn = node.parent
        while type(pn) is not ast.Module:
            print("Error, doesn't support nested classes")
            pn = pn.parent
            assert False
        class_node = ClassTreeNode(node.name, None)
        self.__root.declared_classes.add(class_node)
        self.generic_visit(node)

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        pn = node.parent
        func_name = node.name
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef):
            print("Error, doesn't support nested functions")
            pn = pn.parent
            assert False
        if type(pn) is ast.Module:
            func_node = self.__root.GetFunctionNode(func_name, None)
            if func_node is None:
                func_node = FunctionTreeNode(func_name, None)
                self.__root.declared_functions.add(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                assert False
        elif type(pn) is ast.ClassDef:
            class_node = self.__root.GetClassNode(pn.name, None)
            func_node = class_node.GetFunctionNode(func_name, None)
            if func_node is None:
                func_node = FunctionTreeNode(func_name, None)
                class_node.declared_functions.add(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                assert False
        self.generic_visit(node)

    # Create node for variables without type annotation
    def visit_Assign(self, node):
        pn = node.parent
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent

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
                # global variable
                if type(pn) is ast.Module:
                    var_node = self.__root.GetVariableNode(var_name, id_name, None)
                    if var_node is None:
                        var_node = VariableTreeNode(var_name, id_name, None)
                        self.__root.declared_variables.add(var_node)

                # class variable
                elif type(pn) is ast.ClassDef:
                    p_node = self.__root.GetClassNode(pn.name, None)
                    var_node = p_node.GetVariableNode(var_name, id_name, None)
                    # if variable is not in a class it maybe a global one
                    if var_node is None:
                        var_node = self.__root.GetVariableNode(var_name, id_name, None)
                        # if it is still None it means that it is a new variable
                        if var_node is None:
                            var_node = VariableTreeNode(var_name, id_name, None)
                            p_node.declared_variables.add(var_node)

                # local variable in function
                elif type(pn) is ast.FunctionDef:
                    pnn = pn.parent
                    while (type(pnn) is not ast.Module) and (type(pnn) is not ast.ClassDef):
                        pnn = pnn.parent
                    # global function
                    if type(pnn) is ast.Module:
                        p_node = self.__root.GetFunctionNode(pn.name, None)
                        var_node = p_node.GetVariableNode(var_name, id_name, None)
                        # if variable is not in this function it maybe a global variable
                        if var_node is None:
                            var_node = self.__root.GetVariableNode(var_name, id_name, None)
                            if var_node is None:
                                var_node = VariableTreeNode(var_name, id_name, None)
                                p_node.declared_variables.add(var_node)
                    # class function
                    elif type(pnn) is ast.ClassDef:
                        pp_node = self.__root.GetClassNode(pnn.name, None)
                        p_node = pp_node.GetFunctionNode(pn.name, None)
                        var_node = p_node.GetVariableNode(var_name, id_name, None)
                        # if variable is not in this function it maybe a class variable
                        if var_node is None:
                            var_node = pp_node.GetVariableNode(var_name, id_name, None)
                            # if variable is not in this class it maybe a global variable
                            if var_node is None:
                                var_node = self.__root.GetVariableNode(var_name, id_name, None)
                                if var_node is None:
                                    var_node = VariableTreeNode(var_name, id_name, None)
                                    p_node.declared_variables.add(var_node)
        self.generic_visit(node)

    # Create node for variables with type annotation
    def visit_AnnAssign(self, node):
        pn = node.parent
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent
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
            # global variable
            if type(pn) is ast.Module:
                var_node = self.__root.GetVariableNode(var_name, id_name, ann)
                if var_node is None:
                    var_node = VariableTreeNode(var_name, id_name, ann)
                    self.__root.declared_variables.add(var_node)

            # class variable
            elif type(pn) is ast.ClassDef:
                p_node = self.__root.GetClassNode(pn.name, None)
                var_node = p_node.GetVariableNode(var_name, id_name, ann)
                # if variable is not in a class it maybe a global one
                if var_node is None:
                    var_node = self.__root.GetVariableNode(var_name, id_name, ann)
                    # if it is still None it means that it is a new variable
                    if var_node is None:
                        var_node = VariableTreeNode(var_name, id_name, ann)
                        p_node.declared_variables.add(var_node)

            # local variable in function
            elif type(pn) is ast.FunctionDef:
                pnn = pn.parent
                while (type(pnn) is not ast.Module) and (type(pnn) is not ast.ClassDef):
                    pnn = pnn.parent
                # global function
                if type(pnn) is ast.Module:
                    p_node = self.__root.GetFunctionNode(pn.name, None)
                    var_node = p_node.GetVariableNode(var_name, id_name, ann)
                    # if variable is not in this function it maybe a global variable
                    if var_node is None:
                        var_node = self.__root.GetVariableNode(var_name, id_name, ann)
                        if var_node is None:
                            var_node = VariableTreeNode(var_name, id_name, ann)
                            p_node.declared_variables.add(var_node)
                # class function
                elif type(pnn) is ast.ClassDef:
                    pp_node = self.__root.GetClassNode(pnn.name, None)
                    p_node = pp_node.GetFunctionNode(pn.name, None)
                    var_node = p_node.GetVariableNode(var_name, id_name, ann)
                    # if variable is not in this function it maybe a class variable
                    if var_node is None:
                        var_node = pp_node.GetVariableNode(var_name, id_name, ann)
                        # if variable is not in this class it maybe a global variable
                        if var_node is None:
                            var_node = self.__root.GetVariableNode(var_name, id_name, ann)
                            if var_node is None:
                                var_node = VariableTreeNode(var_name, id_name, ann)
                                p_node.declared_variables.add(var_node)

        self.generic_visit(node)

    # Mark all device data in ast 'node'
    def MarkDeviceData(self, node):
        dds = DeviceDataSearcher(self.__root)
        dds.visit(node)
        # mark all device data in the BlockTree
        self.__root.MarkDeviceDataByClassName(dds.classes)
        # edit the abstract syntax tree
        nm = NodeMarker(self.__root)
        nm.visit(node)


# Analyze function calling relationships and find all device data
class DeviceDataSearcher(ast.NodeVisitor):
    __root: BlockTree
    __classes = []

    def __init__(self, bt):
        self.__root = bt

    @property
    def classes(self):
        return self.__classes

    # Analyze function calling relationships
    def visit_Call(self, node):

        # Find device classes
        if type(node.func) is ast.Attribute:
            if node.func.attr == 'new_':
                if node.args[0].id not in self.classes:
                    self.classes.append(node.args[0].id)
                    # print(node.args[0].id)

        pn = node.parent
        # Get the function name
        func_name = None
        # todo id_name maybe class name
        id_name = None
        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            id_name = node.func.value.id
            if id_name == 'self':
                self.generic_visit(node)
                return
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        # Locate where the function is called
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent

        # Called in global block
        if type(pn) is ast.Module:
            call_node = self.__root.GetFunctionNode(func_name, id_name)
            if call_node is None:
                call_node = FunctionTreeNode(func_name, id_name)
                self.__root.declared_functions.add(call_node)
            self.__root.called_functions.add(call_node)

        # Called in the class
        elif type(pn) is ast.ClassDef:
            class_node = self.__root.GetClassNode(pn.name, None)
            call_node = class_node.GetFunctionNode(func_name, id_name)
            # only add the function for the first time called
            if call_node is None:
                call_node = self.__root.GetFunctionNode(func_name, id_name)
                if call_node is None:
                    # Todo maybe functions from another package
                    # print("Called in the class, {} {}".format(func_name, id_name))
                    call_node = FunctionTreeNode(func_name, id_name)
                    self.__root.declared_functions.add(call_node)
            class_node.called_functions.add(call_node)

        # Called in another function
        elif type(pn) is ast.FunctionDef:
            pnn = pn.parent
            # To figure out which the calling function is
            while (type(pnn) is not ast.Module) and (type(pnn) is not ast.ClassDef):
                pnn = pnn.parent
            # The calling function is declared in global block
            if type(pnn) is ast.Module:
                p_node = self.__root.GetFunctionNode(pn.name, None)
                # todo avoid calling class functions directly
                call_node = self.__root.GetFunctionNode(func_name, id_name)
                if call_node is None:
                    # Todo maybe functions from another package
                    # print("Called in the global function, {} {}".format(func_name, id_name))
                    call_node = FunctionTreeNode(func_name, id_name)
                    self.__root.declared_functions.add(call_node)
                p_node.called_functions.add(call_node)
            # The calling function is declared in a class
            elif type(pnn) is ast.ClassDef:
                pp_node = self.__root.GetClassNode(pnn.name, None)
                p_node = pp_node.GetFunctionNode(pn.name, None)
                call_node = pp_node.GetFunctionNode(func_name, id_name)
                if call_node is None:
                    # Todo maybe functions from another package
                    # print("Called in the class function, {} {}".format(func_name, id_name))
                    call_node = self.__root.GetFunctionNode(func_name, id_name)
                    if call_node is None:
                        call_node = FunctionTreeNode(func_name, id_name)
                        self.__root.declared_functions.add(call_node)
                p_node.called_functions.add(call_node)
        self.generic_visit(node)


class NodeMarker(ast.NodeVisitor):
    def __init__(self, rt: BlockTree):
        self.__root: BlockTree = rt

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


# Mark the tree and return a marked BlockTree
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
