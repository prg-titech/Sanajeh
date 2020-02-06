# -*- coding: utf-8 -*-
# Mark all device functions

import ast
from blockTree import BlockTree, ClassTreeNode, FunctionTreeNode


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
            # todo
            print("Nested classes")
            pn = pn.parent
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
        if type(pn) is ast.Module:
            func_node = self.root.GetFunctionNode(func_name, None, None)
            if func_node is None:
                func_node = FunctionTreeNode(func_name, None)
                self.root.declared_functions.add(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                return
        elif type(pn) is ast.ClassDef:
            func_node = self.root.GetFunctionNode(func_name, pn.name, None)
            if func_node is None:
                class_node = self.root.GetClassNode(pn.name, None)
                if class_node is None:
                    # Program shouldn't come to here, which means the parent class does not exist
                    print("The class {} does not exist.".format(pn.name))
                    assert False
                func_node = FunctionTreeNode(func_name, None)
                class_node.declared_functions.add(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                assert False
        self.generic_visit(node)

    # Analyze function calling relationships
    def visit_Call(self, node):
        pn = node.parent

        # Get the function name
        func_name = None
        # todo id_name maybe class name
        id_name = None
        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            id_name = node.func.value.id
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        # Locate where the function is called
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent

        # Called in global block
        if type(pn) is ast.Module:
            call_node = self.root.GetFunctionNode(func_name, None, id_name)
            if call_node is None:
                call_node = FunctionTreeNode(func_name, id_name)
                self.root.declared_functions.add(call_node)
            self.root.called_functions.add(call_node)

        # Called in the class
        elif type(pn) is ast.ClassDef:
            class_node = self.root.GetClassNode(pn.name, None)
            if class_node is None:
                # Program shouldn't come to here, which means the parent class does not exist
                print("The class {} does not exist.".format(pn.name))
                assert False
            call_node = self.root.GetFunctionNode(func_name, None, id_name)
            if call_node is None:
                call_node = FunctionTreeNode(func_name, id_name)
                # Maybe a functions from another package
                self.root.declared_functions.add(call_node)
            class_node.called_functions.add(call_node)

        # Called by another function
        elif type(pn) is ast.FunctionDef:
            pnn = pn.parent
            # To figure out which the calling function is
            while (type(pnn) is not ast.Module) and (type(pnn) is not ast.ClassDef):
                pnn = pnn.parent
            # The calling function is declared in global block
            if type(pnn) is ast.Module:
                p_node = self.root.GetFunctionNode(pn.name, None, None)
                call_node = self.root.GetFunctionNode(func_name, None, id_name)
                if call_node is None:
                    call_node = FunctionTreeNode(func_name, id_name)
                    self.root.declared_functions.add(call_node)
                p_node.called_functions.add(call_node)
            # The calling function is declared in a class
            elif type(pnn) is ast.ClassDef:
                p_node = self.root.GetFunctionNode(pn.name, pnn.name, None)
                # print(pnn.name+'.'+pn.name + '->' + func_name)
                call_node = self.root.GetFunctionNode(func_name, None, id_name)
                if call_node is None:
                    call_node = FunctionTreeNode(func_name, id_name)
                    self.root.declared_functions.add(call_node)
                p_node.called_functions.add(call_node)
        self.generic_visit(node)

    # Mark all functions that needs to be allocated in the allocator
    def MarkFunctionsByClassName(self, class_names):
        self.__root.MarkFunctionsByClassName(class_names)


class MarkVisitor(ast.NodeVisitor):
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


# Find classes which will be allocated in the device memory. MARK THEM!
class ScoutVisitor(ast.NodeVisitor):
    # Class names
    __classes = []
    __gen_pyTree_visitor = GenPyTreeVisitor()

    @property
    def classes(self):
        return self.__classes

    # Just for DEBUG propose
    @property
    def gen_pyTree_visitor(self):
        return self.__gen_pyTree_visitor

    # When visit a file first visit it by the pyTree generator
    def visit_Module(self, node):
        self.__gen_pyTree_visitor.visit(node)
        for x in node.body:
            self.visit(x)

    # Find the new_() function, and mark the class used in that function to device class
    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            if node.func.attr == 'new_':
                if node.args[0].id not in self.classes:
                    self.classes.append(node.args[0].id)
                    # print(node.args[0].id)

    # Mark all functions that needs to be allocated in the allocator
    def MarkFunctions(self, t):
        self.__gen_pyTree_visitor.MarkFunctionsByClassName(self.__classes)
        root = self.__gen_pyTree_visitor.root
        mv = MarkVisitor(root)
        mv.visit(t)


# mark the tree and return a marked BlockTree
def mark(tree):
    # Let declared_functions nodes know their parents
    for parent_node in ast.walk(tree):
        for child in ast.iter_child_nodes(parent_node):
            child.parent = parent_node
    sv = ScoutVisitor()  # Just for DEBUG propose
    sv.visit(tree)
    sv.MarkFunctions(tree)
    return sv.gen_pyTree_visitor.root
