# -*- coding: utf-8 -*-
# Mark all device functions

import ast
from typing import List
from typing import Set


# Function tree
class BlockTreeNode:

    def __init__(self, class_or_function, function_name, class_name, identifier_name):
        # whether this block is a function or a class, true for class and false for func, global is explained as class
        self.c_or_f = class_or_function
        self.f_name: str = function_name  # function name
        self.c_name: str = class_name  # class name
        self.i_name: str = identifier_name  # identifier name

        # if the function is called in a class but not in any of the functions in that class,
        # the class name will be added here for marking purpose
        self.called_c_name = set()
        # classes declared in this block
        self.declared_classes: List[BlockTreeNode] = []
        # functions declared in this block
        self.declared_Functions: List[BlockTreeNode] = []
        # variables declared in this block
        self.declared_variables: List[str] = []
        self.called_Functions: Set[BlockTreeNode] = set()  # functions called in this block
        self.is_device = False  # if it is an __device__ block

    # Find the function 'function_name' by BFS
    def GetFunctionNode(self, function_name, class_name, identifier_name):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_Functions:
                if x.f_name == function_name and x.c_name == class_name and x.i_name == identifier_name:
                    return x
                q.append(x)
            q.pop(0)
        return None

    # Mark all the functions called by this function node to device function
    def RecursiveMark(self):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            self.is_device = True
            # print('Marking, function: {}, class: {}, identifier: {}'.format(nd.f_name, nd.c_name, nd.i_name))
            for x in nd.called_Functions:
                q.append(x)
            q.pop(0)
        return None

    # Query whether the function is a device function
    def IsDeviceFunction(self, function_name, class_name, identifier_name):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_Functions:
                if x.f_name == function_name and x.c_name == class_name and x.i_name == identifier_name:
                    return x.is_device
                q.append(x)
            q.pop(0)
        return False


# Generate python function/variable tree
class GenPyTreeVisitor(ast.NodeVisitor):
    __root = BlockTreeNode(True, 'global', None, None)

    @property
    def root(self):
        return self.__root

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        pn = node.parent
        while type(pn) is not ast.Module:
            print("Error, doesn't support nested classes")
            pn = pn.parent
        
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
                func_node = BlockTreeNode(False, func_name, None, None)
                self.root.declared_Functions.append(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                return
        elif type(pn) is ast.ClassDef:
            func_node = self.root.GetFunctionNode(func_name, pn.name, None)
            if func_node is None:
                func_node = BlockTreeNode(False, func_name, pn.name, None)
                self.root.declared_Functions.append(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                return
        self.generic_visit(node)

    # Analyze function calling relationships
    def visit_Call(self, node):
        pn = node.parent

        # Get the function name
        func_name = None
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
                call_node = BlockTreeNode(False, func_name, None, id_name)
                self.root.declared_Functions.append(call_node)
            self.root.called_Functions.add(call_node)

        # Called in the class
        elif type(pn) is ast.ClassDef:
            call_node = self.root.GetFunctionNode(False, func_name, None, id_name)
            if call_node is None:
                call_node = BlockTreeNode(False, func_name, None, id_name)
                self.root.declared_Functions.append(call_node)
            call_node.called_c_name.add(pn.name)

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
                    call_node = BlockTreeNode(False, func_name, None, id_name)
                    self.root.declared_Functions.append(call_node)
                p_node.called_Functions.add(call_node)
            # The calling function is declared in a class
            elif type(pnn) is ast.ClassDef:
                p_node = self.root.GetFunctionNode(pn.name, pnn.name, None)
                # print(pnn.name+'.'+pn.name + '->' + func_name)
                call_node = self.root.GetFunctionNode(func_name, None, id_name)
                if call_node is None:
                    call_node = BlockTreeNode(False, func_name, None, id_name)
                    self.root.declared_Functions.append(call_node)
                p_node.called_Functions.add(call_node)
        self.generic_visit(node)

    # Mark all functions that needs to be allocated in the allocator
    def MarkFunctionsByClassName(self, class_names):
        for cln in class_names:
            for func in self.__root.declared_Functions:
                # Functions declared in device class
                if func.c_name == cln:
                    func.RecursiveMark()

                # Functions called in device class
                elif cln in func.called_c_name:
                    func.RecursiveMark()


class MarkVisitor(ast.NodeVisitor):
    def __init__(self, rt: BlockTreeNode):
        self.__root: BlockTreeNode = rt

    def visit_FunctionDef(self, node):
        pn = node.parent
        func_name = node.name
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef):
            pn = pn.parent
        cln = None
        if hasattr(pn, "name"):
            cln = pn.name
        if self.__root.IsDeviceFunction(func_name, cln, None):
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


sv = ScoutVisitor()  # Just for DEBUG propose


def mark(tree):
    # Let declared_Functions nodes know their parents
    for parent_node in ast.walk(tree):
        for child in ast.iter_child_nodes(parent_node):
            child.parent = parent_node
    sv.visit(tree)
    sv.MarkFunctions(tree)
