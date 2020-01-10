import ast
from typing import List
from typing import Set

source = open('./benchmarks/nbody.py', encoding="utf-8").read()
source2 = open('./python2cpp_examples/Sample.py', encoding="utf-8").read()
tree = ast.parse(source)

'''Let declared_Functions nodes know their parents'''
for parent_node in ast.walk(tree):
    for child in ast.iter_child_nodes(parent_node):
        child.parent = parent_node


# function tree
# does not support nested functions and nested class
class FunctionTreeNode:

    def __init__(self, function_name, class_name, identifier_name):
        self.f_name: str = function_name  # function name
        self.c_name: str = class_name  # class name
        self.i_name: str = identifier_name  # identifier name

        # if the function is called in a class but not in any of the functions in that class,
        # the class name will be added here for marking purpose
        self.called_c_name: set[str] = set()

        self.declared_Functions: List[FunctionTreeNode] = []  # functions declared in this function (nested functions)
        self.called_Functions: Set[FunctionTreeNode] = set()  # functions called by this function
        self.is_Device_Func = False  # if it is an __device__ function

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
            self.is_Device_Func = True
            # print('Marking, function: {}, class: {}, identifier: {}'.format(nd.f_name, nd.c_name, nd.i_name))
            for x in nd.called_Functions:
                q.append(x)
            q.pop(0)
        return None


# Generate python function tree
class GenPyTreeVisitor(ast.NodeVisitor):
    __root = FunctionTreeNode('global', None, None)

    @property
    def root(self):
        return self.__root

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        pn = node.parent
        func_name = node.name
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef):
            pn = pn.parent
        if type(pn) is ast.Module:
            func_node = self.root.GetFunctionNode(func_name, None, None)
            if func_node is None:
                func_node = FunctionTreeNode(func_name, None, None)
                self.root.declared_Functions.append(func_node)
            else:
                # Program shouldn't come to here, which means a function is defined twice
                print("The function {} is defined twice.".format(func_name))
                return
        elif type(pn) is ast.ClassDef:
            func_node = self.root.GetFunctionNode(func_name, pn.name, None)
            if func_node is None:
                func_node = FunctionTreeNode(func_name, pn.name, None)
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
                call_node = FunctionTreeNode(func_name, None, id_name)
                self.root.declared_Functions.append(call_node)
            self.root.called_Functions.add(call_node)

        # Called in the class
        elif type(pn) is ast.ClassDef:
            call_node = self.root.GetFunctionNode(func_name, None, id_name)
            if call_node is None:
                call_node = FunctionTreeNode(func_name, None, id_name)
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
                    call_node = FunctionTreeNode(func_name, None, id_name)
                    self.root.declared_Functions.append(call_node)
                p_node.called_Functions.add(call_node)
            # The calling function is declared in a class
            elif type(pnn) is ast.ClassDef:
                p_node = self.root.GetFunctionNode(pn.name, pnn.name, None)
                # print(pnn.name+'.'+pn.name + '->' + func_name)
                call_node = self.root.GetFunctionNode(func_name, None, id_name)
                if call_node is None:
                    call_node = FunctionTreeNode(func_name, None, id_name)
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


# Find classes which will be allocated in the device memory. MARK THEM!
class ScoutVisitor(ast.NodeVisitor):
    # Class names
    __classes = []
    __gen_pyTree_visitor = GenPyTreeVisitor()

    @property
    def classes(self):
        return self.__classes

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
    def MarkFunctions(self):
        self.__gen_pyTree_visitor.MarkFunctionsByClassName(self.__classes)


# haven't support import yet
class GenCppVisitor(ast.NodeVisitor):
    __sv = ScoutVisitor()
    __h_code = ''
    __h_include = '#include "allocator_config.h"\n'
    __h_class_pre_declare = 'class '

    # Find classes that needs to be allocated in device memory when visit a Module
    def visit_Module(self, node):
        self.__sv.visit(node)
        self.__sv.MarkFunctions()
        for x in node.body:
            self.visit(x)

    # def generic_visit(self, node):
    #     return ast.NodeVisitor.generic_visit(self, node)

    def visit_ClassDef(self, node):
        pass
        # for x in node.body:
        #     self.generic_visit(x)

    def visit_FunctionDef(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    # only support int and float
    # def visit_AnnAssign(self, node):
    #     ant = node.annotation.id
    #     code = ""
    #     if ant == "float":
    #         code += "float "
    #     elif ant == "int":
    #         code += "int "
    #     elif ant == "str":
    #         pass
    #     else:
    #         pass
    #     # value = node.value
    #     # print(value)
    #
    #     # ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node):
        pass
        # print(node.id)

    def visit_Num(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    # Just for debug purpose
    def print_classes(self):
        print(self.__sv.classes)


if __name__ == '__main__':
    gcv = GenCppVisitor()
    gcv.visit(tree)
