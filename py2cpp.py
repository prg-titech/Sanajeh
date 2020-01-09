import ast

source = open('./benchmarks/nbody.py', encoding="utf-8").read()
source2 = open('./python2cpp_examples/Sample.py', encoding="utf-8").read()
tree = ast.parse(source)

'''Let declared_Functions nodes know their parents'''
for parent_node in ast.walk(tree):
    for child in ast.iter_child_nodes(parent_node):
        child.parent = parent_node


# function tree
# haven't thought about nested functions
# reserved for nested class 
class FunctionTreeNode:

    def __init__(self, nm):
        self.name = nm  # function name
        self.declared_Functions = set()  # functions declared in this function (nested functions)
        self.called_Functions = set()  # functions called by this function
        self.is_Device_Func = False  # if it is an __device__ function

    # Find the function 'nm' by BFS
    def GetFunctionNode(self, nm):
        q = [self]
        while len(q) != 0:
            nd = q[0]
            for x in nd.declared_Functions:
                if x.name == nm:
                    return x
                q.append(x)
            q.pop(0)
        return None


# Generate python function tree
class GenPyTreeVisitor(ast.NodeVisitor):
    root = FunctionTreeNode('global')

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        pn = node.parent
        func_node = None
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef):
            pn = pn.parent
        if type(pn) is ast.Module:
            func_node = FunctionTreeNode(node.name)
        elif type(pn) is ast.ClassDef:
            func_node = FunctionTreeNode(pn.name+'.'+node.name)
        self.root.declared_Functions.add(func_node)
        self.generic_visit(node)

    # Analyze function calling relationships
    def visit_Call(self, node):
        pn = node.parent

        # Get the function name
        func_name = None
        if type(node.func) is ast.Attribute:
            func_name = node.func.value.id+'.'+node.func.attr
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        # Locate where the function is called
        while (type(pn) is not ast.Module) and (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent

        # Called in global block
        if type(pn) is ast.Module:
            call_node = self.root.GetFunctionNode(func_name)
            if call_node is None:
                call_node = FunctionTreeNode(func_name)
                self.root.declared_Functions.add(call_node)
            self.root.called_Functions.add(call_node)
        # function calling in class
        # elif type(pn) is ast.ClassDef:
        #     func_name = pn.name + '->' + func_name
        #     print(func_name)

        # Called by another function
        elif type(pn) is ast.FunctionDef:
            pnn = pn.parent
            # To figure out which the calling function is
            while (type(pnn) is not ast.Module) and (type(pnn) is not ast.ClassDef):
                pnn = pnn.parent
            # The calling function is declared in global block
            if type(pnn) is ast.Module:
                p_node = self.root.GetFunctionNode(pn.name)
                call_node = self.root.GetFunctionNode(func_name)
                if call_node is None:
                    call_node = FunctionTreeNode(func_name)
                    self.root.declared_Functions.add(call_node)
                p_node.called_Functions.add(call_node)
            # The calling function is declared in a class
            elif type(pnn) is ast.ClassDef:
                p_node = self.root.GetFunctionNode(pnn.name+'.'+pn.name)
                # print(pnn.name+'.'+pn.name + '->' + func_name)
                call_node = self.root.GetFunctionNode(func_name)
                if call_node is None:
                    call_node = FunctionTreeNode(func_name)
                    self.root.declared_Functions.add(call_node)
                p_node.called_Functions.add(call_node)
        self.generic_visit(node)


# Find classes which will be allocated in the device memory. MARK THEM!
class ScoutVisitor(ast.NodeVisitor):
    # Class names
    __classes = []

    @property
    def classes(self):
        return self.__classes

    # Find the new_() function, and mark the class used in that function to device class
    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            if node.func.attr == 'new_':
                if node.args[0].id not in self.classes:
                    self.classes.append(node.args[0].id)
                    # print(node.args[0].id)


# haven't support import yet
class GenCppVisitor(ast.NodeVisitor):
    __sv = ScoutVisitor()
    __h_code = ''
    __h_include = '#include "allocator_config.h"\n'
    __h_class_pre_declare = 'class '

    # Find classes that needs to be allocated in device memory when visit a Module
    def visit_Module(self, node):
        self.__sv.visit(node)
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
        print(self.__hv.classes)


if __name__ == '__main__':
    # gcv = GenCppVisitor()
    # gcv.visit(tree)
    # gcv.print_classes()
    pt = GenPyTreeVisitor()
    pt.visit(tree)
