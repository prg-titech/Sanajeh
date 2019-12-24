import ast

source = open('./benchmarks/nbody.py', encoding="utf-8").read()
source2 = open('./python2cpp_examples/Sample.py', encoding="utf-8").read()
tree = ast.parse(source)

'''Let children nodes know their parents'''
for parent_node in ast.walk(tree):
    for child in ast.iter_child_nodes(parent_node):
        child.parent = parent_node


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


# Find all functions and variables they will be copied in device memory. HUNT THEM!
class HunterVisitor(ast.NodeVisitor):
    __sv = ScoutVisitor()
    variables = {}
    functions = []

    @property
    def classes(self):
        return self.__sv.classes

    def visit_Module(self, node):
        self.__sv.visit(node)

        for x in node.body:
            # Only visit those classes that are intended to be allocated in device
            if type(x) is ast.ClassDef and x.name in self.classes:
                self.visit(x)

    # def visit_FunctionDef(self, node):
    #     pass

    def visit_AnnAssign(self, node):
        self.variables.setdefault(node.target.id, []).append(node.annotation.id)
        print(self.variables)
        self.generic_visit(node)

        pn = node.parent
        #
        # To know whether a variable which appears in the value of the assignment is an argument, if so, then we don't
        # want to create a device version of this variable, if not so, the variable probably is an global variable(?) so
        # we need to mark that
        #
        while (type(pn) is not ast.ClassDef) and (type(pn) is not ast.FunctionDef):
            pn = pn.parent
        # print(type(pn))

    def visit_BinOp(self, node):
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)


# haven't support import yet
class GenCppVisitor(ast.NodeVisitor):
    __hv = HunterVisitor()
    __h_code = ''
    __h_include = '#include "allocator_config.h"\n'
    __h_class_pre_declare = 'class '

    # Find classes that needs to be allocated in device memory when visit a Module
    def visit_Module(self, node):
        self.__hv.visit(node)
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
    gcv = GenCppVisitor()
    gcv.visit(tree)
    # gcv.print_classes()
