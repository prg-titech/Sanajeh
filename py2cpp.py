import ast

source = open('./benchmarks/nbody.py', encoding="utf-8").read()
source2 = open('./python2cpp_examples/Sample1.py', encoding="utf-8").read()
tree = ast.parse(source2)


class GenCppVisitor(ast.NodeVisitor):

    cpp_code = ""

    def generic_visit(self, node):
        print(type(node).__name__)
        for x in node.body:
            x.visit()
        # ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node):
        print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node: ast.Name):
        self.cpp_code += "{}".format(node.id)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Num(self, node):
        print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

# class CppBuilder:
#     code = ""
#
#     def build_Assign(self, *targets: ast.expr, value: ast.expr):
#        for x in targets:
#             self.code.join(x.)


if __name__ == '__main__':
    gcv = GenCppVisitor()
    gcv.visit(tree)
    print(gcv.cpp_code)



