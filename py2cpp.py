import ast

source = open('./benchmarks/nbodyclass.py', encoding="utf-8").read()
source2 = open('./python2cpp_examples/Sample.py', encoding="utf-8").read()
tree = ast.parse(source)




class CppBuilder:
    h_code = ''
    h_include = '#include "allocator_config.h"\n'
    h_class_pre_declare = 'class '
    classes = []

    def addclassname(self, clsn):
        self.classes.append(clsn)


# haven't support import yet
class GenCppVisitor(ast.NodeVisitor):

    cb = CppBuilder()

    def generic_visit(self, node):
        return ast.NodeVisitor.generic_visit(self, node)

    def visit_ClassDef(self, node):
        self.cb.addclassname(node.name)
        for x in node.body:
            self.generic_visit(x)

    def visit_FunctionDef(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_AnnAssign(self, node):
        annotation = node.annotation
        print(annotation)
        # ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node):
        print(node.id)

    def visit_Num(self, node):
        # print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)




if __name__ == '__main__':
    gcv = GenCppVisitor()
    gcv.visit(tree)
    print(gcv.cb.classes)



