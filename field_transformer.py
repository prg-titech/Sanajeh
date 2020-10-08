import ast
import sys

from call_graph import CallGraph
from py2cpp import GenPyCallGraphVisitor


class FieldTransformer:
    class Searcher(ast.NodeVisitor):
        """Find all classes that are used for fields or variables types in device codes"""

        def __init__(self, root: CallGraph):
            self.__root: CallGraph = root
            self.__node_path = [self.__root]
            self.__current_node = None
            self.dev_cls = set()  # device classes
            self.sdef_csl = set()  # classes that are used for fields or variables types

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            super(FieldTransformer.Searcher, self).visit(node)

        def visit_Module(self, node):
            for x in node.body:
                if type(x) is ast.ClassDef:
                    self.visit(x)

        def visit_FunctionDef(self, node):
            name = node.name
            func_node = self.__current_node.GetFunctionNode(name, None)
            if func_node is None:
                # Program shouldn't come to here, which means the function is not analyzed by the marker yet
                print("The function {} does not exist.".format(name), file=sys.stderr)
                sys.exit(1)
            # If it is not a device function just skip
            if func_node.is_device:
                self.__node_path.append(func_node)
                for x in node.body:
                    self.visit(x)
                self.__node_path.pop()

        def visit_ClassDef(self, node):
            name = node.name
            class_node = self.__current_node.GetClassNode(name, None)
            if class_node is None:
                # Program shouldn't come to here, which means the class is not analyzed by the marker yet
                print("The class {} does not exist.".format(name), file=sys.stderr)
                sys.exit(1)
            # If it is not a device class just skip
            if class_node.is_device:
                self.dev_cls.add(name)
                self.__node_path.append(class_node)
                for x in node.body:
                    self.visit(x)
                self.__node_path.pop()

        def visit_AnnAssign(self, node):
            ann = node.annotation.id


    class Normalizer(ast.NodeTransformer):
        """Use variables to rewrite nested expressions"""

        def __init__(self, root: CallGraph):
            self.__root: CallGraph = root
            self.__node_path = [self.__root]
            self.__current_node = None

        def visit(self, node):
            self.__current_node = self.__node_path[-1]
            self.generic_visit(node)
            return node

    class Expander(ast.NodeTransformer):
        """Replace function callings with specific implementation"""
        pass

    class Eliminator(ast.NodeTransformer):
        """Remove useless classes and objects and dissemble seld-defined type fields"""
        pass

    def transform(self, node, call_graph):
        """Replace all self-defined type fields with specific implementation"""
        scr = self.Searcher(call_graph)
        scr.visit(node)
        # print(scr.dev_cls, scr.sdef_csl)
        # node = ast.fix_missing_locations(self.Normalizer(call_graph).visit(node))
        # node = ast.fix_missing_locations(self.Expander().visit(node))
        # node = ast.fix_missing_locations(self.Eliminator().visit(node))
        return node


tree = ast.parse(open('./benchmarks/nbody_vector.py', encoding="utf-8").read())
# Generate python call graph
gpcgv = GenPyCallGraphVisitor()
gpcgv.visit(tree)
# Mark all device data on the call graph
if not gpcgv.mark_device_data(tree):
    print("No device data found")
    sys.exit(1)
fr = FieldTransformer()
fr.transform(tree, gpcgv.root)
pass
