import ast
import sys
import pprint

from call_graph import CallGraph
from py2cpp import GenPyCallGraphVisitor


class DeviceCodeVisitor(ast.NodeTransformer):
    def __init__(self, root: CallGraph):
        self.__root: CallGraph = root
        self.node_path = [self.__root]
        self.current_node = None

    def visit(self, node):
        self.current_node = self.node_path[-1]
        super(DeviceCodeVisitor, self).visit(node)
        return node

    def visit_Module(self, node):
        for x in node.body:
            if type(x) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                self.visit(x)

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.current_node.GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.current_node.GetFunctionNode(name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()


class Searcher(DeviceCodeVisitor):
    """Find all classes that are used for fields or variables types in device codes"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.dev_cls = set()  # device classes
        self.sdef_csl = set()  # classes that are used for fields or variables types

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.current_node.GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()

    def visit_AnnAssign(self, node):
        ann = node.annotation.id
        if ann not in ("str", "float", "int"):
            self.sdef_csl.add(ann)


class Normalizer(DeviceCodeVisitor):
    """Use variables to rewrite nested expressions"""

    def __init__(self, root: CallGraph):
        super().__init__(root)


class Expander(DeviceCodeVisitor):
    """Replace function callings with specific implementation"""

    def __init__(self, root: CallGraph):
        super().__init__(root)


class Eliminator(DeviceCodeVisitor):
    """Remove useless classes and objects and dissemble seld-defined type fields"""

    def __init__(self, root: CallGraph):
        super().__init__(root)


def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_csl:
            scr.sdef_csl.remove(cls)

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

transform(tree, gpcgv.root)
pass
