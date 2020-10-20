import ast
import sys

from call_graph import CallGraph
from py2cpp import GenPyCallGraphVisitor


class DeviceCodeVisitor(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.__root: CallGraph = root
        self.node_path = [self.__root]

    def visit_Module(self, node):
        for x in node.body:
            if type(x) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name, None)
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
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            self.visit(node.args)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node


class Searcher(DeviceCodeVisitor):
    """Find all classes that are used for fields or variables types in device codes"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.dev_cls = set()  # device classes
        self.sdef_csl = set()  # classes that are used for fields or variables types

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.dev_cls.add(name)
            self.node_path.append(class_node)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        ann = node.annotation.id
        if ann not in ("str", "float", "int"):
            self.sdef_csl.add(ann)
        return node

    def visit_arg(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in ("str", "float", "int", None):
                self.sdef_csl.add(ann)
        return node


class Normalizer(DeviceCodeVisitor):
    """Use variables to rewrite nested expressions"""

    def __init__(self, root: CallGraph):
        super().__init__(root)

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            self.visit(node.args)
            for x in node.body:
                self.visit(x)
            self.node_path.pop()
        return node


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
    print(scr.dev_cls, scr.sdef_csl)

    ret_node = ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    # ret_node = ast.fix_missing_locations(Expander(call_graph).visit(node))
    # ret_node = ast.fix_missing_locations(Eliminator(call_graph).visit(node))
    return ret_node


tree = ast.parse(open('./benchmarks/nbody_vector.py', encoding="utf-8").read())
# Generate python call graph
gpcgv = GenPyCallGraphVisitor()
gpcgv.visit(tree)
# Mark all device data on the call graph
if not gpcgv.mark_device_data(tree):
    print("No device data found")
    sys.exit(1)

new = transform(tree, gpcgv.root)
pass
