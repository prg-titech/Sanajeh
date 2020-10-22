import ast
import astunparse
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
    # todo: nested expressions in ifexp, assign, annassign nodes

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.v_counter = 0  # used to count the auto generated variables

    def visit_FunctionDef(self, node):
        self.v_counter = 0  # counter needs to be reset in every function
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if type(node.body[i]) == list:
                    for n in node.body[i]:
                        node.body.insert(i, n)
                        i += 1
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Expr(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.Expr(value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.Assign(targets=node.targets, value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Call:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Call(self, node):
        ret = []
        # self.f.A().B()...
        if hasattr(node.func, "value") and type(node.func.value) == ast.Call:
            assign_nodes = self.visit(node.func.value)
            new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
            ret.extend(assign_nodes[:-1])
            ret.append(ast.Assign(targets=[ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store())],
                                  value=assign_nodes[-1]
                                  ))
            self.v_counter += 1
            node.func.value = new_var_node
        for x in range(len(node.args)):
            # self.f.A(B())
            if type(node.args[x]) == ast.Call:
                assign_nodes = self.visit(node.args[x])
                new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
                ret.extend(assign_nodes[:-1])
                ret.append(ast.Assign(targets=[ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store())],
                                      value=assign_nodes[-1]
                                      ))
                self.v_counter += 1
                node.args[x] = new_var_node
        ret.append(node)
        return ret


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

    ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    # ast.fix_missing_locations(Expander(call_graph).visit(node))
    # ast.fix_missing_locations(Eliminator(call_graph).visit(node))


tree = ast.parse(open('./benchmarks/nbody_vector_test.py', encoding="utf-8").read())
# Generate python call graph
gpcgv = GenPyCallGraphVisitor()
gpcgv.visit(tree)
# Mark all device data on the call graph
if not gpcgv.mark_device_data(tree):
    print("No device data found")
    sys.exit(1)

transform(tree, gpcgv.root)

path_w = './test_output.py'
s = astunparse.unparse(tree)
with open(path_w, mode='w') as f:
    f.write(s)
pass
