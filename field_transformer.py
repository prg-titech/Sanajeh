import ast
import astunparse
import sys
import copy

from call_graph import CallGraph
from py2cpp import GenPyCallGraphVisitor


class DeviceCodeVisitor(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.root: CallGraph = root
        self.node_path = [self.root]

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
        self.sdef_cls = set()  # classes that are used for fields or variables types

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
            self.sdef_cls.add(ann)
        return node

    def visit_arg(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in ("str", "float", "int", None):
                self.sdef_cls.add(ann)
        return node


class Normalizer(DeviceCodeVisitor):
    """Use variables to rewrite nested expressions"""

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


class FunctionBodyGenerator(ast.NodeTransformer):
    """Gernerate new ast nodes which are used by the Inliner to inline functions"""

    def __init__(self, node, sdef_cls):
        self.func_name = None
        self.caller = None
        self.args = []
        self.func_args = []
        self.new_ast_nodes = []
        self.node = copy.deepcopy(node)
        self.self_define_class = sdef_cls

    def visit_Module(self, node):
        for x in node.body:
            if type(x) == ast.ClassDef:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        if node.name not in self.self_define_class:
            return node
        else:
            node.body = [self.visit(x) for x in node.body]
        return node

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            #  check if argument number is correct
            if len(node.args.args)-1 != len(self.args):
                print("Invalid argument number.", file=sys.stderr)
                sys.exit(1)
            for arg in node.args.args:
                if arg.arg != "self":
                    self.func_args.append(arg.arg)
            for x in node.body:
                self.new_ast_nodes.append(self.visit(x))
        return node

    def visit_Name(self, node):
        if node.id == "self":
            return copy.deepcopy(self.caller)
        for i in range(len(self.func_args)):
            if node.id == self.func_args[i]:
                return copy.deepcopy(self.args[i])
        return node

    def GetTransformedNodes(self, caller, func_name, args):
        """Traverse the ast node given, find the target function and return the transformed implementation"""
        self.caller = copy.deepcopy(caller)
        self.func_name = func_name
        self.args = copy.deepcopy(args)
        self.visit(self.node)


class Inliner(DeviceCodeVisitor):
    """Replace function callings with specific implementation"""

    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls

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
        if type(node.value) == ast.Call:
            if type(node.value.func) == ast.Attribute:
                if type(node.value.func.value) == ast.Attribute:
                    func_body_gen = FunctionBodyGenerator(self.node, self.sdef_cls)
                    func_body_gen.GetTransformedNodes(node.value.func.value,
                                                      node.value.func.attr,
                                                      node.value.args)
                    if len(func_body_gen.new_ast_nodes) != 0:
                        return func_body_gen.new_ast_nodes[:-1]
        return node

    #
    # def visit_Assign(self, node):
    #     ret = []
    #     node.value = self.visit(node.value)
    #     if type(node.value) == list:
    #         for v in node.value:
    #             if type(v) == ast.Call:
    #                 v = ast.Assign(targets=node.targets, value=v)
    #             ret.append(v)
    #         return ret
    #     else:
    #         return node
    #
    # def visit_AnnAssign(self, node):
    #     if node.value is None:
    #         return node
    #     ret = []
    #     node.value = self.visit(node.value)
    #     if type(node.value) == list:
    #         for v in node.value:
    #             if type(v) == ast.Call:
    #                 v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v)
    #             ret.append(v)
    #         return ret
    #     else:
    #         return node

    # def visit_Call(self, node):
    #     ret = []
    #     print(node.func)
    #     #self.func_body_gen.GetTransformedNodes("a", "b")
    #     return node


class Eliminator(DeviceCodeVisitor):
    """Remove useless classes and objects and dissemble seld-defined type fields"""

    def __init__(self, root: CallGraph):
        super().__init__(root)


def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_cls:
            scr.sdef_cls.remove(cls)

    ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    ast.fix_missing_locations(Inliner(call_graph, node, scr.sdef_cls).visit(node))
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
