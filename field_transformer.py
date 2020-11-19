import ast
import astunparse
import sys
import copy

from call_graph import CallGraph, ClassNode, FunctionNode
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
        class_node = self.node_path[-1].GetClassNode(name)
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
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
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
        class_node = self.node_path[-1].GetClassNode(name)
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
        self.has_auto_variables = False
        self.last_annotation = None

    def visit_FunctionDef(self, node):
        self.v_counter = 0  # counter needs to be reset in every function
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
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
        self.has_auto_variables = False
        self.last_annotation = "None"
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
        self.has_auto_variables = False
        self.last_annotation = "None"
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
        self.has_auto_variables = False
        self.last_annotation = "None"
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
        # if self.has_auto_variables:
        if hasattr(node.func, "value") and type(node.func.value) == ast.Name and self.has_auto_variables:
            if type(self.node_path[-1]) == FunctionNode:
                var_type = self.node_path[-1].GetVariableType(node.func.value.id)
                if var_type is None:
                    class_node = self.node_path[0].GetClassNode(self.last_annotation)
                    for x in class_node.declared_functions:
                        if x.name == node.func.attr:
                            self.last_annotation = x.ret_type
                            break
                else:
                    for x in self.node_path[-1].called_functions:
                        if x.name == node.func.attr and x.c_name == var_type:
                            self.last_annotation = x.ret_type
                            break

        elif hasattr(node.func, "value") and type(node.func.value) == ast.Attribute and self.has_auto_variables:
            if node.func.value.value.id == "self":
                var_type = None
                for x in self.node_path[-2].declared_variables:
                    if x.name == node.func.value.attr:
                        var_type = x.v_type
                        break
                for x in self.node_path[-1].called_functions:
                    if x.name == node.func.attr and x.c_name == var_type:
                        self.last_annotation = x.ret_type
                        break
            else:
                var_type = None
                caller_type = self.node_path[-1].GetVariableType(node.func.value.value.id)
                var_class = self.node_path[0].GetClassNode(caller_type)
                for x in var_class.declared_variables:
                    if x.name == node.func.value.attr:
                        var_type = x.v_type
                        break
                caller_class = self.node_path[0].GetClassNode(var_type)
                for x in caller_class.declared_functions:
                    if x.name == node.func.attr:
                        self.last_annotation = x.ret_type
                        break

        # self.f.A().B()...
        if hasattr(node.func, "value") and type(node.func.value) == ast.Call:
            self.has_auto_variables = True
            assign_nodes = self.visit(node.func.value)
            new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
            self.visit(assign_nodes[-1])
            ret.extend(assign_nodes[:-1])
            if type(self.node_path[-2]) == ClassNode:
                pass
            else:
                print("Invalid data structure.", file=sys.stderr)
                sys.exit(1)
            ret.append(ast.AnnAssign(target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store()),
                                     value=assign_nodes[-1],
                                     simple=1,
                                     annotation=ast.Name(id=self.last_annotation, ctx=ast.Load())
                                     ))
            self.v_counter += 1
            node.func.value = new_var_node

        for x in range(len(node.args)):
            # self.f.A(B())
            if type(node.args[x]) == ast.Call:
                self.has_auto_variables = True
                assign_nodes = self.visit(node.args[x])
                new_var_node = ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load())  # change argument
                ret.extend(assign_nodes[:-1])
                self.visit(assign_nodes[-1])
                ret.append(ast.AnnAssign(target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Store()),
                                         value=assign_nodes[-1],
                                         simple=1,
                                         annotation=ast.Name(id=self.last_annotation, ctx=ast.Load())
                                         ))
                self.v_counter += 1
                node.args[x] = new_var_node
        ret.append(node)
        return ret


class FunctionBodyGenerator(ast.NodeTransformer):
    """Gernerate new ast nodes which are used by the Inliner to inline functions"""

    def __init__(self, node, caller_type):
        self.func_name = None
        self.caller = None
        self.args = []
        self.func_args = []
        self.new_ast_nodes = []
        self.node = copy.deepcopy(node)
        self.caller_type = caller_type

    def visit_Module(self, node):
        for x in node.body:
            if type(x) == ast.ClassDef:
                self.visit(x)
        return node

    def visit_ClassDef(self, node):
        if node.name != self.caller_type:
            return node
        else:
            node.body = [self.visit(x) for x in node.body]
        return node

    def visit_FunctionDef(self, node):
        if node.name == self.func_name:
            #  check if argument number is correct
            if len(node.args.args) - 1 != len(self.args):
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
            return self.caller
        for i in range(len(self.func_args)):
            if node.id == self.func_args[i]:
                return self.args[i]
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
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
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
                if type(v) == ast.Return:
                    continue
                ret.append(v)
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.Assign(targets=node.targets, value=v.value)
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
                if type(v) == ast.Return:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v.value)
                ret.append(v)
            return ret
        else:
            return node

    def visit_Call(self, node):
        if type(node.func) == ast.Attribute:
            if type(node.func.value) == ast.Attribute:
                if type(node.func.value.value) == ast.Name:
                    if node.func.value.value.id == "self" and type(self.node_path[-2]) is ClassNode:
                        caller_type = None
                        for x in self.node_path[-2].declared_variables:
                            if x.name == node.func.value.attr:
                                caller_type = x.v_type
                        if caller_type not in self.sdef_cls:
                            return node
                        func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                        func_body_gen.GetTransformedNodes(node.func.value,
                                                          node.func.attr,
                                                          node.args)
                        if len(func_body_gen.new_ast_nodes) != 0:
                            return func_body_gen.new_ast_nodes
                    else:
                        var_type = None
                        caller_type = self.node_path[-1].GetVariableType(node.func.value.value.id)
                        var_class = self.node_path[0].GetClassNode(caller_type)
                        for x in var_class.declared_variables:
                            if x.name == node.func.value.attr:
                                var_type = x.v_type
                                break
                        if var_type not in self.sdef_cls:
                            return node
                        func_body_gen = FunctionBodyGenerator(self.node, var_type)
                        func_body_gen.GetTransformedNodes(node.func.value,
                                                          node.func.attr,
                                                          node.args)
                        if len(func_body_gen.new_ast_nodes) != 0:
                            return func_body_gen.new_ast_nodes
            elif type(node.func.value) == ast.Name:
                caller_type = self.node_path[-1].GetVariableType(node.func.value.id)
                if caller_type not in self.sdef_cls:
                    return node
                func_body_gen = FunctionBodyGenerator(self.node, caller_type)
                func_body_gen.GetTransformedNodes(node.func.value,
                                                  node.func.attr,
                                                  node.args)
                if len(func_body_gen.new_ast_nodes) != 0:
                    return func_body_gen.new_ast_nodes
        return node


class Eliminator(DeviceCodeVisitor):
    """Inlining constructor functions"""

    def __init__(self, root: CallGraph, node, sdef_cls):
        super().__init__(root)
        self.node = node
        self.sdef_cls = sdef_cls
        self.var_dict = {}

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if func_node.is_device:
            self.var_dict = {}
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
                elif node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.annotation.id in self.sdef_cls and node.value is not None:
            if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                if node.value.func.id in self.sdef_cls:
                    func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                    func_body_gen.GetTransformedNodes(node.target,
                                                      "__init__",
                                                      node.value.args)
                    if len(func_body_gen.new_ast_nodes) != 0:
                        return func_body_gen.new_ast_nodes
            else:
                if type(node.value) is ast.Name:
                    self.var_dict[node.target.id] = self.var_dict[node.value.id]
                else:
                    self.var_dict[node.target.id] = node.value
                return None
        return node

    def visit_Assign(self, node):
        ret = []
        self.generic_visit(node)
        for field in node.targets:
            if type(field) == ast.Attribute and type(self.node_path[-2]) == ClassNode:
                var_type = None
                for x in self.node_path[-2].declared_variables:
                    if x.name == field.attr:
                        var_type = x.v_type
                        break
                if node.value is not None and var_type in self.sdef_cls:
                    if type(node.value) == ast.Call and type(node.value.func) == ast.Name:
                        if node.value.func.id in self.sdef_cls:
                            func_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                            func_body_gen.GetTransformedNodes(field,
                                                              "__init__",
                                                              node.value.args)
                            if len(func_body_gen.new_ast_nodes) != 0:
                                ret.extend(func_body_gen.new_ast_nodes)
        if len(ret) != 0:
            return ret
        return node

    def visit_Name(self, node):
        if node.id in self.var_dict:
            return self.var_dict[node.id]
        return node


class FieldSynthesizer(DeviceCodeVisitor):
    """Synthesize new fields from the references to the nested objects"""

    def __init__(self, root: CallGraph, sdef_cls):
        super().__init__(root)
        self.sdef_cls = sdef_cls
        self.field_dict = {}

    def visit_ClassDef(self, node):
        name = node.name
        class_node = self.node_path[-1].GetClassNode(name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if class_node.is_device:
            self.field_dict = {}
            self.node_path.append(class_node)
            node.body = [self.visit(x) for x in node.body]
            i = 0
            for x in range(len(node.body)):
                if node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            i = 0
            for field in self.field_dict:
                node.body.insert(i, ast.AnnAssign(target=ast.Name(id=field, ctx=ast.Store()),
                                                  annotation=ast.Name(id=self.field_dict[field], ctx=ast.Load()),
                                                  simple=1,
                                                  value=None
                                                  )
                                 )
                i += 1
            self.node_path.pop()
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.node_path[-1].GetFunctionNode(name, self.node_path[-1].name)
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
                elif node.body[i] is None:
                    del node.body[i]
                else:
                    i += 1
            self.node_path.pop()
        return node

    def visit_Attribute(self, node):
        ctx = None
        if type(node.ctx) == ast.Load:
            ctx = ast.Load()
        elif type(node.ctx) == ast.Store:
            ctx = ast.Store()
        if type(node.value) == ast.Name:
            if self.node_path[-1].GetVariableType(node.value.id) in self.sdef_cls:
                return ast.Name(id=node.value.id + "_" + node.attr, ctx=ctx)
        elif type(node.value) == ast.Attribute and type(self.node_path[-2]) == ClassNode:
            var_type = None
            for x in self.node_path[-2].declared_variables:
                if x.name == node.value.attr:
                    var_type = x.v_type
                    break
            if var_type in self.sdef_cls:
                return ast.Attribute(attr=node.value.attr + "_" + node.attr, ctx=ctx, value=node.value.value)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.annotation.id in self.sdef_cls:
            if node.value is None:
                return None
        if type(node.target) == ast.Attribute and node.target.value.id == "self":
            self.field_dict[node.target.attr] = node.annotation.id
            return ast.Assign(targets=[node.target], value=node.value)
        node.simple = 1
        return node


def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    ast.fix_missing_locations(Normalizer(call_graph).visit(node))
    gpcgv1 = GenPyCallGraphVisitor()
    gpcgv1.visit(node)
    if not gpcgv1.mark_device_data(tree):
        print("No device data found")
        sys.exit(1)
    call_graph = gpcgv1.root
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_cls:
            scr.sdef_cls.remove(cls)
    ast.fix_missing_locations(Inliner(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(Eliminator(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(FieldSynthesizer(call_graph, scr.sdef_cls).visit(node))


tree = ast.parse(open('./benchmarks/nbody_vector_test.py', encoding="utf-8").read())
# Generate python call graph
gpcgv = GenPyCallGraphVisitor()
gpcgv.visit(tree)
# Mark all device data on the call graph
if not gpcgv.mark_device_data(tree):
    print("No device data found")
    sys.exit(1)

transform(tree, gpcgv.root)

path_w = 'test_output_final.py'
s = astunparse.unparse(tree)
with open(path_w, mode='w') as f:
    f.write(s)
