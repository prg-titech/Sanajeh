import ast
import copy
import type_converter
import astunparse

from call_graph import CallGraph, TypeNode, ListTypeNode, RefTypeNode, ClassNode, FunctionNode, VariableNode

def pprint(node):
    print(astunparse.unparse(node))

def get_annotation(ann):
    if ann:
        if type(ann) is ast.Name:
            return ann.id, None
        elif type(ann) is ast.Attribute:
            return ann.attr, None
        elif type(ann) is ast.Subscript and ann.value.id == "list":
            return ann.value.id, get_annotation(ann.slice.value)[0]
    return None, None  

class Typer(ast.NodeVisitor):

    def __init__(self, node_path):
        self.node_path = node_path
    
    def get_VariableNode(self, var_name):
        i = len(self.node_path) - 1
        while i >= 0:
            for var_node in self.node_path[i].declared_variables:
                if var_node.name == var_name:
                    return var_node
            if type(self.node_path[i]) is FunctionNode:
                for arg_node in self.node_path[i].arguments:
                    if arg_node.name == var_name:
                        return arg_node
            i -= 1

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        if node.id == "self":
            if type(self.node_path[-2]) is ClassNode:
                return self.node_path[-2]
        else:
            var_node = self.get_VariableNode(node.id)
            if var_node:
                if var_node.v_type == "list":
                    elem_node = self.node_path[0].GetTypeNode(var_node.e_type)
                    return ListTypeNode(elem_node) if elem_node is not None else None
                else:
                    type_node = self.node_path[0].GetTypeNode(var_node.v_type)
                    if var_node.name.split("_")[-1] == "ref":
                        return RefTypeNode(type_node) if type_node is not None else None
                    else:
                        return type_node
            type_node = self.node_path[0].GetTypeNode(node.id)
            if type_node:
                return type_node
        return None

    def visit_Attribute(self, node):
        receiver_type = self.visit(node.value)
        if receiver_type and type(receiver_type) is not ListTypeNode:
            if type(receiver_type) is RefTypeNode:
                receiver_type = receiver_type.type_node
            for field in receiver_type.declared_fields:
                if field.name == node.attr:
                    if field.v_type == "list":
                        elem_node = self.node_path[0].GetTypeNode(field.e_type)
                        return ListTypeNode(elem_node) if elem_node is not None else None
                    else:
                        type_node = self.node_path[0].GetTypeNode(field.v_type)
                        if field.name.split("_")[-1] == "ref":
                            return RefTypeNode(type_node) if type_node is not None else None
                        else:
                            return type_node
        return None
    
    def visit_Subscript(self, node):
        receiver_type = self.visit(node.value)
        if type(receiver_type) is ListTypeNode:
            return receiver_type.type_node
        return None
    
    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            receiver_type = self.visit(node.func.value)
            if receiver_type:
                for func in receiver_type.declared_functions:
                    if func.name == node.func.attr:
                        return self.node_path[0].GetTypeNode(func.ret_type)
        return None


"""
Visitors for Python code transformers
"""
class DeviceCodeVisitor(ast.NodeTransformer):

    def __init__(self, root: CallGraph):
        self.root: CallGraph = root
        self.node_path = [self.root]

    def visit_Module(self, node):
        for node_body in node.body:
            if type(node_body) in [ast.ClassDef, ast.FunctionDef, ast.AnnAssign]:
                self.visit(node_body)
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

    def GetVariableNode(self, var_name):
        i = len(self.node_path) - 1
        while i >= 0:
            for var_node in self.node_path[i].declared_variables:
                if var_node.name == var_name:
                    return var_node
            i -= 1
        return None

    """
    Type a possibly nested attribute
    """
    def attribute_type(self, attribute):
        rec_type = None
        if type(attribute.value) == ast.Name:
            if attribute.value.id == "self":
                rec_type = self.node_path[-2].name
            else:
                rec_type = self.node_path[-1].GetVariableType(attribute.value.id)
        elif type(attribute.value) == ast.Attribute:
            rec_type = self.attribute_type(attribute.value)
        elif type(attribute.value) == ast.Subscript:
            rec_type = self.attribute_type(attribute.value.value)
        if rec_type is not None and rec_type not in ["int", "bool", "float"]:
            if self.node_path[0].GetClassNode(rec_type) is not None:
                rec_class = self.root.GetClassNode(rec_type)
                for field in rec_class.declared_fields:
                    if field.name == attribute.attr:
                        return field.v_type
                """
                rec_class_name = self.node_path[0].GetClassNode(rec_type).name
                if attribute.attr in Checker.original[rec_class_name]:
                    return Checker.original[rec_class_name][attribute.attr]
                """
        return None 

class Searcher(DeviceCodeVisitor):
    """Find all classes that are used for fields or variables types in device codes"""

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.dev_cls = set()    # device classes
        self.sdef_cls = set()   # classes that are used for fields or variables types

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
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in type_converter.type_map:
                self.sdef_cls.add(ann)
        return node

    def visit_arg(self, node):
        if hasattr(node.annotation, "id"):
            ann = node.annotation.id
            if ann not in type_converter.type_map and ann is not None:
                self.sdef_cls.add(ann)
        return node

class Normalizer(DeviceCodeVisitor):
    """
    Declare new variables to replace method calls nested inside other expressions.
    Example
      -- self.vel.add(self.force.multiply(kDt).divide(self.mass))
    is converted into
      -- __auto_v0: Vector = self.force.multiply(kDt)
      -- __auto_v1: Vector = __auto_v0.divide(self.mass)
      -- self.vel.add(__auto_v1)
    """

    def __init__(self, root: CallGraph):
        super().__init__(root)
        self.v_counter = 0  # used to count the auto generated variables
        self.last_annotation = None
        self.has_auto_variables = False
        self.built_nodes = []
        self.current_attr = None

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
        self.last_annotation = "None"
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.Expr(value=self.current_attr))
            return ret
        else:
            return node

    def visit_Assign(self, node):
        ret = []
        self.last_annotation = "None"
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.Assign(targets=node.targets, value=self.current_attr))
            return ret
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        ret = []
        self.last_annotation = "None"
        self.built_nodes = []
        self.current_attr = node.value
        node.value = self.visit(node.value)
        if self.built_nodes:
            for new_node in self.built_nodes:
                ret.append(new_node)
            ret.append(ast.AnnAssign(
                annotation=node.annotation, simple=node.simple, 
                target=node.target, value=self.current_attr))
            return ret
        else:
            return node

    def visit_Name(self, node):
        if node.id == "self":
            self.last_annotation = self.node_path[-2].name
        elif self.GetVariableNode(node.id) is not None:
            self.last_annotation = self.GetVariableNode(node.id).v_type
        self.current_attr = node
        return node
        
    def visit_Attribute(self, node):
        self.visit(node.value)
        if type(self.current_attr) == ast.Call and self.last_annotation is not None:
            new_node = ast.AnnAssign(
                target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
            self.current_attr = ast.Attribute(
                value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=node.value.ctx if hasattr(node.value, "ctx") else None),
                attr=node.attr, ctx=node.ctx)
            self.built_nodes.append(new_node)
            self.v_counter += 1
        else:
            self.current_attr = ast.Attribute(value=self.current_attr, attr=node.attr, ctx=node.ctx)
        class_node = self.root.GetClassNode(self.last_annotation)
        if class_node is not None:
            for field in class_node.declared_fields:
                if field.name == node.attr:
                    self.last_annotation = field.v_type
        return node

    def visit_Subscript(self, node):
        self.current_attr = node.slice.value
        self.visit(node.slice.value)
        new_index = ast.Index(self.current_attr)
        if type(node.value) == ast.Name:
            var_node = self.GetVariableNode(node.value.id)
            if var_node is not None:
                self.last_annotation = var_node.e_type[0]
            self.current_attr = node
        else:
            self.visit(node.value.value)
            if type(node.value.value) == ast.Call and self.last_annotation is not None:
                new_node = ast.AnnAssign(
                    target=ast.Name(id="__auto_v"+ str(self.v_counter), ctx=ast.Load()),
                    value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                self.built_nodes.append(new_node)
                self.current_attr = ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)
            else:
                self.current_attr = ast.Subscript(
                    value=ast.Attribute(value=self.current_attr, attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)
            class_node = self.root.GetClassNode(self.last_annotation)
            if class_node is not None:
                for field in class_node.declared_fields:
                    if field.name == node.value.attr:
                        self.last_annotation = field.e_type
        return node

    def visit_Call(self, node):
        new_args = []
        # Processing arguments
        for arg in node.args:
            self.current_attr = arg
            self.visit(arg)
            if type(arg) == ast.Call and self.last_annotation is not None:
                if hasattr(arg.func.value, "id") and arg.func.value.id == "random" \
                and arg.func.attr in ["getrandbits", "uniform"]:
                    new_args.append(arg)
                else:
                    new_node = ast.AnnAssign(
                        target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                    self.built_nodes.append(new_node)
                    new_args.append(ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()))
                    self.v_counter += 1
            else:
                new_args.append(self.current_attr)
        if type(node.func) == ast.Name:
            for class_func in self.node_path[-2].declared_functions:
                if class_func.name == node.func.id:
                    self.last_annotation = class_func.ret_type
            for class_node in self.root.declared_classes:
                if class_node.name == node.func.id:
                    self.last_annotation = class_node.name
            self.current_attr = node
        elif hasattr(node.func.value, "id") and node.func.value.id == "random" \
        and node.func.attr in ["getrandbits", "uniform"]:
            self.current_attr = node
        else:
            if hasattr(node.func.value, "func") and hasattr(node.func.value.func, "id") \
            and node.func.value.func.id == "super":
                return node
            self.visit(node.func.value)
            if type(node.func.value) == ast.Call and self.last_annotation is not None:
                new_node = ast.AnnAssign(
                    target=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                    value=self.current_attr, simple=1, annotation=ast.Name(id=self.last_annotation, ctx=ast.Load()))
                self.built_nodes.append(new_node)
                self.current_attr = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__auto_v" + str(self.v_counter), ctx=ast.Load()),
                        attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.func.value.keywords)
                self.v_counter += 1
            else:
                self.current_attr = ast.Call(
                    func=ast.Attribute(value=self.current_attr, attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.keywords)
            class_node = self.root.GetClassNode(self.last_annotation)
            if class_node is not None:
                for class_func in class_node.declared_functions:
                    if class_func.name == node.func.attr:
                        self.last_annotation = class_func.ret_type
        return node

class FunctionBodyGenerator(ast.NodeTransformer):
    """Generate new ast nodes which are used by the Inliner to inline functions"""

    def __init__(self, node, caller_type):
        self.func_name = None
        self.caller = None
        self.args = []
        self.func_args = []
        self.new_ast_nodes = []
        self.node = copy.deepcopy(node)
        self.caller_type = caller_type
        self.var_dict = {}

    def visit_Module(self, node):
        for class_node in node.body:
            if type(class_node) == ast.ClassDef and class_node.name == self.caller_type:
                self.visit(class_node)
        return node
    
    def visit_ClassDef(self, node):
        node.body = [self.visit(body_node) for body_node in node.body]
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

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if type(node.target) is ast.Name:
            if node.target.id not in self.var_dict:
                self.var_dict[node.target.id] = node.target.id + "_v" + str(len(self.var_dict))
                return ast.AnnAssign(
                    target=ast.Name(id=self.var_dict[node.target.id], ctx=node.target.ctx),
                    annotation=node.annotation, value=node.value, simple=node.simple)
        return node

    def visit_Name(self, node):
        if node.id == "self":
            return self.caller
        for i in range(len(self.func_args)):
            if node.id == self.func_args[i]:
                return self.args[i]
        if node.id in self.var_dict:
            return self.var_dict[node.id]
        return node

    def GetTransformedNodes(self, caller, func_name, args):
        """Traverse the ast node given, find the target function and return the transformed implementation"""
        self.caller = copy.deepcopy(caller)
        self.func_name = func_name
        self.args = copy.deepcopy(args)
        self.visit(self.node)

class Inliner(DeviceCodeVisitor):
    """
    Replace function call on non-device classes with the specific implementations
    Example
      -- __auto_v0: Vector = self.force.multiply(kDt)
    is converted into
      -- __auto_v0: Vector = Vector((self.force.x * kDt), (self.force.y * kDt))
    """
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
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        if type(Typer(self.node_path).visit(node.target)) is RefTypeNode:
            return node
        ret = []
        node.value = self.visit(node.value)  
        if type(node.value) == list:
            for v in node.value:
                if type(v) == ast.Return:
                    v = ast.AnnAssign(annotation=node.annotation, simple=node.simple, target=node.target, value=v.value)
                ret.append(v)
            return ret
        return node

    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            caller_type = Typer(self.node_path).visit(node.func.value)
            if type(caller_type) is ClassNode and caller_type.name in self.sdef_cls:
                func_body_gen = FunctionBodyGenerator(self.node, caller_type.name)
                func_body_gen.GetTransformedNodes(node.func.value, node.func.attr, node.args)
                if len(func_body_gen.new_ast_nodes) != 0:
                    return func_body_gen.new_ast_nodes
        return node                

# DONE: simplify implementation
class Eliminator(DeviceCodeVisitor):
    """
    ISSUE: 
    1. The following line causes problem, passes when there is no type annotation
    -- other.force: Vector = new_force
    // Solved by removing the part using var_dict in visit_AnnAssign //
    2. Reference type's nested fields are not expanded
    -- other.force = new_force
       is not converted into
    -- other.force.x = new_force.x
    -- other.force.y = new_force.y
    // Solved by hijacking the else in visit_AnnAssign and visit_Assign
    3. Assigning method call to attributes get annotations added although unnecessary
    -- m.pos = Vector((__auto_v3.x / 2), (__auto_v3.y / 2))
       is converted into
    -- m.pos.x: float = (__auto_v3.x / 2)
    -- m.pos.y: float = (__auto_v3.y / 2)
    // Solved by removing annotation during field synthesizing
    """
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
            # self.var_dict = {}
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
        target_type = Typer(self.node_path).visit(node.target)
        if type(target_type) is ClassNode and target_type.name in self.sdef_cls:
            # object instantiation
            if type(node.value) is ast.Call and type(node.value.func) is ast.Name:
                inst_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                inst_body_gen.GetTransformedNodes(node.target, "__init__", node.value.args)
                if len(inst_body_gen.new_ast_nodes) != 0:
                    return inst_body_gen.new_ast_nodes
            # attribute or name
            else:
                new_nodes = []
                for field_name in target_type.expanded_fields:
                    for nested_field in target_type.expanded_fields[field_name]:
                        annotation = None
                        if nested_field.v_type != "list":
                            annotation = ast.Name(id=nested_field.v_type, ctx=node.annotation.ctx)
                        else:
                            annotation = ast.Subscript(
                                value=ast.Name(id="list", ctx=node.annotation.ctx), 
                                slice=ast.Index(value=ast.Name(id=nested_field.e_type, ctx=node.annotation.ctx)), 
                                ctx=node.annotation.ctx)
                        value_ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                        new_node = ast.AnnAssign(
                            target=ast.Attribute(value=node.target, attr=nested_field.name, ctx=node.target.ctx),
                            annotation=annotation,
                            value=ast.Attribute(value=node.value, attr=nested_field.name, ctx=value_ctx),
                            simple=node.simple)
                        new_nodes.append(new_node)
                if len(new_nodes) != 0:
                    return new_nodes
        return node

    def visit_Assign(self, node):
        result = []
        for target in node.targets:
            target_type = Typer(self.node_path).visit(target)
            if type(target_type) is ClassNode and target_type.name in self.sdef_cls:
                if type(node.value) is ast.Call and type(node.value.func) is ast.Name:
                    inst_body_gen = FunctionBodyGenerator(self.node, node.value.func.id)
                    inst_body_gen.GetTransformedNodes(target, "__init__", node.value.args)
                    if len(inst_body_gen.new_ast_nodes) != 0:
                        result.extend(inst_body_gen.new_ast_nodes)
                else:
                    new_nodes = []
                    for field_name in target_type.expanded_fields:
                        for nested_field in target_type.expanded_fields[field_name]:
                            value_ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                            new_node = ast.Assign(
                                value=ast.Attribute(value=node.value, attr=nested_field.name, ctx=value_ctx),
                                targets=[ast.Attribute(value=target, attr=nested_field.name, ctx=target.ctx)]
                            )
                            result.append(new_node)
        return result if len(result) != 0 else node


# TODO: turn this into nested compatible
# TODO: field synthesizing goes wrong -> join with inliner?
class FieldSynthesizer(DeviceCodeVisitor):
    def __init__(self, root: CallGraph, sdef_cls):
        super().__init__(root)
        self.sdef_cls = sdef_cls
    
    def visit_ClassDef(self, node):
        class_node = self.node_path[-1].GetClassNode(node.name)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        if class_node.is_device:
            self.node_path.append(class_node)
            # node.body = [self.visit(body) for body in node.body]
            node_body = []
            for body in node.body:
                rewritten = self.visit(body)
                if type(rewritten) == list:
                    for singular_body in rewritten:
                        node_body.append(singular_body)
                else:
                    node_body.append(rewritten)
            node.body= node_body
            self.node_path.pop()

    def visit_FunctionDef(self, node):
        func_node = self.node_path[-1].GetFunctionNode(node.name, self.node_path[-1].name)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)    
        if func_node.is_device:
            self.node_path.append(func_node)
            node.args = self.visit(node.args)
            node_body = []
            # Add random_state_ field
            if node.name == "__init__" and self.node_path[-2].has_random_state:
                node_body.append(ast.AnnAssign(
                    target=ast.Attribute(attr="random_state_", value=ast.Name(id="self")),
                    annotation=ast.Attribute(attr="RandomState", value=ast.Name(id="DeviceAllocator")),
                    simple=1,
                    value=ast.Constant(value=None, kind=None)
                ))
            for x in node.body:
                node_body.append(self.visit(x))
            node.body = node_body
            self.node_path.pop()           
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if type(node.target) is ast.Attribute and self.node_path[-1].name != "__init__":
            return ast.Assign(targets=[node.target], value=node.value)
        node.simple = 1
        return node
    
    def visit_Attribute(self, node):
        ctx = None
        if type(node.ctx) == ast.Load:
            ctx = ast.Load()
        elif type(node.ctx) == ast.Store:
            ctx = ast.Store()
        value_type = Typer(self.node_path).visit(node.value)
        if type(value_type) is ClassNode and value_type.name in self.sdef_cls:
            if type(node.value) is ast.Name:
                return ast.Name(id=node.value.id + "_" + node.attr, ctx=ctx)
            elif type(node.value) is ast.Attribute:
                return ast.Attribute(value=node.value.value, attr=node.value.attr + "_" + node.attr, ctx=ctx)
        return node    