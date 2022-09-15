# -*- coding: utf-8 -*-

import ast, copy
import call_graph
import astunparse

class FunctionInliner(ast.NodeTransformer):
    """
    Generate new ast nodes which are used by 
    the Inliner to inline functions
    """
    def __init__(self, node, caller, args):
        self.caller = copy.deepcopy(caller)
        self.args = copy.deepcopy(args)
        self.node = copy.deepcopy(node)
        self.func_args = []
        self.var_dict = {}
        self.built_nodes = []
    
    def inline(self):
        self.visit(self.node)
        return self.built_nodes

    def visit_FunctionDef(self, node):
        if len(node.args.args)-1 != len(self.args):
            ast_error("Invalid number of arguments", node)
        for arg in node.args.args:
            if arg.arg != "self":
                self.func_args.append(arg.arg)
        for body in node.body:
            self.built_nodes.append(self.visit(body))
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

class DeviceCodeVisitor(ast.NodeTransformer):
    def __init__(self, root: call_graph.RootNode):
        self.stack = [root]

    @property
    def root(self):
        return self.stack[0]

    def visit_Module(self, node):
        for body in node.body:
            if type(body) in [ast.ClassDef, ast.FunctionDef]:
                self.visit(body)
        return node

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.stack[-1].get_ClassNode(class_name)
        if class_node.is_device:
            self.stack.append(class_node)
            for body in node.body:
                self.visit(body)
            self.stack.pop()
        return node    

"""
Replace method calls nested inside other expressions 
as new variables.

EXAMPLE:
    self.vel.add(self.force.multiply(kDt).divide(self.mass))
CONVERTED INTO:
    __auto_v0: Vector = self.force.multiply(kDt)
    __auto_v1: Vector = __auto_v0.divide(self.mass)
    self.vel.add(__auto_v1)
"""
class Normalizer(DeviceCodeVisitor):
    def __init__(self, root: call_graph.RootNode):
        super().__init__(root)
        self.var_counter = 0
        self.built_nodes = []
        self.receiver_type = call_graph.TypeNode()
        self.receiver = None

    def visit_FunctionDef(self, node):
        self.var_counter = 0
        name = node.name
        func_node = self.stack[-1].get_FunctionNode(name, self.stack[-1].name)
        if func_node is None:
            call_graph.ast_error("The function {} does not exist.".format(name), node)
        if func_node.is_device:
            self.stack.append(func_node)
            node.args = self.visit(node.args)
            new_function_body = []
            for func_body in node.body:
                transformed = self.visit(func_body)
                if type(transformed) == list:
                    new_function_body.extend(transformed)
                else:
                    new_function_body.append(transformed)
            node.body = new_function_body
            self.stack.pop()
        return node

    def visit_Expr(self, node):
        self.built_nodes = []
        node.value = self.visit(node.value)
        if self.built_nodes:
            result = []
            for new_node in self.built_nodes:
                result.append(new_node)
            result.append(ast.Expr(value=self.receiver))
            return result
        else:
            return node
        
    def visit_Assign(self, node):
        self.built_nodes = []
        self.visit(node.value)
        if self.built_nodes:
            result = []
            for new_node in self.built_nodes:
                result.append(new_node)
            result.append(ast.Assign(targets=node.targets, value=self.receiver))
            return result
        else:
            return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        self.built_nodes = []
        self.visit(node.value)
        if self.built_nodes:
            result = []
            for new_node in self.built_nodes:
                result.append(new_node)
            result.append(ast.AnnAssign(
                annotation=node.annotation, simple=node.simple, 
                target=node.target, value=self.receiver))
            return result
        else:
            return node

    def visit_Name(self, node):
        if node.id == "self":
            self.receiver_type = self.stack[-2].type
        elif self.find_VariableNode(node.id) is not None:
            self.receiver_type = self.find_VariableNode(node.id).type
        self.receiver = node
        return node

    def visit_Attribute(self, node):
        self.visit(node.value)
        if type(self.receiver) is ast.Call and type(self.receiver_type) is call_graph.ClassTypeNode:
            new_name = "__auto_v" + str(self.var_counter)
            self.stack[-1].declared_variables.add(call_graph.VariableNode(new_name, self.receiver_type))
            new_node = ast.AnnAssign(
                target=ast.Name(id=new_name, ctx=ast.Load()),
                value=self.receiver, simple=1, annotation=ast.Name(id=self.receiver_type.name, ctx=ast.Load()))
            self.receiver = ast.Attribute(
                value=ast.Name(id=new_name, ctx=node.value.ctx if hasattr(node.value, "ctx") else None),
                attr=node.attr, ctx=node.ctx)
            self.built_nodes.append(new_node)
            self.var_counter += 1
        else:
            self.receiver = ast.Attribute(value=self.receiver, attr=node.attr, ctx=node.ctx)
        class_node = self.root.get_ClassNode(self.receiver_type.name)
        if class_node is not None:
            for field in class_node.declared_fields:
                if field.name == node.attr:
                    self.receiver_type = field.type
        return node

    def visit_Subscript(self, node):
        self.receiver = node.slice.value
        self.visit(node.slice.value)
        new_index = ast.Index(self.receiver)
        if type(node.value) is ast.Name:
            var_node = self.find_VariableNode(node.value.id)
            if var_node is not None:
                self.receiver_type = var_node.element_type
            self.receiver = node
        else:
            self.visit(node.value.value)
            if type(node.value.value) is ast.Call and self.receiver_type is not None:
                new_name = "__auto_v" + str(self.var_counter)
                self.stack[-1].declared_variables.add(call_graph.VariableNode(new_name, self.receiver_type))                
                new_node = ast.AnnAssign(
                    target=ast.Name(id=new_name, ctx=ast.Load()),
                    value=self.receiver, simple=1, annotation=ast.Name(id=self.receiver_type.name, ctx=ast.Load()))
                self.receiver = ast.Subscript(
                    value=ast.Attribute(
                        value=ast.Name(id=new_name, ctx=ast.Load()),
                        attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)
                self.built_nodes.append(new_node)
            else:
                self.receiver = ast.Subscript(
                    value=ast.Attribute(value=self.receiver, attr=node.value.attr, ctx=node.value.ctx),
                    slice=new_index, ctx=node.ctx)  
            class_node = self.root.get_ClassNode(self.receiver_type.name)
            if class_node is not None:
                for field in class_node.declared_fields:
                    if field.name == node.value.attr:
                        self.receiver_type == field.type.element_type
        return node

    def visit_Call(self, node):
        new_args = []
        for arg in node.args:
            self.receiver = arg
            self.visit(arg)
            if type(arg) is ast.Call and type(self.receiver_type) is not call_graph.TypeNode:
                if hasattr(arg.func.value, "id") and arg.func.value.id == "random" and \
                        arg.func.attr in ["getrandbits", "uniform"]:
                    new_args.append(arg)
                else:
                    new_name = "__auto_v" + str(self.var_counter)
                    self.stack[-1].declared_variables.add(call_graph.VariableNode(new_name, self.receiver_type))
                    new_node = ast.AnnAssign(
                        target=ast.Name(id=new_name, ctx=ast.Load()),
                        value=self.receiver, simple=1, annotation=ast.Name(id=self.receiver_type.name, ctx=ast.Load()))
                    self.built_nodes.append(new_node)
                    new_args.append(ast.Name(id=new_name, ctx=ast.Load()))
                    self.var_counter += 1
            else:
                new_args.append(self.receiver)
        if type(node.func) == ast.Name:
            for class_func in self.stack[-2].declared_functions:
                if class_func.name == node.func.id:
                    self.receiver_type = class_func.return_type
            for class_node in self.root.declared_classes:
                if class_node.name == node.func.id:
                    self.receiver_type = class_node.type
            self.receiver = node
        elif hasattr(node.func.value, "id") and node.func.value.id == "random" and \
                node.func.attr in ["getrandbits", "uniform"]:
            self.receiver = node
        else:
            if hasattr(node.func.value, "func") and hasattr(node.func.value.func, "id") \
                    and node.func.value.func.id == "super":
                return node
            self.visit(node.func.value)
            if type(node.func.value) == ast.Call and type(self.receiver_type) is not call_graph.TypeNode:
                new_name = "__auto_v" + str(self.var_counter)
                self.stack[-1].declared_variables.add(call_graph.VariableNode(new_name, self.receiver_type))                
                new_node = ast.AnnAssign(
                    target=ast.Name(id=new_name, ctx=ast.Load()),
                    value=self.receiver, simple=1, annotation=ast.Name(id=self.receiver_type.name, ctx=ast.Load()))
                self.built_nodes.append(new_node)
                self.receiver = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=new_name, ctx=ast.Load()),
                        attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.func.value.keywords)
                self.var_counter += 1
            else:
                self.receiver = ast.Call(
                    func=ast.Attribute(value=self.receiver, attr=node.func.attr, ctx=node.func.ctx),
                    args=new_args, keywords=node.keywords)
            class_node = self.root.get_ClassNode(self.receiver_type.name)
            if class_node is not None:
                for class_func in class_node.declared_functions:
                    if class_func.name == node.func.attr:
                        self.receiver_type = class_func.return_type
        return node

    def find_VariableNode(self, var_name):
        i = len(self.stack)-1
        while i>=0:
            for var_node in self.stack[i].declared_variables:
                if var_node.name == var_name:
                    return var_node
            i -= 1
        return None

"""
Replace function calls on non-device classes with the 
specific implementations

EXAMPLE
    __auto_v0: Vector = self.force.multiply(kDt)
CONVERTED INTO
    __auto_v0: Vector = Vector((self.force.x * kDt), (self.force.y * kDt))
"""
class Inliner(DeviceCodeVisitor):
    def __init__(self, root: call_graph.RootNode):
        super().__init__(root)
    
    def visit_Module(self, node):
        self.node = node
        for node_body in node.body:
            if type(node_body) in [ast.ClassDef, ast.FunctionDef, ast.AnnAssign]:
                self.visit(node_body)
        return node

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.stack[-1].get_FunctionNode(name, self.stack[-1].name)
        if func_node is None:
            call_graph.ast_error("The function {} does not exist.".format(name), node)
        if func_node.is_device:
            self.stack.append(func_node)
            node.args = self.visit(node.args)
            new_function_body = []
            for func_body in node.body:
                transformed = self.visit(func_body)
                if type(transformed) == list:
                    new_function_body.extend(transformed)
                else:
                    new_function_body.append(transformed)
            node.body = new_function_body
            self.stack.pop()
        return node 

    def visit_Expr(self, node):
        result = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for new_node in node.value:
                if type(new_node) == ast.Return:
                    continue
                result.append(new_node)
            return result
        else:
            return node

    def visit_Assign(self, node):
        result = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for new_node in node.value:
                if type(new_node) == ast.Return:
                    new_node = ast.Assign(targets=node.targets, value=new_node.value)
                ret.append(new_node)
            return result
        return node
    
    def visit_AnnAssign(self, node):
        if node.value is None:
            return node
        if type(call_graph.ast_to_call_graph_type(self.stack, node.target)) is call_graph.RefTypeNode:
            return node
        result = []
        node.value = self.visit(node.value)
        if type(node.value) == list:
            for new_node in node.value:
                if type(new_node) is ast.Return:
                    new_node = ast.AnnAssign(
                        annotation=node.annotation, simple=node.simple,
                        target=node.target, value=new_node.value)
                result.append(new_node)
            return result
        return node
    
    def visit_Call(self, node):
        if type(node.func) is ast.Attribute:
            receiver_type = call_graph.ast_to_call_graph_type(self.stack, node.func.value)
            if type(receiver_type) is call_graph.ClassTypeNode and receiver_type.name in self.root.fields_class_names:
                for func_node in receiver_type.class_node.declared_functions:
                    if func_node.name == node.func.attr:
                        func_body_gen = FunctionInliner(func_node.ast_node, node.func.value, node.args)
                        result = func_body_gen.inline()
                        if len(result) != 0:
                            return result
        return node

"""
Replace constructor of non-reference classes with their fields

EXAMPLE:
    self.pos: Vector = Vector(0, 0)
CONVERTED INTO:
    self.pos.x: float = 0
    self.pos.y: float = 0
"""
class Eliminator(DeviceCodeVisitor):
    def __init__(self, root: call_graph.RootNode):
        super().__init__(root)
    
    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.stack[-1].get_FunctionNode(name, self.stack[-1].name)
        if func_node is None:
            call_graph.ast_error("The function {} does not exist.".format(name), node)
        if func_node.is_device:
            self.stack.append(func_node)
            node.args = self.visit(node.args)
            new_function_body = []
            for func_body in node.body:
                transformed = self.visit(func_body)
                if type(transformed) == list:
                    new_function_body.extend(transformed)
                else:
                    new_function_body.append(transformed)
            node.body = new_function_body
            self.stack.pop()
        return node

    def visit_AnnAssign(self, node):
        target_type = call_graph.ast_to_call_graph_type(self.stack, node.target)
        if type(target_type) is call_graph.ClassTypeNode and target_type.name in self.root.fields_class_names:
            if type(node.value) is ast.Call and type(node.value.func) is ast.Name:
                for func_node in target_type.class_node.declared_functions:
                    if func_node.name == "__init__":
                        inst_body_gen = FunctionInliner(func_node.ast_node, node.target, node.value.args)
                        result = inst_body_gen.inline()
                        if len(result) != 0:
                            return result
            else:
                new_nodes = []
                for field_name in target_type.class_node.expanded_fields:
                    for nested_field in target_type.expanded_fields[field_name]:
                        annotation = None
                        if type(nested_field.type) is ListTypeNode:
                            annotation = ast.Subscript(
                                value=ast.Name(id="list", ctx=node.annotation.ctx),
                                slice=ast.Index(value=ast.Name(id=nested_field.type.element_type, ctx=node.annotation.ctx)),
                                ctx=node.annotation.ctx)
                        else:
                            annotation = ast.Name(id=nested_field.type.name, ctx=node.annotation.ctx)
                        value_ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                        new_node = ast.AnnAssign(
                            target=ast.Attribute(value=node.target, attr=nested_field.name, ctx=node.target.ctx),
                            annotation=annotation,
                            value=ast.Attribute(value=node.value, attr=nested_field.name, ctx=value_ctx),
                            simple=node.simple)
                        new_nodes.append(new_node)
                return new_nodes
        return node

    def visit_Assign(self, node):
        result = []
        for target in node.targets:
            target_type = call_graph.ast_to_call_graph_type(self.stack, target)
            if type(target_type) is call_graph.ClassTypeNode and target_type.name in self.root.fields_class_names:
                if type(node.value) is ast.Call and type(node.value.func) is ast.Name:
                    for func_node in target_type.class_node.declared_functions:
                        if func_node.name == "__init__":
                            inst_body_gen = FunctionInliner(func_node.ast_node, target, node.value.args)
                            result = inst_body_gen.inline()
                            if len(result) != 0:
                                return result
                else:
                    new_nodes = []
                    for field_name in target_type.class_node.expanded_fields:
                        for nested_field in target_type.class_node.expanded_fields[field_name]:
                            value_ctx = node.value.ctx if hasattr(node.value, "ctx") else None
                            new_node = ast.Assign(
                                value=ast.Attribute(value=node.value, attr=nested_field.name, ctx=value_ctx),
                                targets=[ast.Attribute(value=target, attr=nested_field.name, ctx=target.ctx)])
                            result.append(new_node)
        if len(result) != 0:
            return result
        return node

class FieldSynthesizer(DeviceCodeVisitor):
    def __init__(self, root: call_graph.RootNode):
        super().__init__(root)

    def visit_FunctionDef(self, node):
        func_node = self.stack[-1].get_FunctionNode(node.name, self.stack[-1].name)
        if func_node is None:
            call_graph.ast_error("The function {} does not exist.".format(name), node)
        if func_node.is_device:
            self.stack.append(func_node)
            node.args = self.visit(node.args)
            node_body = []
            # Add random_state_ field
            if node.name == "__init__" and self.stack[-2].has_random_state:
                node_body.append(ast.AnnAssign(
                    target=ast.Attribute(attr="random_state_", value=ast.Name("self")),
                    annotation=ast.Attribute(attr="RandomState", value=ast.Name(id="DeviceAllocator")),
                    simple=1,
                    value=ast.Constant(value=None, kind=None)))
            for func_body in node.body:
                node_body.append(self.visit(func_body))
            node.body = node_body
            self.stack.pop()           
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if type(node.target) is ast.Attribute and self.stack[-1].name != "__init__":
            return ast.Assign(targets=[node.target], value=node.value)
        node.simple = 1
        return node

    def visit_Attribute(self, node):
        ctx = None
        if type(node.ctx) == ast.Load:
            ctx = ast.Load()
        elif type(node.ctx) == ast.Store:
            ctx = ast.Store()
        value_type = call_graph.ast_to_call_graph_type(self.stack, node.value)
        if type(value_type) is call_graph.ClassTypeNode and value_type.name in self.root.fields_class_names:
            if type(node.value) is ast.Name:
                return ast.Name(id=node.value.id + "_" + node.attr, ctx=ctx)
            elif type(node.value) is ast.Attribute:
                return ast.Attribute(value=node.value.value, attr=node.value.attr + "_" + node.attr, ctx=ctx)
        return node   