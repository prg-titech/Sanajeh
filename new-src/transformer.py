# -*- coding: utf-8 -*-

import ast, astunparse
import copy
import call_graph

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
                ast_error("Invalid number of arguments", node)
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
                    new_args.append(ast.Name(id="__auto_v" + str(self.var_counter), ctx=ast.Load()))
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
            if type(receiver_type) is call_graph.ClassTypeNode and receiver_type.name in self.root.device_class_names:
                func_body_gen = FunctionBodyGenerator(self.node, receiver_type.name)
                func_body_gen.GetTransformedNodes(node.func.value, node.func.attr, node.args)
                if len(func_body_gen.new_ast_nodes) != 0:
                    return func_body_gen.new_ast_nodes
        return node                