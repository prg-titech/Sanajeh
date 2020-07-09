# -*- coding: utf-8 -*-
# Generate C/C++/CUDA AST from python source code

import ast
import sys

import type_converter
from call_graph import CallGraph, ClassNode
import gencpp as cpp
import six


BOOLOP_MAP = {
    ast.And: "&&",
    ast.Or: "||",
}

# Binary operator map
OPERATOR_MAP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    # ast.Pow: "**",  # special case
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    # ast.FloorDiv: "//",  # special case
}

# Unary operator map
UNARYOP_MAP = {
    ast.Invert: "~",
    ast.Not: "!",
    ast.UAdd: "+",
    ast.USub: "-",
}

# Compare operator map
CMPOP_MAP = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",

    ast.Is: "==",  # special case
    ast.IsNot: "!=",  # special case
    # todo
    # ast.In: " in ",  # special case
    # ast.NotIn: " not in ",  # special case
}


class GenCppVisitor(ast.NodeVisitor):

    def __init__(self, root: CallGraph):
        self.__root: CallGraph = root
        self.__node_path = [self.__root]
        self.__current_node = None
        self.__classes = []
        self.__field = []

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        ret = super(GenCppVisitor, self).visit(node)
        if ret is None:
            return cpp.UnsupportedNode(node)
        return ret

    #
    # Module
    #

    def visit_Module(self, node):
        body = []
        self.__classes = []
        for x in node.body:
            # C++ doesn't support all kinds of python expressions in global scope
            if type(x) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                body.append(self.visit(x))
        return cpp.Module(body=body, classes=self.__classes)

    #
    # Statements
    #

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.__current_node.GetFunctionNode(name, None)
        if func_node is None:
            # Program shouldn't come to here, which means the function is not analyzed by the marker yet
            print("The function {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device function just skip
        if not func_node.is_device:
            return cpp.IgnoredNode(node)
        self.__node_path.append(func_node)
        args = self.visit(node.args)
        body = [self.visit(x) for x in node.body]
        docstring = ast.get_docstring(node)
        if docstring:
            body = body[1:]
        # todo return type
        returns = None
        if hasattr(node, "returns"):
            if node.returns is not None:
                returns = self.visit(node.returns)
        # TODO: decorator_list
        self.__node_path.pop()
        return cpp.FunctionDef(name=name, args=args, body=body, returns=returns)

    def visit_ClassDef(self, node):
        # todo do not support nested class
        self.__field = {}
        name = node.name
        class_node = self.__current_node.GetClassNode(name, None)
        if class_node is None:
            # Program shouldn't come to here, which means the class is not analyzed by the marker yet
            print("The class {} does not exist.".format(name), file=sys.stderr)
            sys.exit(1)
        # If it is not a device class just skip
        if not class_node.is_device:
            return cpp.IgnoredNode(node)
        self.__classes.append(name)
        self.__node_path.append(class_node)
        bases = [self.visit(x) for x in node.bases]
        keywords = None
        if six.PY3:
            keywords = [self.visit(x) for x in node.keywords]
        body = [self.visit(x) for x in node.body]
        # TODO: decorator_list
        self.__node_path.pop()
        if six.PY3:
            return cpp.ClassDef(name=name, bases=bases, keywords=keywords, body=body, fields=self.__field)
        else:
            return cpp.ClassDef(name=name, bases=bases, body=body, fields=self.__field)

    def visit_Return(self, node):
        if node.value:
            value = self.visit(node.value)
        else:
            value = None
        return cpp.Return(value=value)

    def visit_Assign(self, node):
        targets = [self.visit(x) for x in node.targets]
        value = self.visit(node.value)
        return cpp.Assign(targets, value)

    def visit_AnnAssign(self, node):
        var = node.target
        anno = node.annotation
        is_global = False
        if type(var) is ast.Name:
            var_node = self.__current_node.GetVariableNode(var.id, None, anno.id)
            if var_node is None:
                # Program shouldn't come to here, which means the variable is not analyzed by the marker yet
                print("The variable {} does not exist.".format(var.id), file=sys.stderr)
                sys.exit(1)
            if not var_node.is_device:
                return cpp.IgnoredNode(node)
        target = self.visit(node.target)
        if node.value:
            value = self.visit(node.value)
        else:
            value = None
        annotation = self.visit(node.annotation)
        if type(self.__current_node) is ClassNode:
            self.__field[var.id] = type_converter.convert(anno.id)
        if type(self.__current_node) is CallGraph:
            is_global = True
        return cpp.AnnAssign(target, value, annotation, is_global)

    def visit_AugAssign(self, node):
        assert node.op.__class__ not in [ast.Pow, ast.FloorDiv]
        target = self.visit(node.target)
        op = OPERATOR_MAP[node.op.__class__]
        value = self.visit(node.value)
        return cpp.AugAssign(target, op, value)

    def visit_For(self, node):
        target = self.visit(node.target)
        iter = self.visit(node.iter)
        body = [self.visit(x) for x in node.body]
        orelse = [self.visit(x) for x in node.orelse]
        return cpp.For(target=target, iter=iter, body=body, orelse=orelse)

    def visit_While(self, node):
        test = self.visit(node.test)
        body = [self.visit(x) for x in node.body]
        orelse = [self.visit(x) for x in node.orelse]
        return cpp.While(test=test, body=body, orelse=orelse)

    def visit_If(self, node):
        test = self.visit(node.test)
        body = [self.visit(x) for x in node.body]
        orelse = [self.visit(x) for x in node.orelse]
        return cpp.If(test=test, body=body, orelse=orelse)

    def visit_Raise(self, node):
        if six.PY3:
            exc = self.visit(node.exc) if node.exc else None
            cause = self.visit(node.cause) if node.cause else None
            return cpp.Raise(exc=exc, cause=cause)
        elif six.PY2:
            type = self.visit(node.type) if node.type else None
            inst = self.visit(node.inst) if node.inst else None
            tback = self.visit(node.tback) if node.tback else None
            return cpp.Raise(type=type, inst=inst, tback=tback)

    def visit_Expr(self, node):
        return cpp.Expr(self.visit(node.value))

    def visit_Pass(self, node):
        return cpp.Pass()

    def visit_Break(self, node):
        return cpp.Break()

    def visit_Continue(self, node):
        return cpp.Continue()

    #
    # Expressions
    #

    def visit_BoolOp(self, node):
        op = BOOLOP_MAP[node.op.__class__]
        return cpp.BoolOp(op=op, values=[self.visit(x) for x in node.values])

    def visit_BinOp(self, node):
        assert node.op.__class__ not in [ast.Pow, ast.FloorDiv]
        left = self.visit(node.left)
        op = OPERATOR_MAP[node.op.__class__]
        right = self.visit(node.right)
        return cpp.BinOp(left=left, op=op, right=right)

    def visit_UnaryOp(self, node):
        op = UNARYOP_MAP[node.op.__class__]
        operand = self.visit(node.operand)
        return cpp.UnaryOp(op=op, operand=operand)

    def visit_Lambda(self, node):
        args = self.visit(node.args)
        body = self.visit(node.body)
        return cpp.Lambda(args=args, body=body)

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return cpp.IfExp(test=test, body=body, orelse=orelse)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = [CMPOP_MAP[x.__class__] for x in node.ops]
        comparators = [self.visit(x) for x in node.comparators]
        return cpp.Compare(left=left, ops=ops, comparators=comparators)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(x) for x in node.args]
        keywords = [self.visit(x) for x in node.keywords]
        if six.PY2:
            starargs = self.visit(node.starargs) if node.starargs else None
            kwargs = self.visit(node.kwargs) if node.kwargs else None
            return cpp.Call(func, args, keywords, starargs, kwargs)
        return cpp.Call(func, args, keywords)

    def visit_Num(self, node):
        return cpp.Num(node.n)

    def visit_Str(self, node):
        return cpp.Str(s=node.s)

    def visit_NameConstant(self, node):
        """for python3 ast
        """
        value = node.value
        return cpp.NameConstant(value=value)

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return cpp.Attribute(value, node.attr)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return cpp.Subscript(value=value, slice=slice)

    def visit_Name(self, node):
        # change 'self' to 'this'
        if node.id == "self":
            node.id = "this"
        return cpp.Name(node.id)

    # def visit_Tuple(self, node):
    #     elts = [self.visit(x) for x in node.elts]
    #     return cpp.Tuple(elts=elts)

    # slice

    def visit_Index(self, node):
        value = self.visit(node.value)
        return cpp.Index(value=value)

    def visit_arguments(self, node):
        args = [self.visit(x) for x in node.args]
        vararg = node.vararg
        kwarg = node.kwarg
        defaults = [self.visit(x) for x in node.defaults]
        ret = cpp.arguments(args=args, vararg=vararg, kwarg=kwarg, defaults=defaults)
        return ret

    def visit_arg(self, node):
        """for python3 ast
        """
        arg = node.arg
        annotation = None
        if hasattr(node, "annotation"):
            if node.annotation is not None:
                annotation = self.visit(node.annotation)
        else:
            annotation = None
        # TODO: node.lineno
        # TODO: node.col_offset
        return cpp.arg(arg=arg, annotation=annotation)

    def visit_keyword(self, node):
        if six.PY3:
            name = node.arg
        else:
            name = node.name
        value = self.visit(node.value)
        return cpp.keyword(name=name, value=value)


