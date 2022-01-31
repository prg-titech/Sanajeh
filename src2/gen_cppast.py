# -*- coding: utf-8 -*-
# Generate C/C++/CUDA AST from python source code

import ast
import sys

import type_converter
from call_graph import CallGraph, ClassNode, FunctionNode
import build_cpp as cpp
import six

import astunparse

def pprint(node):
    print(astunparse.unparse(node))

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
    ast.FloorDiv: "/",  # special case
    ast.Mod: "%"
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


class GenCppAstVisitor(ast.NodeVisitor):

    def __init__(self, root: CallGraph):
        self.__root: CallGraph = root
        self.__node_path = [self.__root]
        self.__classes = []
        self.__field = []

    def visit(self, node):
        ret = super(GenCppAstVisitor, self).visit(node)
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

        # add parent ref
        classes = [ x for x in body if type(x) is cpp.ClassDef ]
        for cls in classes:
            cls.CheckParent( classes )

        return cpp.Module(body=body, classes=self.__classes)

    #
    # Statements
    #

    def visit_FunctionDef(self, node):
        name = node.name
        func_node = self.__node_path[-1].GetFunctionNode(name, self.__node_path[-1].name)
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
        class_node = self.__node_path[-1].GetClassNode(name)
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
        ann = None
        if type(node.annotation) is ast.Subscript:
            ann = node.annotation.value.id
        else:
            if type(node.annotation) is ast.Attribute:
                ann = node.annotation.attr
            else:
                ann = node.annotation.id
        is_global = False
        if type(var) is ast.Name:
            var_node = self.__node_path[-1].GetVariableNode(var.id, ann)
            if var_node is None:
                # Program shouldn't come to here, which means the variable is not analyzed by the marker yet
                print("The variable {} does not exist.".format(var.id), file=sys.stderr)
                sys.exit(1)
            if not var_node.is_device:
                return cpp.IgnoredNode(node)
        target = self.visit(node.target)
        array_size = None
        if node.value:
            value = self.visit(node.value)
            if type(value) == cpp.ListInitialization and hasattr(value.size, "n"):
                array_size = value.size.n
        else:
            value = None
        annotation = self.visit(node.annotation)
        if self.__node_path[-1].name == "__init__" and hasattr(node.target.value, "id") \
        and node.target.value.id == "self":
            ann_type = type_converter.convert_ann(node.annotation)
            # Field with list type needs a fixed size
            if ann_type == "list":
                self.__field[node.target.attr] = \
                    "DeviceArray<{}, {}>".format(type_converter.convert_ann(node.annotation.slice.value), array_size)
            else:
                self.__field[node.target.attr] = ann_type
        if type(self.__node_path[-1]) is CallGraph:
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
        value = self.visit(node.value)
        if type(value) is cpp.InitializerList:
            return value
        return cpp.Expr(value)

    def visit_Pass(self, node):
        return cpp.Pass()

    def visit_Break(self, node):
        return cpp.Break()

    def visit_Continue(self, node):
        return cpp.Continue()

    def visit_Assert(self, node):
        return cpp.Assert(self.visit(node.test))
    
    #
    # Expressions
    #

    def visit_BoolOp(self, node):
        op = BOOLOP_MAP[node.op.__class__]
        return cpp.BoolOp(op=op, values=[self.visit(x) for x in node.values])

    def visit_BinOp(self, node):
        assert node.op.__class__ not in [ast.Pow]
        left = self.visit(node.left)
        op = OPERATOR_MAP[node.op.__class__]
        right = self.visit(node.right)
        # For list initialization in the form of [None] * n, the size must be determined statically
        if type(left) == cpp.List and len(left.elts) == 1 and type(left.elts[0]) == cpp.NameConstant \
        and left.elts[0].value is None and op == "*" and type(right) == cpp.Num:
            return cpp.ListInitialization(size=right)
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
        if type(self.__node_path[-1]) == FunctionNode:
            """
            if hasattr(node.func, "attr") and node.func.attr == "__init__" and hasattr(node.func.value, "func") \
            and hasattr(node.func.value.func, "id") and node.func.value.func.id == "super":
                args = [self.visit(x) for x in node.args]
                return cpp.InitializerList(self.__node_path[-2].super_class, args)
            """
            # # case: super()._(args) in the method that initializes the object
            # # print("debug <{}>".format(self.__node_path[-1].name))
            # if hasattr(node.func, "attr") and node.func.attr == self.__node_path[-2].super_class \
            # and self.__node_path[-1].name == self.__node_path[-2].name and hasattr(node.func.value, "func") \
            # and hasattr(node.func.value.func, "id") and node.func.value.func.id == "super":
            #     args = [self.visit(x) for x in node.args]
            #     return cpp.InitializerList(self.__node_path[-2].super_class, args)

            # if inside constructor
            if self.__node_path[-1].name == self.__node_path[-2].name \
            and hasattr(node.func,"value") and hasattr(node.func.value,"func") and node.func.value.func.id=="super":
                print("super call",node.func.attr)
                args = [self.visit(x) for x in node.args]
                return cpp.InitializerList(node.func.attr, args)

        # EXTENSION: Type Cast
        if type(node.func)==ast.Name and node.func.id in [x.name for x in self.__root.declared_classes]:
            print("maybe cast",node.func.id)
            args = [self.visit(x) for x in node.args]
            keywords = [self.visit(x) for x in node.keywords]

            return cpp.Cast(node.func.id,args,keywords)


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
        nid = node.id
        # change 'self' to 'this'
        if nid == "self":
            nid = "this"
        return cpp.Name(nid)

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

    def visit_List(self, node):
        elements = []
        for elt in node.elts:
            elements.append(self.visit(elt))
        return cpp.List(elements)

