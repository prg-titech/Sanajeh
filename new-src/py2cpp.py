# -*- coding: utf-8 -*-

import ast, os, sys
import hashlib
import six

import call_graph as cg

import astunparse

INDENT: str = "\t"

BOOLOP_MAP = {
    ast.And: "&&", ast.Or: "||"
}

OPERATOR_MAP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|", 
    ast.BitXor: "^", ast.BitAnd: "&", ast.FloorDiv: "/", ast.Mod: "%"
}

UNARYOP_MAP = {
    ast.Invert: "~", ast.Not: "!",
    ast.UAdd: "+", ast.USub: "-",
}


CMPOP_MAP = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
    ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Is: "==", ast.IsNot: "!="
}

class ParallelNewBuilder:
    def __init__(self, class_name):
        self.class_name = class_name
    
    def buildCpp(self):
        parallel_new_expr = INDENT + "allocator_handle->parallel_new<{}>(object_num);\n".format(self.class_name)
        return_expr = INDENT + "return 0;"
        return 'extern "C" int parallel_new_{}(int object_num){{\n'.format(self.class_name) \
                + parallel_new_expr \
                + return_expr \
                + "\n}"

    def buildHpp(self):
        return 'extern "C" int parallel_new_{}(int object_num);'.format(self.class_name)

    def buildCdef(self):
        return 'int parallel_new_{}(int object_num);'.format(self.class_name) 

class DoAllBuilder():
    def __init__(self, class_node):
        self.class_node = class_node

    def buildCpp(self):
        fields_str = ""
        field_types_str = ""
        for i in range(len(self.class_node.declared_fields)):
            field = self.class_node.declared_fields[i]
            field_type = field.type
            if type(field_type) not in [cg.IntNode, cg.FloatNode, cg.BoolNode]:
                # fields_str += "(int) this->{}, ".format(field)                    
                fields_str += "0".format(field.name)
                field_types_str += "{}".format("int")
            else:
                fields_str += "this->{}".format(field.name)
                field_types_str += "{}".format(field_type.name)
            if i != len(self.class_node.declared_fields) - 1:
                fields_str += ", "
                field_types_str += ", "
        func_exprs = ['\n' +
                        'void {}::_do(void (*pf)({})){{\n'.format(self.class_node.name, field_types_str) +
                        INDENT +
                        'pf({});\n'.format(fields_str) +
                        '}',
                        '\n' +
                        'extern "C" int {}_do_all(void (*pf)({})){{\n'.format(self.class_node.name, field_types_str) +
                        INDENT +
                        'allocator_handle->template device_do<{}>(&{}::_do, pf);\n '.format(self.class_node.name,
                                                                                            self.class_node.name) +
                        INDENT + 'return 0;\n' +
                        '}']
        return "\n".join(func_exprs)
        
    def buildHpp(self):
        return 'extern "C" {}\n'.format(self.buildCdef())
    
    def buildCdef(self):
        field_types_str = ""
        for i in range(len(self.class_node.declared_fields)):
            field = self.class_node.declared_fields[i]
            field_type = field.type
            if type(field_type) not in [cg.IntNode, cg.FloatNode, cg.BoolNode]:
                field_type = cg.IntNode()
            if i != len(self.class_node.declared_fields) - 1:
                field_types_str += "{}, ".format(field_type.name)
            else:
                field_types_str += "{}".format(field_type.name)
        return 'int {}_do_all(void (*pf)({}));'.format(self.class_node.name, field_types_str)

class ParallelDoBuilder():

    def __init__(self, func_node):
        self.func_node = func_node
        self.args = []
        for arg in func_node.arguments:
            if arg.name != "self":
                self.args.append(arg)

    def buildCpp(self):
        arg_strs = []
        args = []
        for arg in self.args:
            args.append(arg.name)
            arg_strs.append("{} {}".format(arg.type.name, arg.name))
        parallel_do_expr = INDENT + "allocator_handle->parallel_do<{}, &{}::{}>({});\n".format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ", ".join(args)
        )
        return_expr = INDENT + "return 0;"
        return 'extern "C" int {}_{}_{}({}){{\n'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ", ".join(arg_strs)) \
                + parallel_do_expr \
                + return_expr \
                + "\n}"
    
    def buildHpp(self):
        arg_strs = []
        for arg in self.args:
            arg_strs.append("{} {}".format(arg.type.name, arg.name))
        return 'extern "C" int {}_{}_{}({});'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ",".join(arg_strs)
        )
    
    def buildCdef(self):
        arg_strs = []
        for arg in self.args:
            args_strs.append("{} {}".format(arg.type.name. arg.name))
        return 'int {}_{}_{}({});'.format(
            self.func_node.host_name,
            self.func_node.host_name,
            self.func_node.name,
            ",".join(arg_strs)
        )

class Preprocessor(ast.NodeVisitor):
    
    def __init__(self, root: cg.RootNode):
        self.stack = [root]
        self.ast_root = None
        self.classes = []
        self.cpp_parallel_new_codes = []
        self.hpp_parallel_new_codes = []
        self.cdef_parallel_new_codes = []
        self.cpp_do_all_codes = []
        self.hpp_do_all_codes = []
        self.cdef_do_all_codes = []
        self.parallel_do_hashtable = []
        self.cpp_parallel_do_codes = []
        self.hpp_parallel_do_codes = []
        self.cdef_parallel_do_codes = []
        self.global_device_variables = {}

    @property
    def root(self):
        return self.stack[0]

    def __gen_Hash(self, lst):
        """
        Helper function, generate same hash value for tuple with same strings
            Used to prevent generate mutiple code for a same parallel_do function
        """
        m = hashlib.md5()
        for elem in lst:
            m.update(elem.encode('utf-8'))
        return m.hexdigest()

    def build_parallel_do_cpp(self):
        return '\n\n' + '\n\n'.join(self.cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n'.join(self.hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.cdef_parallel_do_codes)

    def build_do_all_cpp(self):
        return '\n\n'.join(self.cpp_do_all_codes)

    def build_do_all_hpp(self):
        return '\n\n' + '\n'.join(self.hpp_do_all_codes)

    def build_do_all_cdef(self):
        return '\n' + '\n'.join(self.cdef_do_all_codes)

    def build_parallel_new_cpp(self):
        return '\n\n' + '\n\n'.join(self.cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.cdef_parallel_new_codes)

    def build_global_device_variables_init(self):
        result = []
        for var in self.global_device_variables:
            var_node = self.root.get_VariableNode(var, None)
            elem_type = var_node.type.element_type
            n = self.global_device_variables[var]
            ret.append(INDENT + "{}* host_{};\n".format(elem_type.to_cpp_type(), var) + \
                       INDENT + "cudaMalloc(&host_{}, sizeof({})*{});\n".format(var, elem_type.to_cpp_type, n) + \
                       INDENT + "cudaMemcpyToSymbol({}, &host_{}, sizeof({}*), 0, cudaMemcpyHostToDevice);\n" \
                       .format(var, var, elem_type.to_cpp_type()))
        return "\n".join(result)

    def build_global_device_variables_unit(self):
        result = []
        for var in self.global_device_variables:
            ret.append(INDENT + "cudaFree(host_{});\n".format(var))
        return "\n".join(result) 

    def visit_Module(self, node):
        self.ast_root = node
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.stack[-1].get_ClassNode(class_name)
        self.stack.append(class_node)
        self.generic_visit(node)
        self.stack.pop()
    
    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.stack[-1].get_FunctionNode(func_name, self.stack[-1].name)
        self.stack.append(func_node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node):
        # Find device classes through device code
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name:
            if node.func.value.id == "DeviceAllocator":
                if node.func.attr == "device_class":
                    for class_arg in node.args:
                        class_name = class_arg.id
                        if class_name not in self.classes:
                            self.classes.append(class_name)
                            pnb = ParallelNewBuilder(class_name)
                            self.cpp_parallel_new_codes.append(pnb.buildCpp())
                            self.hpp_parallel_new_codes.append(pnb.buildHpp())
                            self.cdef_parallel_new_codes.append(pnb.buildCdef())
                            class_node = self.root.get_ClassNode(class_name)
                            dab = DoAllBuilder(class_node)
                            self.cpp_do_all_codes.append(dab.buildCpp())
                            self.hpp_do_all_codes.append(dab.buildHpp())
                            self.cdef_do_all_codes.append(dab.buildCdef())
                elif node.func.attr == "parallel_do":
                    hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                    if hval not in self.parallel_do_hashtable:
                        self.parallel_do_hashtable.append(hval)
                        func_node = self.root.get_FunctionNode(node.args[1].attr, node.args[1].value.id)
                        pdb = ParallelDoBuilder(func_node)
                        self.cpp_parallel_do_codes.append(pdb.buildCpp())
                        self.hpp_parallel_do_codes.append(pdb.buildHpp())
                        self.cdef_parallel_do_codes.append(pdb.buildCdef())
                elif node.func.attr == "array_size":
                    self.global_device_variables[str(node.args[0].id)] = node.args[1].n
            # Find device classes through host code
            elif node.func.value.id == "allocator" or node.func.value.id == "PyAllocator":
                if node.func.attr == "parallel_new":
                    class_name = node.args[0].id
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                        pnb = ParallelNewBuilder(class_name)
                        self.cpp_parallel_new_codes.append(pnb.buildCpp())
                        self.hpp_parallel_new_codes.append(pnb.buildHpp())
                        self.cdef_parallel_new_codes.append(pnb.buildCdef())
                        class_node = self.root.get_ClassNode(class_name)
                        dab = DoAllBuilder(class_node)
                        self.cpp_do_all_codes.append(dab.buildCpp())
                        self.hpp_do_all_codes.append(dab.buildHpp())
                        self.cdef_do_all_codes.append(dab.buildCdef())
                elif node.func.attr == "parallel_do":
                    hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                    if hval not in self.parallel_do_hashtable:
                        self.parallel_do_hashtable.append(hval)
                        func_node = self.root.get_FunctionNode(node.args[1].attr, node.args[1].value.id)
                        pdb = ParallelDoBuilder(func_node)
                        self.cpp_parallel_do_codes.append(pdb.buildCpp())
                        self.hpp_parallel_do_codes.append(pdb.buildHpp())
                        self.cdef_parallel_do_codes.append(pdb.buildCdef())
        self.generic_visit(node)
    
class CppVisitor(ast.NodeVisitor):
    super_call = None

    def __init__(self, root: cg.RootNode):
        self.stack = [root]
        self.indent_level = 0

    @property
    def root(self):
        return self.stack[0]

    def indent(self, dec=0):
        return (self.indent_level-dec)*INDENT
        
    def return_type(self, func_ast):
        if func_ast.returns is not None:
            return cg.ast_to_call_graph_type(self.stack, func_ast.returns).to_cpp_type()
        else:
            return "void"

    def visit_Module(self, node):
        body = []
        for mod_body in node.body:
            if type(mod_body) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                body.append(self.visit(mod_body))
        while "" in body:
            body.remove("")
        return "\n".join(body)        

    def visit_ClassDef(self, node):
        class_node = self.stack[-1].get_ClassNode(node.name)
        if not class_node.is_device:
            return ""
        self.stack.append(class_node)
        body = [self.visit(class_body) for class_body in node.body]
        self.stack.pop()
        while "" in body:
            body.remove("")
        return "\n".join(body)

    def visit_FunctionDef(self, node):
        """ self.super_call = None """
        cpp_code = ""
        func_node = self.stack[-1].get_FunctionNode(node.name, self.stack[-1].name)
        if not func_node.is_device or node.name == "__init__":
            return ""
        self.stack.append(func_node)
        self.indent_level += 1
        # TODO: Initializer list
        if type(self.stack[-2]) is cg.ClassNode:
            body = [self.visit(func_body) for func_body in node.body]
            args = self.visit(node.args)
            if ast.get_docstring(node) is not None:
                body = body[1:]
            while "" in body:
                body.remove("")
            if node.name == self.stack[-2].name:
                # Constructor
                """
                if self.super_call:
                    cpp_code = "\n".join([
                        "\n{}__device__ {}::{}({}) : {}{{".format(
                            self.indent(1),
                            self.stack[-1].name,
                            self.stack[-1].name,
                            args,
                            self.super_call),
                        "\n".join(body),
                        self.indent(1) + "}"
                    ])
                else:
                    cpp_code = "\n".join([
                        "\n{}__device__ {}::{}({}) {{".format(
                            self.indent(1),
                            self.stack[-1].name,
                            self.stack[-1].name,
                            args,
                        ),
                        "\n".join(body),
                        self.indent(1) + "}",
                    ])
                """
                cpp_code = "\n".join([
                    "\n{}__device__ {}::{}({}) {{".format(
                        self.indent(1),
                        self.stack[-1].name,
                        self.stack[-1].name,
                        args,
                    ),
                    "\n".join(body),
                    self.indent(1) + "}"
                ])
                arg_count = len(node.args.args) - 1
                if arg_count > 0:
                    cpp_code += "\n"
                    cpp_code += "\n".join([
                        "\n{}__device__ void {}::{}__init({}) {{".format(
                            self.indent(1),
                            self.stack[-1].name,
                            self.stack[-1].name,
                            args,
                        ),
                        "\n".join(body),
                        self.indent(1) + "}"
                    ])
            else:
                # Methods
                rtype = self.return_type(node)
                cpp_code = "\n".join([
                    "\n{}__device__ {} {}::{}({}) {{".format(
                        self.indent(1),
                        rtype,
                        self.stack[-2].name,
                        node.name,
                        args,
                    ),
                    "\n".join(body),
                    self.indent(1) + "}",
                ])
        else:
            body = [self.visit(func_body) for func_body in node.body]
            args = self.visit(node.args)
            rtype = self.return_type(node)
            cpp_code = "\n".join([
                "\n{}__device__ {} {}({}) {{".format(
                    self.indent(1),
                    rtype,
                    node.name,
                    args,
                ),
                "\n".join(body),
                self.indent(1) + "}",
            ])
        self.indent_level -= 1
        self.stack.pop()
        return cpp_code
    
    def visit_Assign(self, node):
        return self.indent() + "{} = {};".format(
            " = ".join([self.visit(target) for target in node.targets]),
            self.visit(node.value))
    
    def visit_AnnAssign(self, node):
        cpp_code = ""
        node_type = cg.ast_to_call_graph_type(self.stack, node.annotation)
        # ignore non-device variables
        if type(node.target) is ast.Name:
            var_node = self.stack[-1].get_VariableNode(node.target.id)
            if not var_node.is_device:
                return ""
        if type(self.stack[-1]) is cg.RootNode:
            if type(node_type) is cg.ListTypeNode and type(node.value) is ast.Call \
                    and type(node.value.func.value) is ast.Name \
                    and node.value.func.value.id == "DeviceAllocator" \
                    and node.value.func.attr == "array":
                        return self.indent() + "__device__ {}* {}[{}];".format(
                            self.visit(node.annotation.slice),
                            self.visit(node.target),
                            ", ".join([self.visit(arg) for arg in node.value.args]))
        elif type(self.stack[-1]) is cg.FunctionNode and node.value is not None:
            if type(self.stack[-1]) is cg.FunctionNode and self.stack[-1].name == "__init__":
                if not type(node_type) is cg.ListTypeNode:
                    return self.indent() + "{} = {};".format(
                        self.visit(node.target),
                        self.visit(node.value)
                    ) 
            else:
                if type(node_type) is cg.ListTypeNode and type(node.value) is ast.BinOp \
                        and self.is_none(node.value.left):
                    return self.indent() + "{} {}[{}];".format(
                        node_type.element_type.to_cpp_type(),
                        self.visit(node.target),
                        self.visit(node.value.right)
                    )
                return self.indent() + "{} {} = {};".format(
                    cg.ast_to_call_graph_type(self.stack, node.annotation).to_cpp_type(),
                    self.visit(node.target),
                    self.visit(node.value)
                )
        return cpp_code

    def visit_AugAssign(self, node):
        assert node.op.__class__ not in [ast.Pow, ast.FloorDiv]
        return self.indent() + "{} {}= {};".format(
            self.visit(node.target),
            OPERATOR_MAP[node.op.__class__],
            self.visit(node.value))

    def visit_Subscript(self, node):
        return "{}[{}]".format(
            self.visit(node.value),
            self.visit(node.slice))
    
    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_If(self, node):
        self.indent_level += 1
        body = [self.visit(if_body) for if_body in node.body]
        self.indent_level -= 1
        result = [
            "{}if ({}) {{".format(
                self.indent(),
                self.visit(node.test)
            ),
            "\n".join(body),
            self.indent() + "}",
        ]
        if len(node.orelse) == 1 and node.orelse[0].__class__ == ast.If:
            lines = self.visit(node.orelse[0]).split("\n")
            assert len(lines) > 1
            result[-1] = "{}}} else {}".format(self.indent(), lines[0])
            result.extend(lines[1:])
        elif node.orelse:
            result[-1] = self.indent() + "} else {"
            self.indent_level += 1
            orelse = [self.visit(orelse_body) for orelse_body in node.orelse]
            self.indent_level -= 1
            built = "\n".join(orelse)
            result.append(built)
            result.append(self.indent() + "}")
        return "\n".join(result)

    def visit_For(self, node):
        self.indent_level += 1
        body = [self.visit(for_body) for for_body in node.body]
        self.indent_level -= 1
        target = self.visit(node.target)
        if type(node.iter) is ast.Call:
            if type(node.iter.func) is ast.Name and node.iter.func.id == "range":
                # for _ in range(_)
                if len(node.iter.args) == 1:
                    return "\n".join([
                        "{}for (int {} = {}; {} < {}; ++{}) {{".format(
                            self.indent(),
                            target,
                            "0",
                            target,
                            self.visit(node.iter.args[0]),
                            target),
                        "\n".join(body),
                        self.indent() + "}"])
                # for _ in range(_,_)
                elif len(node.iter.args) == 2:
                    return "\n".join([
                        "{}for (int {} = {}; {} < {}; ++{}) {{".format(
                            self.indent(),
                            target,
                            self.visit(node.iter.args[0]),
                            target,
                            self.visit(node.iter.args[1]),
                            target),
                        "\n".join(body),
                        self.indent() + "}"])
                # for _ in range(_,_,_)
                elif len(node.iter.args) == 3:
                    return "\n".join([
                        "{}for (int {} = {}; {} < {}; {} += {}) {{".format(
                            self.indent(),
                            target,
                            self.visit(node.iter.args[0]),
                            target,
                            self.visit(node.iter.args[1]),
                            target,
                            self.visit(node.iter.args[2])),
                        "\n".join(body),
                        self.indent() + "}"])
                else:
                    # TODO: pass error
                    pass
        
        return "\n".join([
            "{}for (auto {} : {}) {{".format(
                self.indent(),
                target,
                self.visit(iter)),
            "\n".join(body),
            self.indent() + "}"])

    def visit_While(self, node):
        self.indent_level += 1
        body = [self.visit(while_body) for while_body in node.body]
        self.indent_level -= 1
        return "\n".join([
            "{}while ({}) {{".format(
                self.indent(),
                self.visit(node.test)),
            "\n".join(body),
            self.indent() + "}"])

    def visit_Expr(self, node):
        value = self.visit(node.value)
        if value == "":
            return ""
        return self.indent() + "{};".format(value)

    def visit_BoolOp(self, node):
        values = []
        for value in node.values:
            if type(value) is ast.BoolOp:
                values.append("({})".format(self.visit(value)))
            else:
                values.append(self.visit(value))
        return " {} ".format(BOOLOP_MAP[node.op.__class__]).join(values)


    def visit_BinOp(self, node):
        if (type(node.left)) == ast.BinOp:
            left_exp = "({})".format(self.visit(node.left))
        else:
            left_exp = self.visit(node.left)
        if (type(node.right)) == ast.BinOp:
            right_exp = "({})".format(self.visit(node.right))
        else:
            right_exp = self.visit(node.right)
        return " ".join([left_exp, OPERATOR_MAP[node.op.__class__], right_exp])

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if type(operand) is ast.BoolOp:
            operand = "({})".format(operand)
        return "{}{}".format(UNARYOP_MAP[node.op.__class__], operand)

    def visit_Num(self, node):
        return "{}".format(node.n)

    def visit_Attribute(self, node):
        if type(node.value) is ast.Name:
            if node.value.id == "math":
                return node.attr
            elif node.value.id == "DeviceAllocator" and node.attr == "RandomState":
                return "curandState"
        return "{}->{}".format(self.visit(node.value), node.attr)

    def visit_Name(self, node):
        cpp_code = node.id
        if cpp_code == "self":
            cpp_code = "this"
        return cpp_code

    def visit_NameConstant(self, node):
        if type(node.value) == bool:
            return "true" if node.value else "false"
        elif node.value is None:
            return "nullptr"
        return node.value

    def visit_Compare(self, node):
        result = [self.visit(node.left)]
        if type(node.left) is ast.Call and type(node.left.func) is ast.Name \
                and node.left.func.id == "type" and len(node.ops) == 1 \
                and node.ops[0].__class__ is ast.Eq and len(node.comparators) == 1:
            return "{}->cast<{}>() != nullptr".format(
                self.visit(node.left.args[0]),
                self.visit(node.comparators[0]))
        for op, comp in zip(node.ops, node.comparators):
            result += [CMPOP_MAP[op.__class__], self.visit(comp)]
        return " ".join(result)

    def visit_Call(self, node):
        if type(self.stack[-1]) is cg.FunctionNode:
            # case: super()._(args) in the method that initializes the object
            """
            if type(node.func) is ast.Attribute and node.func.attr == self.stack[-2].super_class \
                    and self.stack[-1].name == self.stack[-2].name \
                    and type(node.func.value) is ast.Call \
                    and type(node.func.value.func) is ast.Name and node.func.value.func.id == "super":
                args = [self.visit(arg) for arg in node.args]
                self.super_call = "{}({}) ".format(self.stack[-2].super_class, ", ".join(args))
                return ""
            """
            if type(node.func) is ast.Attribute and self.stack[-1].name == self.stack[-2].name \
                    and type(node.func.value) is ast.Call \
                    and type(node.func.value.func) is ast.Name and node.func.value.func.id == "super":
                args = [self.visit(arg) for arg in node.args]
                return "this->{}::{}__init({})".format(node.func.attr, node.func.attr, ", ".join(args))
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name:
            if node.func.value.id == "DeviceAllocator":
                if node.func.attr == "device_do":
                    return "device_allocator->template device_do<{}>(&{}::{}, {})".format(
                        self.visit(node.args[0]),
                        self.visit(node.args[1].value),
                        node.args[1].attr,
                        ", ".join([self.visit(arg) for arg in node.args[2:]]))
                elif node.func.attr == "new":
                    args = ", ".join([self.visit(arg) for arg in node.args[1:]])
                    return "new(device_allocator) {}({})".format(self.visit(node.args[0]), args)
                elif node.func.attr == "destroy":
                    return "destroy(device_allocator, {})".format(self.visit(node.args[0]))
                elif node.func.attr == "virtual":
                    # TODO: get children name
                    func_name = node.args[0].attr
                    children = ["Fish", "Shark"]
                    result = ""
                    for child in children:
                        result += "this->cast<{}>() != nullptr ? this->cast<{}>()->{}() : " \
                            .format(child, child, func_name)
                    result += "nullptr"
                    return result
                else:
                    # No API provided by Sanajeh
                    print("{} is not provided by Sanajeh".format(node.func.attr))
                    assert False
            if node.func.value.id == "random":
                if node.func.attr == "getrandbits":
                    return "curand(&random_state_)"
                elif node.func.attr == "uniform":
                    return "curand_uniform(&random_state_)"
                # Set a random state field for the class
                elif node.func.attr == "seed":
                    args = ", ".join([self.visit(arg) for arg in node.args])
                    return "curand_init(kSeed, {}, 0, &random_state_)".format(args)
                else:
                    # No API provided by Sanajeh
                    assert False
        if type(node.func) is ast.Name:
            if node.func.id == "cast":
                if len(node.args) != 2:
                    assert False
                cast_target = self.visit(node.args[0])
                cast_value = self.visit(node.args[1])
                return "{}->cast<{}>()".format(cast_value, cast_target)
        args = ", ".join([self.visit(arg) for arg in node.args])
        return "{}({})".format(self.visit(node.func), args)

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return "(({}) ? ({}) : ({}))".format(test, body, orelse)

    def visit_arguments(self, node):
        args = []
        for arg in node.args:
            converted_arg = self.visit(arg)
            if len(self.stack) > 2 and type(self.stack[-1]) is cg.FunctionNode \
                    and type(self.stack[-2]) is cg.ClassNode and converted_arg == "self":
                continue
            arg_type = cg.ast_to_call_graph_type(self.stack, arg.annotation).to_cpp_type()
            args.append("{} {}".format(arg_type, converted_arg))
        return ", ".join(args)
    
    def visit_arg(self, node):
        return node.arg
    
    def visit_Assert(self, node):
        return self.indent() + "assert({})".format(self.visit(node.test)) + ";"

    def visit_Pass(self, node):
        return ""

    def visit_Return(self, node):
        if node.value:
            return self.indent() + "return {};".format(self.visit(node.value))
        else:
            return self.indent() + "return;"

    def is_none(self, node):
        return type(node) is ast.List and type(node.elts[0]) is ast.NameConstant \
            and node.elts[0].value is None

class HppVisitor(ast.NodeVisitor):
    def __init__(self, root: cg.RootNode):
        self.stack = [root]
        self.indent_level = 0

    @property
    def root(self):
        return self.stack[0]

    def indent(self, dec=0):
        return (self.indent_level-dec)*INDENT

    def do_all_convert(self, type_str):
        return type_str if type_str in ["int", "bool", "float"] else "int"

    def return_type(self, func_ast):
        if func_ast.returns is not None:
            return cg.ast_to_call_graph_type(self.stack, func_ast.returns).to_cpp_type()
        else:
            return "void"

    def visit_Module(self, node):
        body = []
        for mod_body in node.body:
            if type(mod_body) in [ast.FunctionDef, ast.ClassDef, ast.AnnAssign]:
                body.append(self.visit(mod_body))
        while "" in body:
            body.remove("")
        class_str = ", ".join(self.root.device_class_names)
        class_predefine = "\n\n"
        for class_name in self.root.device_class_names:
            class_predefine += "class " + class_name + ";\n"
        allocator = "\nusing AllocatorT = SoaAllocator<" + "KNUMOBJECTS" + ", " + class_str + ">;\n\n"
        return class_predefine + allocator + "\n".join(body)

    def visit_ClassDef(self, node):
        class_node = self.stack[-1].get_ClassNode(node.name)
        if not class_node.is_device:
            return ""
        self.stack.append(class_node)
        body = [self.visit(class_body) for class_body in node.body]
        self.stack.pop()
        field_types = []
        do_field_types = []
        field_templates = []
        for i, field in enumerate(class_node.declared_fields):
            field_types.append(field.type.to_field_type())
            do_field_types.append(self.do_all_convert(field.type.name))
            field_templates.append(INDENT + "Field<{}, {}> {};".format(
                node.name, i, field.name))
        field_predeclaration = ""
        base = ""
        if len(node.bases) > 0:
            base = "\n" + self.indent() + INDENT + "using BaseClass = {};\n\n".format(self.visit(node.bases[0]))
        if len(field_types) == 0:
            field_predeclaration = self.indent() + "public:\n" \
                                    + self.indent() \
                                    + INDENT \
                                    + "declare_field_types({})\n".format(node.name) \
                                    + base
        else:
            field_predeclaration = self.indent() + "public:\n" \
                                    + self.indent() \
                                    + INDENT \
                                    + "declare_field_types({}, {})\n".format(node.name, ", ".join(field_types)) \
                                    + base \
                                    + self.indent() \
                                    + ("\n" + self.indent()).join(field_templates)
        _do_function = self.indent() \
                        + INDENT \
                        + "void _do(void (*pf)({}));".format(", ".join(do_field_types))
        body.append("__device__ {}() {{}};".format(node.name))
        body = [INDENT + built for built in body]
        return "\n".join([
            "\n{}class {}{} \n{{".format(
                self.indent(),
                node.name,
                " : {}".format(", ".join(["public " + self.visit(base) for base in node.bases])) if node.bases
                else " : public AllocatorT::Base",
            ),
            field_predeclaration,
            ("\n").join(body),
            _do_function,
            self.indent() + "};"
        ])

    def visit_FunctionDef(self, node):
        func_node = self.stack[-1].get_FunctionNode(node.name, self.stack[-1].name)
        if not func_node.is_device:
            return ""
        self.stack.append(func_node)
        self.indent_level += 1
        args = self.visit(node.args)
        self.indent_level -= 1
        self.stack.pop()
        if func_node.name == "__init__" and type(self.stack[-1]) is cg.ClassNode:
            return ""
        if func_node.name == self.stack[-1].name and type(self.stack[-1]) is cg.ClassNode:
            """
            return "\n".join([
                "{}__device__ {}({});".format(
                    self.indent(),
                    self.stack[-1].name,
                    args,
                )
            ])
            """
            result = ["{}__device__ {}({});".format(
                self.indent(),
                self.stack[-1].name,
                args,
            )]
            arg_count = len(node.args.args) - 1
            if arg_count > 0:
                # TODO: I don't like this INDENT
                result.append(INDENT + "{}__device__ void {}__init({});".format(
                    self.indent(),
                    self.stack[-1].name,
                    args,
                ))
            return "\n".join(result)
        return "\n".join([
            "{}__device__ {} {}({});".format(
                self.indent(),
                self.return_type(node),
                node.name,
                args,
            )
        ])

    def visit_AnnAssign(self, node):
        # ignore non-device variables
        if type(node.target) is ast.Name:
            var_node = self.stack[-1].get_VariableNode(node.target.id)
            if not var_node.is_device:
                return ""
        # ignore _: list[_] = DeviceAllocator.array(_)
        if type(cg.ast_to_call_graph_type(self.stack, node.annotation)) is cg.ListTypeNode \
                and self.is_device_allocator_array(node.value):
            return ""
        if (type(self.stack[-1]) is cg.RootNode and type(node.annotation) is not ast.Subscript) \
                or not type(self.stack[-1]) is cg.RootNode and node.value is not None:
            return self.indent() + "static const {} {} = {};".format(
                cg.ast_to_call_graph_type(self.stack, node.annotation).to_cpp_type(),
                self.visit(node.target),
                self.visit(node.value))     
        return ""
     
    def visit_Name(self, node):
        return node.id

    def visit_Num(self, node):
        return "{}".format(node.n)

    def visit_NameConstant(self, node):
        if type(node.value) is bool:
            return "true" if node.value else "false"
        elif node.value is None:
            return "nullptr"
        return node.value

    def visit_arguments(self, node):
        args = []
        for arg in node.args:
            converted_arg = self.visit(arg)
            if len(self.stack) > 2 and type(self.stack[-1]) is cg.FunctionNode \
                    and type(self.stack[-2]) is cg.ClassNode and converted_arg == "self":
                continue
            arg_type = cg.ast_to_call_graph_type(self.stack, arg.annotation).to_cpp_type()
            args.append("{} {}".format(arg_type, converted_arg))
        return ", ".join(args)

    def visit_arg(self, node):
        return node.arg

    def is_device_allocator_array(self, node):
        return type(node) is ast.Call and type(node.func) is ast.Attribute \
            and type(node.func.value) is ast.Name \
            and node.func.value.id == "DeviceAllocator" and node.func.attr == "array"