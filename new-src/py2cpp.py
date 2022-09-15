# -*- coding: utf-8 -*-

import ast, os, sys
import hashlib
import six

import build_cpp as cpp
import call_graph as cg
from transformer import Normalizer, Inliner, Eliminator, FieldSynthesizer

import astunparse

INDENT: str = "\t"

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
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name \
                and node.func.value.id == "DeviceAllocator":
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
        if type(node.func) is ast.Attribute \
                and type(node.func.value) is ast.Name \
                and (node.func.value.id == "allocator" or node.func.value.id == "PyAllocator"):
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
                    self.__cpp_do_all_codes.append(dab.buildCpp())
                    self.__hpp_do_all_codes.append(dab.buildHpp())
                    self.__cdef_do_all_codes.append(dab.buildCdef())
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

class CppVisitor(ast.NodeVisitor):
    def __init__(self, root: cg.RootNode):
        self.stack = [root]
        self.indent_level = 0

    @property
    def root(self):
        return self.stack[0]

    def indent(self, dec=0):
        return (self.indent_level-dec)*"\t"
        
    def return_type(self, func_ast):
        if func_ast.returns is not None:
            return cg.ast_to_call_graph_type(self.stack, func_ast.returns).to_cpp_type()
        else:
            return "void"

    def visit_ClassDef(self, node):
        body = []
        self.stack.append(self.stack[-1].get_ClassNode(node.name))
        for class_body in node.body:
            body.append(self.visit(class_body))
        while "" in body:
            body.remove("")
        self.stack.pop()
        print("\n".join(body))
        return "\n".join(body)

    def visit_FunctionDef(self, node):
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
        if type(self.stack[-1]) is cg.RootNode:
            if type(node_type) is cg.ListTypeNode and type(node.value) is ast.Call \
                    and type(node.value.func.value) is ast.Name \
                    and node.value.func.value.id == "DeviceAllocator" \
                    and node.value.func.attr == "array":
                        return indent() + "__device__ {}* {}[{}];".format(
                            self.visit(node.annotation.slice),
                            self.visit(node.target),
                            ", ".join([self.visit(arg) for arg in node.value.args])) 
        else:
            if node.value is not None:
                if type(self.stack[-1]) is cg.FunctionNode and self.stack[-1].name == "__init__":
                    return self.indent() + "{} = {};".format(
                        self.visit(node.target),
                        self.visit(node.value)
                    ) 
                else:
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
            for orelse_body in node.orelse:
                built = self.visit(orelse_body)
                built = "\n".join([self.indent() + built_line for built_line in built.split("\n")])
                result.append(built)
            result.append(self.indent() + "}")
        return "\n".join(result)

    def visit_Expr(self, node):
        if self.visit(node.value) == "":
            return ""
        return self.indent() + "{};".format(self.visit(node.value))

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
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name \
                and node.func.value.id == "DeviceAllocator":
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
            else:
                # No API provided by Sanajeh
                assert False
        if type(node.func) is ast.Attribute and type(node.func.value) is ast.Name \
                and node.func.value.id == "random":
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
        args = ", ".join([self.visit(arg) for arg in node.args])
        return "{}({})".format(self.visit(node.func), args)

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


def compile(source_code, dir_path, file_name):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name

    # Generate python ast
    py_ast = ast.parse(source_code)

    # Generate python call graph and mark device data
    cgv = cg.CallGraphVisitor()
    cgv.visit(py_ast)
    mdv = cg.MarkDeviceVisitor()
    mdv.visit(cgv.root)

    # Transformation passes
    normalizer = Normalizer(mdv.root)
    ast.fix_missing_locations(normalizer.visit(py_ast))
    inliner = Inliner(normalizer.root)
    ast.fix_missing_locations(inliner.visit(py_ast))
    eliminator = Eliminator(inliner.root)
    ast.fix_missing_locations(eliminator.visit(py_ast))
    synthesizer = FieldSynthesizer(eliminator.root)
    ast.fix_missing_locations(synthesizer.visit(py_ast))
    
    # Rebuild the call graph after transformation
    recgv = cg.CallGraphVisitor()
    recgv.visit(py_ast)
    remdv = cg.MarkDeviceVisitor()
    remdv.visit(recgv.root)

    # Preprocessor (find device class in python code and compile parallel_do expressions into c++ ones)
    pp = Preprocessor(remdv.root)
    pp.visit(py_ast)

    cav = CppVisitor(pp.root)
    cav.visit(py_ast)

    new_py_ast = astunparse.unparse(py_ast)

    return new_py_ast, None, None, None
    
