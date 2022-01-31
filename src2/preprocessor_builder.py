import ast, sys
import type_converter
from config import INDENT
from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode

# Build codes for parallel_new in c++
class ParallelNewBuilder:
    def __init__(self, class_name):
        self.__class_name = class_name  # The class of the object

    def buildCpp(self):
        parallel_new_expr = INDENT + "allocator_handle->parallel_new<{}>(object_num);\n".format(self.__class_name)
        return_expr = INDENT + "return 0;"
        return 'extern "C" int parallel_new_{}(int object_num){{\n'.format(self.__class_name) \
               + parallel_new_expr \
               + return_expr \
               + "\n}"

    def buildHpp(self):
        return 'extern "C" int parallel_new_{}(int object_num);'.format(self.__class_name)

    def buildCdef(self):
        return 'int parallel_new_{}(int object_num);'.format(self.__class_name)


# Collect information of those functions used in the parallel_do function, and build codes for that function in c++
class ParallelDoBuilder(ast.NodeVisitor):
    def __init__(self, rt, class_name, func_class_name, func_name):
        self.__root = rt
        self.__node_path = [rt]
        self.__current_node = None
        self.__object_class_name = class_name  # The class of the object
        self.__func_class_name = func_class_name  # The class of the function executed
        self.__func_name = func_name
        self.__args = {}

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        super().visit(node)

    def visit_ClassDef(self, node):
        if node.name != self.__func_class_name:
            return
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} is not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is None:
            # Program shouldn't come to here, which means the function does not exist
            print("The function {} does not exist.".format(func_name), file=sys.stderr)
            sys.exit(1)
        if func_name != self.__func_name or self.__current_node.name != self.__func_class_name:
            return
        for arg_ in node.args.args:
            if arg_.arg == 'self':
                continue
            self.__args[arg_.arg] = arg_.annotation.id

    def buildCpp(self):
        arg_strs = []
        for arg_ in self.__args:
            arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))
        parallel_do_expr = INDENT + "allocator_handle->parallel_do<{}, &{}::{}>({});\n".format(
            self.__object_class_name,
            self.__func_class_name,
            self.__func_name,
            ", ".join(self.__args)
        )
        return_expr = INDENT + "return 0;"

        return 'extern "C" int {}_{}_{}({}){{\n'.format(
            self.__object_class_name,
            self.__func_class_name,
            self.__func_name,
            ", ".join(arg_strs)) \
               + parallel_do_expr \
               + return_expr \
               + "\n}"

    def buildHpp(self):
        arg_strs = []
        for arg_ in self.__args:
            arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))

        return 'extern "C" int {}_{}_{}({});'.format(
            self.__object_class_name,
            self.__func_class_name,
            self.__func_name,
            ",".join(arg_strs)
        )

    def buildCdef(self):
        arg_strs = []
        for arg_ in self.__args:
            # arg_strs.append("{} {}".format(type_converter.convert(self.__args[arg_]), arg_))
            args_strs.append("{} {}".format(type_converter.cdef_convert(self.__args[arg_]). arg_))

        return 'int {}_{}_{}({});'.format(
            self.__object_class_name,
            self.__func_class_name,
            self.__func_name,
            ",".join(arg_strs)
        )


# Collect information of class fields and build codes do_all functions in c++
class DoAllBuilder(ast.NodeVisitor):
    def __init__(self, rt, class_name):
        self.__root = rt
        self.__node_path = [rt]
        self.__current_node = None
        self.__class_name = class_name
        self.__field = {}
        self.__parent = None

        # check for parent classes
        # only supports 1-level inheritance
        class_node = self.__root.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} does not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)
        self.__parent = class_node.super_class

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        super().visit(node)

    def visit_ClassDef(self, node):
        if node.name != self.__class_name and node.name != self.__parent:
            return
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} does not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)

        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is None:
            # Program shouldn't come to here, which means the function does not exist
            print("The function {} does not exist.".format(func_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    def visit_AnnAssign(self, node):
        if type(self.__current_node) is FunctionNode and self.__current_node.name == "__init__":
            var = node.target.attr
            if type(node.annotation) is ast.Subscript and node.annotation.value.id == "list":
                var_type = "list"
            else:
                var_type = type_converter.convert_ann(node.annotation)
            self.__field[var] = var_type

    def buildCpp(self):
        fields_str = ""
        field_types_str = ""
        for i, field in enumerate(self.__field):
            field_type = self.__field[field]
            if i != len(self.__field) - 1:
                if field_type not in ["int", "float", "bool"]:
                    # fields_str += "(int) this->{}, ".format(field)
                    fields_str += "0, ".format(field)
                    field_types_str += "{}, ".format("int")
                else:
                    fields_str += "this->{}, ".format(field)
                    field_types_str += "{}, ".format(field_type)
            else:
                if self.__field[field] not in ["int", "float", "bool"]:
                    fields_str += "0".format(field)
                    field_types_str += "{}".format("int") 
                else:
                    fields_str += "this->{}".format(field)
                    field_types_str += "{}".format(field_type) 
        func_exprs = ['\n' +
                      'void {}::_do(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                      INDENT +
                      'pf({});\n'.format(fields_str) +
                      '}',
                      '\n' +
                      'extern "C" int {}_do_all(void (*pf)({})){{\n'.format(self.__class_name, field_types_str) +
                      INDENT +
                      'allocator_handle->template device_do<{}>(&{}::_do, pf);\n '.format(self.__class_name,
                                                                                          self.__class_name) +
                      INDENT + 'return 0;\n' +
                      '}']
        return "\n".join(func_exprs)

    def buildHpp(self):
        return 'extern "C" {}\n'.format(self.buildCdef())

    def buildCdef(self):
        field_types_str = ""
        for i, field in enumerate(self.__field):
            field_type = self.__field[field] if self.__field[field] in ["int", "bool", "float"] else "int"
            if i != len(self.__field) - 1:
                field_types_str += "{}, ".format(field_type)
            else:
                field_types_str += "{}".format(field_type)
        return 'int {}_do_all(void (*pf)({}));'.format(self.__class_name, field_types_str)