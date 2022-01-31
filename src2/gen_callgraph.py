# GenPyCallGraphVisitor
import ast, sys

import preprocessor
from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode
from transformer import get_annotation

# Generate python call graph
class GenPyCallGraphVisitor(ast.NodeVisitor):
    def __init__(self,callGraph):
        self.__root = callGraph
        self.__node_path = [self.__root]
        self.__current_node = None
        self.__variables = {}

    @property
    def root(self):
        return self.__root

    # JUST FOR DEBUG
    @property
    def node_path(self):
        return self.__node_path

    # JUST FOR DEBUG
    @property
    def variables(self):
        return self.__variables

    def visit(self, node):
        self.__current_node = self.__node_path[-1]
        super(GenPyCallGraphVisitor, self).visit(node)

    def visit_Module(self, node):
        self.generic_visit(node)

    # Create nodes for all classes declared
    def visit_ClassDef(self, node):
        if type(self.__current_node) is not CallGraph:
            print("Doesn't support nested classes", file=sys.stderr)
            sys.exit(1)
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is not None:
            # Program shouldn't come to here, which means a class is defined twice
            print("The class {} is defined twice.".format(class_name), file=sys.stderr)
            sys.exit(1)
        base = None
        # todo does not supports multiple inheritance
        if len(node.bases) > 0:
            base = node.bases[0].id
        class_node = ClassNode(node.name)
        for base in node.bases:
            class_node.super_class.add(base.id)
        self.__current_node.declared_classes.add(class_node)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
    def visit_FunctionDef(self, node):
        func_name = node.name           
        if type(self.__current_node) is not CallGraph and type(self.__current_node) is not ClassNode \
        and self.__current_node.name != "main":
            print("Doesn't support nested functions", file=sys.stderr)
            sys.exit(1)
        func_node = self.__current_node.GetFunctionNode(func_name, self.__current_node.name)
        if func_node is not None:
            # Program shouldn't come to here, which means a function is defined twice
            print("The function {} is defined twice.".format(func_name), file=sys.stderr)
            sys.exit(1)
        ret_type = None
        if type(self.__current_node) is ClassNode and func_name == "__init__":
            ret_type = self.__current_node.name
        elif hasattr(node.returns, "id"):
            ret_type = node.returns.id

        func_node = FunctionNode(func_name, self.__current_node.name, ret_type)
        self.__current_node.declared_functions.add(func_node)
        self.__node_path.append(func_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Add arguments to the environment
    def visit_arguments(self, node):
        if type(self.__current_node) is not FunctionNode:
            print('Unexpected node "{}"'.format(self.__current_node.name), file=sys.stderr)
            sys.exit(1)
        for arg in node.args:
            var_type = None
            element_type = None
            if arg.arg == "self":
                continue
            if hasattr(arg, "annotation") and arg.annotation is not None:
                var_type, element_type = get_annotation(arg.annotation)
            var_node = VariableNode(arg.arg, var_type, element_type)
            self.__current_node.arguments.add(var_node)
            self.__variables.setdefault(self.__current_node.id, []).append(arg.arg)

    # Add global variables to the environment
    def visit_Global(self, node):
        for global_variable in node.names:
            self.__variables.setdefault(self.__current_node.id, []).append(global_variable)
            var_node = self.__root.GetVariableNode(global_variable, None)
            if var_node is None:
                print("The global variable {} is not existed.".format(global_variable), file=sys.stderr)
                sys.exit(1)
            self.__current_node.called_variables.add(var_node)

    # Create nodes for variables without type annotation
    def visit_Assign(self, node):
        for var in node.targets:
            var_name = None
            if type(var) is ast.Attribute:
                pass
                # var_name = var.attr
                # print(var_name, var.value.id)
                # # todo Attribute variables(self should refer to the class not in the current block),
                # # todo haven't thought about other occasions
                # if var.value.id == 'self':
                #     pass
            elif type(var) is ast.Name:
                var_name = var.id
                self.__variables.setdefault(self.__current_node.id, [])
                if var_name not in self.__variables[self.__current_node.id]:
                    var_node = VariableNode(var_name, None)
                    self.__current_node.declared_variables.add(var_node)
                    self.__variables[self.__current_node.id].append(var_name)

        self.generic_visit(node)

    # Create nodes for variables with type annotation
    def visit_AnnAssign(self, node):
        var = node.target
        ann, e_ann = get_annotation(node.annotation)
        if type(var) is ast.Attribute:
            var_name = var.attr
            if hasattr(var.value, "id") and var.value.id == "self" and self.__current_node.name == "__init__":
                field_node = VariableNode(var_name, ann, e_ann)
                self.node_path[-2].declared_fields.add(field_node)
            # todo Attribute variables(self should refer to the class not in the current block),
            # todo haven't thought about other occasions
        elif type(var) is ast.Name:
            var_name = var.id
            self.__variables.setdefault(self.__current_node.id, [])
            if var_name not in self.__variables[self.__current_node.id]:
                var_node = VariableNode(var_name, ann, e_ann)
                self.__current_node.declared_variables.add(var_node)
                self.__variables[self.__current_node.id].append(var_name)
        self.generic_visit(node)

    def visit_Name(self, node):
        self.__variables.setdefault(self.__current_node.id, [])
        if node.id in self.__variables[self.__current_node.id]:
            return
        for annotate_location_node in self.__node_path[-2::-1]:
            self.__variables.setdefault(annotate_location_node.id, [])
            if node.id in self.__variables[annotate_location_node.id]:
                var_node = annotate_location_node.GetVariableNode(node.id, None)
                if var_node is None:
                    print('Unexpected error, can not find variable "{}"', node.id, file=sys.stderr)
                    sys.exit(1)
                self.__current_node.called_variables.add(var_node)
                break

    def visit_Call(self, node):
        if hasattr(node.func, "value") and hasattr(node.func.value, "id") \
        and node.func.value.id == "random" and node.func.attr == "seed" \
        and type(self.__node_path[-2]) is ClassNode:
            self.__node_path[-2].has_random_state = True
        self.generic_visit(node)