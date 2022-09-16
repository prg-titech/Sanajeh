# -*- coding: utf-8 -*-

import ast, sys
from typing import Set

def ast_error(msg: str, ast_node):
    print("({},{}): {}".format(ast_node.lineno, ast_node.col_offset, msg), 
        file=sys.stderr)
    sys.exit(1)

class CallGraphNode:
    def __init__(self):
        self.is_device = False

    @property
    def name(self):
        return None

    @property
    def type(self):
        return TypeNode()        

class RootNode(CallGraphNode):
    def __init__(self):
        self.declared_classes: Set[ClassNode] = set()

        random_class: ClassNode = ClassNode("random", None)
        random_class.declared_functions.add(FunctionNode("getrandbits", "random", IntNode()))
        random_class.declared_functions.add(FunctionNode("uniform", "random", FloatNode()))
        random_class.declared_functions.add(FunctionNode("seed", "random", None))
        self.declared_classes.add(random_class)

        da_class: ClassNode = ClassNode("DeviceAllocator", None)
        da_class.declared_functions.add(FunctionNode("new", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("destroy", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("device_do", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("parallel_do", "DeviceAllocator", None))
        da_class.declared_functions.add(FunctionNode("array", "DeviceAllocator", None))
        self.declared_classes.add(da_class)

        self.declared_functions: Set[FunctionNode] = set()
        self.library_functions: Set[FunctionNode] = set()
        self.called_functions: Set[FunctionNode] = set()
        self.declared_variables: Set[VariableNode] = set()
        self.called_variables: Set[VariableNode] = set()

        self.device_class_names = set()
        self.fields_class_names = set()
        self.has_device_data = False

    def get_ClassNode(self, class_name):
        for class_node in self.declared_classes:
            if class_node.name == class_name:
                return class_node
        return None
    
    def get_FunctionNode(self, function_name, class_name):
        for class_node in self.declared_classes:
            ret = class_node.get_FunctionNode(function_name, class_name)
            if ret is not None:
                return ret 
        else:
            for function_node in self.declared_functions:
                if function_node.name == function_name:
                    return function_node
            for function_node in self.library_functions:
                if function_node.name == function_name:
                    return function_node      
        return None

    def get_VariableNode(self, var_name):
        for var_node in self.declared_variables:
            if var_node.name == var_name:
                return var_node
        return None

class ClassNode(CallGraphNode):
    def __init__(self, class_name, super_class, ast_node=None):
        super().__init__()
        self.class_name = class_name
        self.super_class = super_class
        self.ast_node = ast_node
        self.declared_functions: Set[FunctionNode] = set()
        self.declared_variables: Set[VariableNode] = set()
        self.declared_fields: list[VariableNode] = []
        self.expanded_fields: dict = {}
        self.has_random_state: bool = False

    @property
    def name(self):
        return self.class_name

    @property
    def type(self):
        return ClassTypeNode(self)

    def get_FunctionNode(self, function_name, class_name):
        if class_name is None:
            for func_node in self.declared_functions:
                if func_node.name == function_name:
                    return func_node
        for function_node in self.declared_functions:
            if function_node.name == function_name and \
                    function_node.host_name == class_name:
                return function_node
        return None

    def get_VariableNode(self, var_name):
        for declared_variable in self.declared_variables:
            if var_name == declared_variable.name:
                return declared_variable
        return None

class FunctionNode(CallGraphNode):
    def __init__(self, function_name, host_name, return_type, ast_node=None):
        super().__init__()
        self.function_name = function_name
        self.host_name = host_name
        self.return_type = return_type        
        self.ast_node = ast_node
        self.arguments: list[VariableNode] = []
        self.declared_functions: Set[FunctionNode] = set()
        self.called_functions: Set[FunctionNode] = set()
        self.declared_variables: Set[VariableNode] = set()
        self.called_variables: Set[VariableNode] = set()

    @property
    def name(self):
        return self.function_name

    def get_FunctionNode(self, function_name, class_name):
        if class_name is None:
            for function_node in self.declared_functions:
                if function_node.name == function_name:
                    return function_node
        for function_node in self.declared_functions:
            if function_node.name == function_name and \
                function_node.host_name == class_name:
                return function_node
        return None

    def get_VariableNode(self, var_name):
        for declared_variable in self.declared_variables:
            if var_name == declared_variable.name:
                return declared_variable
        return None 

class VariableNode(CallGraphNode):
    def __init__(self, var_name, var_type):
        super().__init__()
        self.var_name = var_name
        self.var_type = var_type
        self.is_device = True if var_name == "kSeed" else False

    @property
    def name(self):
        return self.var_name

    @property
    def type(self):
        return self.var_type

""" types for nodes in the call graph """

class TypeNode():
    def __init__(self):
        pass

    @property
    def name(self):
        return None
    
    @property
    def element_type(self):
        return None

    def to_cpp_type(self):
        return "auto"

class IntNode(TypeNode):
    def __init__(self, unsigned=False, size=None):
        super().__init__()
        self.unsigned = unsigned
        self.size = size

    @property
    def name(self):
        return "int"
    
    def to_cpp_type(self):
        if self.unsigned:
            return "uint" + str(self.size) + "_t"
        else:
            return "int" 

class FloatNode(TypeNode):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "float"
    
    def to_cpp_type(self):
        return "float"

class BoolNode(TypeNode):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "bool"

    def to_cpp_type(self):
        return "bool"

class CurandStateTypeNode(TypeNode):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "curandState"

    def to_cpp_type(self):
        return "curandState&"

class ListTypeNode(TypeNode):
    def __init__(self, element_type):
        super().__init__()
        self.element_type = element_type
    
    @property
    def name(self):
        return "list[" + self.element_type.name + "]"

    @property
    def element_type(self):
        return self.element_type

class ClassTypeNode(TypeNode):
    def __init__(self, class_node):
        super().__init__()
        self.class_node = class_node

    @property
    def name(self):
        return self.class_node.name

    def to_cpp_type(self):
        return self.name + "*"

class RefTypeNode(TypeNode):
    def __init__(self, type_node):
        super().__init__()
        self.type_node = type_node

    @property
    def name(self):
        return self.type_node.name    

    def to_cpp_type(self):
        return self.name + "*"

""" visit AST and build call graph """

class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack = [RootNode()]
        self.current_node = None

    @property
    def root(self):
        return self.stack[0]

    def visit(self, node):
        self.current_node = self.stack[-1]
        super(CallGraphVisitor, self).visit(node)
    
    def visit_Module(self, node):
        self.generic_visit(node)

    def visit_Global(self, node):
        for global_variable in node.names:
            var_node = self.root.get_VariableNode(global_variable)
            if var_node is None:
                ast_error("The global variable {} does not exist".format(global_variable))
            self.stack[-1].called_variables.add(var_node)

    def visit_ClassDef(self, node):
        if type(self.current_node) is not RootNode:
            ast_error("Sanajeh does not yet support nested classes", node)
        class_name = node.name
        class_node = self.current_node.get_ClassNode(class_name)
        if class_node is not None:
            ast_error("The class {} is already defined".format(class_name), node)
        super_class = None
        if len(node.bases) == 1:
            super_class = node.bases[0].id
        elif len(node.bases) > 1:
            ast_error("Sanajeh does not yet support multiple inheritances", node)
        class_node = ClassNode(class_name, super_class, ast_node=node)
        self.current_node.declared_classes.add(class_node)
        self.stack.append(class_node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        func_name = node.name
        func_node = self.current_node.get_FunctionNode(func_name, self.current_node.name)
        if func_node is not None:
            ast_error("Function {} is already defined".format(func_name), node)
        return_type = None
        if type(self.current_node) is ClassNode and func_name == "__init__":
            return_type = self.current_node.type
        elif hasattr(node.returns, "id"):
            return_type = ast_to_call_graph_type(self.stack, node.returns)
        func_node = FunctionNode(func_name, self.current_node.name, return_type, ast_node=node)
        self.current_node.declared_functions.add(func_node)
        self.stack.append(func_node)
        self.generic_visit(node)
        self.stack.pop()
    
    def visit_arguments(self, node):
        if type(self.current_node) is not FunctionNode:
            ast_error("Argument should be found in a function", node)
        for arg in node.args:
            var_type = ast_to_call_graph_type(self.stack, arg.annotation)
            var_node = VariableNode(arg.arg, var_type)
            self.current_node.arguments.append(var_node)

    def visit_Assign(self, node):
        for var in node.targets:
            var_name = None
            if type(var) is ast.Name:
                var_name = var.id
                if self.current_node.get_VariableNode(var_name) is None:
                    var_node = VariableNode(var_name, TypeNode())
                    self.current_node.declared_variables.add(var_node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        var = node.target
        if type(var) is ast.Attribute:
            var_name = var.attr
            var_type = ast_to_call_graph_type(self.stack, node.annotation, var_name=var_name)
            if hasattr(var.value, "id") and var.value.id == "self" \
                    and self.current_node.name == "__init__":
                field_node = VariableNode(var_name, var_type)
                self.stack[-2].declared_fields.append(field_node)
                if type(var_type) is ClassTypeNode:
                    self.stack[-2].expanded_fields[var_name] = self.expand_field(field_node)
                else:
                    self.stack[-2].expanded_fields[var_name] = [field_node]
        elif type(var) is ast.Name:
            var_name = var.id
            var_type = ast_to_call_graph_type(self.stack, node.annotation, var_name=var_name)
            if self.current_node.get_VariableNode(var_name) is None:
                var_node = VariableNode(var_name, var_type)
                self.current_node.declared_variables.add(var_node)
        self.generic_visit(node)

    def visit_Name(self, node):
        if self.current_node.get_VariableNode(node.id) is None:
            var_node = None
            for ancestor_node in self.stack[-2::-1]:
                if ancestor_node.get_VariableNode(node.id) is not None:
                    var_node = ancestor_node.get_VariableNode(node.id)
                    if var_node is None:
                        ast_error("Cannot find variable " + node.id, node)
                    self.current_node.called_variables.add(var_node)
                    break
        else:
            return
    
    def visit_Call(self, node):
        func_name = None
        var_type = TypeNode()
        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            if type(node.func.value) is ast.Name:
                if node.func.value.id == "random" and func_name == "seed" and \
                        type(self.stack[-2]) is ClassNode:
                    self.stack[-2].has_random_state = True
                if node.func.value.id == "DeviceAllocator" and func_name == "device_class":
                    self.root.has_device_data = True
                    for cls in node.args:
                        if cls.id not in self.root.device_class_names:
                            self.root.device_class_names.add(cls.id)
                if (node.func.value.id == "allocator" or node.func.value.id == "PyAllocator") and \
                        func_name == "parallel_new":
                    self.root.has_device_data = True
                    if node.args[0].id not in self.root.device_class_names:
                        self.root.device_class_names.add(node.args[0].id)
            elif type(node.func.value) is ast.Attribute:
                if hasattr(node.func.value.value, "id") and node.func.value.value.id == "self" and \
                        type(self.stack[-2]) is ClassNode:
                    for var in self.stack[-2].declared_fields:
                        if var.name == node.func.value.attr:
                            var_type = var.type
        elif type(node.func) is ast.Name:
            func_name = node.func.id
        
        call_node = None
        for parent_node in self.stack[::-1]:
            if type(parent_node) is FunctionNode:
                continue
            call_node = parent_node.get_FunctionNode(func_name, var_type.name)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionNode(func_name, var_type, None)
            self.root.library_functions.add(call_node)
        self.current_node.called_functions.add(call_node)
        self.generic_visit(node)

    def expand_field(self, field_node):
        result = []
        if type(field_node.type) is ClassTypeNode and field_node.name.split("_")[-1] == "ref":
            for class_node in self.root.declared_classes:
                if check_equal_type(class_node.type, field_node.type):
                    for nested_field in class_node.declared_fields:
                        result.extend(self.expand_field(nested_field))
        else:
            result.append(field_node)
        return result

""" visit call graph and marks corresponding nodes as device nodes """

"""

class MarkDeviceVisitor:
    def __init__(self):
        self.root = None

    def visit(self, node):
        self.root = node
        if type(node) is RootNode:
            self.visit_RootNode(node, self.root.device_class_names)

    def visit_RootNode(self, node, device_class_names):
        for cln in device_class_names:
            children = []
            for cls in node.declared_classes:
                if cls.super_class == cln and cls.name not in device_class_names \
                        and not cls.is_device:
                    children.append(cls.name)
                if cls.name == cln:
                    self.visit_ClassNode(cls)
                    for field in cls.declared_fields:
                        self.visit_FieldNode(field)
                    for func in cls.declared_functions:
                        self.visit_FunctionNode(func)
                    for var in cls.declared_variables:
                        self.visit_VariableNode(var)
        if children:
            self.root.device_class_names.extend(children)
            self.visit_RootNode(self.root, children)

    def visit_ClassNode(self, node):
        queue = [node]
        while len(queue) > 0:
            check = queue[0]
            if check.is_device:
                queue.pop(0)
                continue
            check.is_device = True
            for var_node in check.declared_variables:
                self.visit_VariableNode(var_node)
            queue.pop(0)
    
    def visit_FieldNode(self, node):
        node.is_device = True
        field_class = None
        if type(node.type) is ListTypeNode and type(node.type.element_type) is ClassTypeNode \
                and node.type.element.class_name != "RandomState":
            field_class = self.root.get_ClassNode(node.type.element.name)
        elif type(node.type) is ClassTypeNode and node.type.name != "RandomState":
            field_class = self.root.get_ClassNode(node.type.name)
        if field_class is not None and not field_class.is_device:
            if not field_class.name in self.root.fields_class_names:
                self.root.fields_class_names.append(field_class.name)
            self.visit_RootNode(self.root, [field_class.name])

    def visit_VariableNode(self, node):
        node.is_device = True
        if type(node.type) is ClassTypeNode \
                and not node.type.name in self.root.fields_class_names:
            self.root.fields_class_names.append(node.type.name)

    def visit_FunctionNode(self, node):
        queue = [node]
        while len(queue) > 0:
            check = queue[0]
            if check.is_device:
                queue.pop(0)
                continue
            check.is_device = True
            for var_node in check.called_variables:
                self.visit_VariableNode(var_node)
            for var_node in check.declared_variables:
                self.visit_VariableNode(var_node)
            for func_node in check.called_functions:
                if not func_node.is_device:
                    queue.append(func_node)
            queue.pop(0)

"""

class MarkDeviceVisitor:
    def __init__(self):
        self.root = None
    
    def visit(self, node):
        self.root = node
        if type(node) is RootNode:
            self.visit_RootNode(node)
        
    def visit_RootNode(self, node):
        for class_node in node.declared_classes:
            if class_node.name in self.root.device_class_names:
                self.visit_ClassNode(class_node)
        for device_class in self.root.device_class_names:
            if device_class in self.root.fields_class_names:
                self.root.fields_class_names.remove(device_class)

    def visit_ClassNode(self, node):
        node.is_device = True
        if not node.name in self.root.device_class_names:
            self.root.device_class_names.add(node.name)
        if node.super_class is not None:
            super_class_node = self.root.get_ClassNode(node.super_class)
            if not super_class_node.is_device:
                self.visit_ClassNode(super_class_node)
        for field_node in node.declared_fields:
            self.visit_FieldNode(field_node)
        for func_node in node.declared_functions:
            self.visit_FunctionNode(func_node)
        for var_node in node.declared_variables:
            self.visit_VariableNode(var_node)
    
    def visit_FieldNode(self, node):
        node.is_device = True
        if type(node.type) is ListTypeNode:
            elem_type = node.type.element_type
            if type(elem_type) is ClassTypeNode:
                if not elem_type.name in self.root.fields_class_names:
                    self.root.fields_class_names.add(elem_type.name)
                self.visit_ClassNode(elem_type.class_node)
        elif type(node.type) is RefTypeNode and type(node.type.type_node) is ClassTypeNode:
            ref_type = node.type.type_node
            if not ref_type.class_node.is_device:
                if not ref_type.name in self.root.fields_class_names:
                    self.root.fields_class_names.add(ref_type.name)
                self.visit_ClassNode(ref_type.class_node)
        node_type = node.type
        if type(node_type) is ClassTypeNode:
            if not node_type.class_node.name in self.root.fields_class_names:
                self.root.fields_class_names.add(node_type.class_node.name)                
    
    def visit_FunctionNode(self, node):
        node.is_device = True
        for arg in node.arguments:
            if type(arg.type) is ClassTypeNode and not arg.type.class_node.name in self.root.fields_class_names:
                self.root.fields_class_names.add(arg.type.class_node.name)
        for var_node in node.called_variables:
            self.visit_VariableNode(var_node)
        for var_node in node.declared_variables:
            self.visit_VariableNode(var_node)
        for func_node in node.called_functions:
            if not func_node.is_device:
                self.visit_FunctionNode(func_node)

    def visit_VariableNode(self, node):
        node.is_device = True
        node_type = node.type
        if type(node_type) is ClassTypeNode:
            if not node_type.class_node.name in self.root.fields_class_names:
                self.root.fields_class_names.add(node_type.class_node.name)  

""" used to check equivalence between two typenodes """

def check_equal_type(ltype: TypeNode, rtype: TypeNode):
    if type(ltype) is TypeNode and type(rtype) is TypeNode:
        return True 
    if type(ltype) is IntNode and type(rtype) is IntNode:
        return True
    if type(ltype) is FloatNode and type(rtype) is FloatNode:
        return True
    if type(ltype) is BoolNode and type(rtype) is BoolNode:
        return True
    if type(ltype) is ListTypeNode and type(rtype) is ListTypeNode:
        return check_equal_type(ltype.element_type, rtype.element_type)
    if type(ltype) is ClassTypeNode and type(rtype) is ClassTypeNode:
        return ltype.name == rtype.name
    if type(ltype) is RefTypeNode and type(rtype) is RefTypeNode:
        return check_equal_type(ltype.type_node, rtype.type_node)
    return False        

""" convert AST nodes into types for call graph """

def ast_to_call_graph_type(stack, node, var_name=None):
    if type(node) is ast.Name:
        type_name = node.id
        if type_name == "self":
            if type(stack[-2]) is ClassNode:
                return ClassTypeNode(stack[-2])
        elif type_name == "int":
            return IntNode()
        elif type_name == "float":
            return FloatNode()
        elif type_name == "bool":
            return BoolNode()
        elif type_name == "uint32_t":
            return IntNode(True, 32)
        elif type_name == "uint8_t":
            return IntNode(True, 8)
        elif type_name == "curandState":
            return CurandStateTypeNode()
        else:
            i = len(stack) - 1
            while i >= 0:
                current_node = stack[i]
                # check declared variable
                var_node = current_node.get_VariableNode(type_name)
                if var_node:
                    return var_node.type
                # check function parameter
                if type(current_node) is FunctionNode:
                    for arg_node in current_node.arguments:
                        if type_name == arg_node.name:
                            return arg_node.type
                i -= 1
            # check declared classes
            for class_node in stack[0].declared_classes:
                if class_node.name == type_name:
                    if var_name:
                        split_var_name = var_name.split("_")
                        if split_var_name[-1] == "ref":
                            return RefTypeNode(ClassTypeNode(class_node))
                    return ClassTypeNode(class_node)    
    elif type(node) is ast.Attribute:
        if type(node.value) is ast.Name and node.value.id == "DeviceAllocator" \
                and node.attr == "RandomState":
            return CurandStateTypeNode()
        receiver_type = ast_to_call_graph_type(stack, node.value, var_name)
        if type(receiver_type) is not TypeNode and type(receiver_type) is not ListTypeNode:
            if type(receiver_type) is RefTypeNode:
                receiver_type = receiver_type.type_node
            if type(receiver_type) is ClassTypeNode:
                for field in receiver_type.class_node.declared_fields:
                    if field.name == node.attr:
                        return field.type
    elif type(node) is ast.Call:
        if type(node.func) is ast.Attribute:
            receiver_type = ast_to_call_graph_type(stack, node.func.value, var_name)
            for func in receiver_type.declared_functions:
                if func.name == node.func.attr:
                    return func.return_type
    elif type(node) is ast.Subscript and node.value.id == "list":
        element_type = ast_to_call_graph_type(stack, node.slice.value, var_name)
        if element_type is None:
            ast_error("Requires element type for list", node)
        return ListTypeNode(element_type)
    return TypeNode()