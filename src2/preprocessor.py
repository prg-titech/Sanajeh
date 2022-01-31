import ast, sys
import hashlib

from config import INDENT
from call_graph import CallGraph, ClassNode, FunctionNode, VariableNode

from preprocessor_builder import ParallelNewBuilder, ParallelDoBuilder, DoAllBuilder

# Find device class in python code and compile parallel_do expressions into c++ ones
class Preprocessor(ast.NodeVisitor):
    def __init__(self, rt: CallGraph):
        self.__classes = []

        self.has_device_data = False
        self.__is_root = True  # the flag of whether visiting the root node of python ast
        self.__node_root = None  # the root node of python ast
        self.__cpp_parallel_do_codes = []
        self.__hpp_parallel_do_codes = []
        self.__cdef_parallel_do_codes = []
        self.__cpp_parallel_new_codes = []
        self.__hpp_parallel_new_codes = []
        self.__cdef_parallel_new_codes = []
        self.__cpp_do_all_codes = []
        self.__hpp_do_all_codes = []
        self.__cdef_do_all_codes = []
        self.__parallel_do_hashtable = []
        self.__root = rt
        self.__node_path = [rt]
        self.__current_node = None
        self.global_device_variables = {}

    def visit(self, node):
        if self.__is_root:
            self.__node_root = node
            self.__is_root = False
        self.__current_node = self.__node_path[-1]
        super(Preprocessor, self).visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        class_node = self.__current_node.GetClassNode(class_name)
        if class_node is None:
            # Program shouldn't come to here, which means the class does not exist
            print("The class {} is not exist.".format(class_name), file=sys.stderr)
            sys.exit(1)
        self.__node_path.append(class_node)
        self.generic_visit(node)
        self.__node_path.pop()

    # Create nodes for all functions declared
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

    # Analyze function calling relationships
    def visit_Call(self, node):
        # Find device classes through device code
        # DeviceAllocator.device_class, .parallel_do or .array_size
        if type(node.func) is ast.Attribute and \
                hasattr(node.func.value, "id") and \
                node.func.value.id == "DeviceAllocator":

            if node.func.attr == 'device_class':
                self.has_device_data = True
                for cls in node.args:
                    if cls.id not in self.__classes:
                        self.__classes.append(cls.id)

                        self.ParallelNewBuild(cls.id)
                        self.DoAllBuild(cls.id)

            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    self.ParallelDoBuild( node.args[0].id, node.args[1].value.id, node.args[1].attr )

            elif node.func.attr == 'array_size':
                self.global_device_variables[str(node.args[0].id)] = node.args[1].n

        # Find device classes through host code
        # PyAllocator/allocator.parallel_new or .parallel_do.
        if type(node.func) is ast.Attribute \
                and hasattr(node.func.value, "id") and \
                (node.func.value.id == "allocator" or node.func.value.id == "PyAllocator"):
            if node.func.attr == 'parallel_new':
                self.has_device_data = True
                if node.args[0].id not in self.__classes:
                    self.__classes.append(node.args[0].id)
                    
                    self.ParallelNewBuild(node.args[0].id)
                    self.DoAllBuild(node.args[0].id)


            elif node.func.attr == 'parallel_do':
                hval = self.__gen_Hash([node.args[0].id, node.args[1].value.id, node.args[1].attr])
                if hval not in self.__parallel_do_hashtable:
                    self.__parallel_do_hashtable.append(hval)
                    self.ParallelDoBuild( node.args[0].id, node.args[1].value.id, node.args[1].attr )

        func_name = None
        call_node = None
        var_type = None

        # what is this code?
        if type(node.func) is ast.Attribute:
            func_name = node.func.attr
            if type(node.func.value) is ast.Attribute:
                if hasattr(node.func.value.value, "id") \
                and node.func.value.value.id == "self" \
                and type(self.__node_path[-2]) is ClassNode:
                    for var in self.__node_path[-2].declared_fields:
                        if var.name == node.func.value.attr:
                            var_type = var.v_type
        elif type(node.func) is ast.Name:
            func_name = node.func.id

        for parent_node in self.__node_path[::-1]:
            if type(parent_node) is FunctionNode:
                continue
            call_node = parent_node.GetFunctionNode(func_name, var_type)
            if call_node is not None:
                break
        if call_node is None:
            call_node = FunctionNode(func_name, var_type, None)
            self.__root.library_functions.add(call_node)
        self.__current_node.called_functions.add(call_node)
        self.generic_visit(node)


    def ParallelNewBuild(self,clsName):
        self.__classes.append(clsName)
        pnb = ParallelNewBuilder(clsName)
        self.__cpp_parallel_new_codes.append(pnb.buildCpp())
        self.__hpp_parallel_new_codes.append(pnb.buildHpp())
        self.__cdef_parallel_new_codes.append(pnb.buildCdef())

    def DoAllBuild(self,clsName):
        dab = DoAllBuilder(self.__root, clsName)
        dab.visit(self.__node_root)
        self.__cpp_do_all_codes.append(dab.buildCpp())
        self.__hpp_do_all_codes.append(dab.buildHpp())
        self.__cdef_do_all_codes.append(dab.buildCdef())

    def ParallelDoBuild(self,clsObject,clsName,funcName):
        pdb = ParallelDoBuilder(self.__root, clsObject, clsName, funcName)
        pdb.visit(self.__node_root)
        self.__cpp_parallel_do_codes.append(pdb.buildCpp())
        self.__hpp_parallel_do_codes.append(pdb.buildHpp())
        self.__cdef_parallel_do_codes.append(pdb.buildCdef())

    @property
    def cpp_parallel_do_codes(self):
        return self.__cpp_parallel_do_codes

    @property
    def hpp_parallel_do_codes(self):
        return self.__hpp_parallel_do_codes

    @property
    def cdef_parallel_do_codes(self):
        return self.__cdef_parallel_do_codes

    @property
    def cpp_parallel_new_codes(self):
        return self.__cpp_parallel_new_codes

    @property
    def hpp_parallel_new_codes(self):
        return self.__hpp_parallel_new_codes

    @property
    def cdef_parallel_new_codes(self):
        return self.__cdef_parallel_new_codes

    @property
    def cpp_do_all_codes(self):
        return self.__cpp_do_all_codes

    @property
    def hpp_do_all_codes(self):
        return self.__hpp_do_all_codes

    @property
    def cdef_do_all_codes(self):
        return self.__cdef_do_all_codes

    @property
    def classes(self):
        return self.__classes

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
        return '\n\n' + '\n\n'.join(self.__cpp_parallel_do_codes)

    def build_parallel_do_hpp(self):
        return '\n'.join(self.__hpp_parallel_do_codes)

    def build_parallel_do_cdef(self):
        return '\n' + '\n'.join(self.__cdef_parallel_do_codes)

    def build_do_all_cpp(self):
        return '\n\n'.join(self.__cpp_do_all_codes)

    def build_do_all_hpp(self):
        return '\n' + '\n'.join(self.__hpp_do_all_codes)

    def build_do_all_cdef(self):
        return '\n' + '\n'.join(self.__cdef_do_all_codes)

    def build_parallel_new_cpp(self):
        return '\n\n' + '\n\n'.join(self.__cpp_parallel_new_codes)

    def build_parallel_new_hpp(self):
        return '\n' + '\n'.join(self.__hpp_parallel_new_codes)

    def build_parallel_new_cdef(self):
        return '\n' + '\n'.join(self.__cdef_parallel_new_codes)

    def build_global_device_variables_init(self):
        ret = []
        for var in self.global_device_variables:
            var_node = self.__root.GetVariableNode(var, None)
            e_type = type_converter.convert(var_node.e_type)
            n = self.__global_device_variables[var]
            ret.append(INDENT + "{}* host_{};\n".format(e_type, var) + \
                       INDENT + "cudaMalloc(&host_{}, sizeof({})*{});\n".format(var, e_type, n) + \
                       INDENT + "cudaMemcpyToSymbol({}, &host_{}, sizeof({}*), 0, cudaMemcpyHostToDevice);\n" \
                       .format(var, var, e_type))
        return "\n".join(ret)

    def build_global_device_variables_unit(self):
        ret = []
        for var in self.global_device_variables:
            ret.append(INDENT + "cudaFree(host_{});\n".format(var))
        return "\n".join(ret)      

