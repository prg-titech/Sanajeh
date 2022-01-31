# -*- coding: utf-8 -*-
# Mark all device functions

import ast, os, sys
import astunparse
import build_cpp

from config import INDENT
from call_graph import CallGraph
from gen_callgraph import GenPyCallGraphVisitor
from preprocessor import Preprocessor
from class_inspector import ClassInspectorMain

from gen_cppast import GenCppAstVisitor
from transformer import Normalizer, Searcher, Inliner, Eliminator, FieldSynthesizer, get_annotation

def pprint(node):
    print(astunparse.unparse(node))

# Should belong to call graph data
def expand_type(callGraphData, ftype):
    result = []
    for class_node in callGraphData.declared_classes:
        if class_node.name == ftype.v_type:
            for nested_field in class_node.declared_fields:
                result.extend(expand_type(callGraphData,nested_field))
            return result
    result.append(ftype)
    return result

# Should belong to call graph data
def expand_fields(callGraphData):
    for class_node in callGraphData.declared_classes:
        for field in class_node.declared_fields:
            if field.name.split("_")[-1] != "ref":
                class_node.expanded_fields[field.name] = expand_type(callGraphData,field)
            else:
                class_node.expanded_fields[field.name] = [field]

def transform(node, call_graph):
    """Replace all self-defined type fields with specific implementation"""
    ast.fix_missing_locations(Normalizer(call_graph).visit(node))

    callGraphData = CallGraph('root')
    gpcgv = GenPyCallGraphVisitor(callGraphData)
    gpcgv.visit(node)

    # Mark all device data on the call graph
    preprocess = Preprocessor(callGraphData)
    preprocess.visit(node)
    if not preprocess.has_device_data:
        print("No device data found")
        sys.exit(1)
    callGraphData.MarkDeviceDataByClassName(preprocess.classes)

    call_graph = callGraphData
    scr = Searcher(call_graph)
    scr.visit(node)
    for cls in scr.dev_cls:
        if cls in scr.sdef_cls:
            scr.sdef_cls.remove(cls)

    ast.fix_missing_locations(Inliner(call_graph, node, scr.sdef_cls).visit(node))
    ast.fix_missing_locations(Eliminator(call_graph, node, scr.sdef_cls).visit(node))  
    ast.fix_missing_locations(FieldSynthesizer(call_graph, scr.sdef_cls).visit(node)) 

def compile(source_code, dir_path, file_name):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name

    classinspector = ClassInspectorMain(source_code)
    classinspector.Start()

    # Generate python ast
    py_ast = classinspector.ast_code
    # py_ast = ast.parse(source_code)

    # Generate python call graph
    callGraphData = CallGraph('root')
    gpcgv = GenPyCallGraphVisitor(callGraphData)
    gpcgv.visit(py_ast)
    expand_fields(callGraphData)

    # Mark all device data on the call graph
    preprocess = Preprocessor(callGraphData)
    preprocess.visit(py_ast)
    if not preprocess.has_device_data:
        print("No device data found")
        sys.exit(1)
    callGraphData.MarkDeviceDataByClassName(preprocess.classes)

    # #-------------------------------------------
    # print("exit")
    # sys.exit(1)
    # print("shouldnt called")
    # #-------------------------------------------

    # Transform nested-objects to basic fields
    transform(py_ast, callGraphData)

    # Generate transformed python ast
    py_code1 = astunparse.unparse(py_ast)
    py_ast1 = ast.parse(py_code1)

    # Generate transformed python call graph
    callGraphData1 = CallGraph('root')
    gpcgv1 = GenPyCallGraphVisitor(callGraphData1)
    gpcgv1.visit(py_ast1)
    expand_fields(callGraphData1)

    # Mark all device data on the transformed call graph
    preprocess1 = Preprocessor(callGraphData1)
    preprocess1.visit(py_ast1)
    if not preprocess1.has_device_data:
        print("No device data found")
        sys.exit(1)
    callGraphData1.MarkDeviceDataByClassName(preprocess1.classes)

    # Generate cpp ast from transformed python ast
    gcv = GenCppAstVisitor(callGraphData1)
    cpp_node = gcv.visit(py_ast1)

    # Generate cpp(hpp) code from cpp ast
    ctx = build_cpp.BuildContext.create()

    # Expression needed for DynaSOAr API
    cpp_include_expr = '#include "{}.h"\n\n'.format(FILE_NAME)
    allocator_declaration = "AllocatorHandle<AllocatorT>* allocator_handle;\n" \
                            "__device__ AllocatorT* device_allocator;\n"
    precompile_expr = "#ifndef SANAJEH_DEVICE_CODE_H" \
                      "\n#define SANAJEH_DEVICE_CODE_H" \
                      "\n#define KNUMOBJECTS 64*64*64*64"
    hpp_include_expr = '\n\n#include <curand_kernel.h>\n#include "dynasoar.h"'


    init_cpp = ['\n\nextern "C" int AllocatorInitialize(){\n',
                INDENT +
                "allocator_handle = new AllocatorHandle<AllocatorT>(/* unified_memory= */ true);\n",
                INDENT +
                "AllocatorT* dev_ptr = allocator_handle->device_pointer();\n",
                INDENT +
                "cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0, cudaMemcpyHostToDevice);\n",
                preprocess1.build_global_device_variables_init(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    init_hpp = '\nextern "C" int AllocatorInitialize();\n'
    init_cdef = '\nint AllocatorInitialize();'


    unit_cpp = ['\n\nextern "C" int AllocatorUninitialize(){\n',
                preprocess1.build_global_device_variables_unit(),
                INDENT +
                "return 0;\n"
                "}"
                ]
    unit_hpp = '\nextern "C" int AllocatorUninitialize();\n'
    unit_cdef = '\nint AllocatorUninitialize();'
    endif_expr = "\n#endif"

    # Source code
    cpp_code = cpp_include_expr \
               + allocator_declaration \
               + cpp_node.buildCpp(ctx) \
               + preprocess1.build_do_all_cpp() \
               + preprocess1.build_parallel_do_cpp() \
               + preprocess1.build_parallel_new_cpp() \
               + "".join(init_cpp) \
               + "".join(unit_cpp)
    # Header code
    hpp_code = precompile_expr \
               + hpp_include_expr \
               + cpp_node.buildHpp(ctx) \
               + preprocess1.build_do_all_hpp() \
               + preprocess1.build_parallel_do_hpp() \
               + preprocess1.build_parallel_new_hpp() \
               + init_hpp \
               + unit_hpp \
               + endif_expr

    # Codes for cffi cdef() function
    cdef_code = preprocess1.build_parallel_do_cdef() \
                + preprocess1.build_parallel_new_cdef() \
                + init_cdef + preprocess1.build_do_all_cdef() \
                + unit_cdef
    """
    if not os.path.isdir(DIC_NAME):
      os.mkdir(DIC_NAME)
    with open(CPP_FILE_PATH, mode='w+') as cpp_file:
        cpp_file.write(cpp_code)
    with open(HPP_FILE_PATH, mode='w+') as hpp_file:
        hpp_file.write(hpp_code)
    with open(CDEF_FILE_PATH, mode='w+') as cdef_file:
        cdef_file.write(cdef_code)
    with open(PY_FILE_PATH, mode='w+') as py_file:
        py_file.write(py_code1)
    """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    pth: str = dir_path + "/" + file_name
    with open(pth + ".cu", mode="w") as cpp_file:
        cpp_file.write(cpp_code)
    with open(pth + ".h", mode="w") as hpp_file:
        hpp_file.write(hpp_code)
    with open(pth + ".cdef", mode="w") as cdef_file:
        cdef_file.write(cdef_code)
    with open(pth + "_py.py", mode="w") as py_file:
        py_file.write(py_code1)
    return cpp_code, hpp_code, cdef_code