# -*- coding: utf-8 -*-

import ast
import os, sys
import astunparse

from call_graph import CallGraphVisitor, MarkDeviceVisitor
from transformer import Normalizer, Inliner

def compile(source_code, dir_path, file_name):
    """
    Compile python source_code into c++ source file and header file
        source_code:    codes written in python
    """
    # Set the global variable for file name
    FILE_NAME = file_name
    # Generate python ast
    py_ast = ast.parse(source_code)
    # Generate python call graph
    cgv = CallGraphVisitor()
    cgv.visit(py_ast)
    # Mark all device data on the call graph
    mdv = MarkDeviceVisitor()
    mdv.visit(cgv.root)
    normalizer = Normalizer(mdv.root)
    ast.fix_missing_locations(normalizer.visit(py_ast))
    inliner = Inliner(normalizer.root)
    inliner.visit(py_ast)

    new_py_ast = astunparse.unparse(py_ast)

    return new_py_ast, None, None, None
    
