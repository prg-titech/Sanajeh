#!/usr/bin/env python3.7

import argparse
import sanajeh
import os, sys

def main():
    parser = argparse.ArgumentParser(description="Sanajeh Compiler")
    parser.add_argument("prog", type=str,
        help="Sanajeh file to compile or run")
    parser.add_argument("--emit-py", action="store_true",
        help="Show the compiled Python file")
    parser.add_argument("--emit-cpp", action="store_true", 
        help="Show the compiled C++ file")
    parser.add_argument("--emit-hpp", action="store_true",
        help="Show the compiled C++ header file")
    parser.add_argument("--emit-cdef", action="store_true",
        help="Show the compiled C++ cdef file")
    parser.add_argument("--run", action="store_true", 
        help="Run the file alongside the compiled C++ file")
    parser.add_argument("--render", action="store_true", 
        help="Run the file with render option")
    parser.add_argument("--cpu", action="store_true",
        help="Run the program sequentially")
    parser.add_argument("--verbose", action="store_true",
        help="Print the multiple inheritance conversion process")
    args = parser.parse_args()
    file_path = args.prog
    emit_py = args.emit_py
    emit_cpp = args.emit_cpp
    emit_hpp = args.emit_hpp
    emit_cdef = args.emit_cdef
    to_run = args.run
    to_render = args.render
    run_cpu = args.cpu
    verbose = args.verbose

    directory = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    file_name, _ = os.path.splitext(basename)

    sys.path.append(directory)
    main_func = getattr(__import__(file_name), "main")

    if run_cpu:
        main_func(sanajeh.SeqAllocator(), to_render)
        return
    elif to_run:
        main_func(sanajeh.PyAllocator(file_name), to_render)
        return

    compiler: PyCompiler = sanajeh.PyCompiler(file_path)
    compiler.compile(emit_py, emit_cpp, emit_hpp, emit_cdef, verbose)

if __name__ == "__main__":
    main()
