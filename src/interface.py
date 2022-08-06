#!/usr/bin/env python3

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
    parser.add_argument("--run", action="store_true", 
        help="Run the file alongside the compiled C++ file")
    parser.add_argument("--render", action="store_true", 
        help="Run the file with render option")
    parser.add_argument("--cpu", action="store_true",
        help="Run the program sequentially")
    args = parser.parse_args()
    file_path = args.prog
    emit_py = args.emit_py
    emit_cpp = args.emit_cpp
    to_run = args.run
    to_render = args.render
    run_cpu = args.cpu

    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).split(".")[0]

    if to_run:
        sys.path.append(directory)
        main_func = getattr(__import__(file_name), "main")
        if run_cpu:
            main_func(sanajeh.SeqAllocator(), to_render)
        else:
            main_func(sanajeh.PyAllocator(file_path, file_name), to_render)
        return

    compiler: PyCompiler = sanajeh.PyCompiler(file_path)
    compiler.compile(emit_py, emit_cpp)

if __name__ == "__main__":
    main()
