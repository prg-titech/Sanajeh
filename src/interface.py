import sanajeh
import sys

"""
Options parser
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("prog", help="program to compile/run", type=str)
parser.add_argument("--c", help="compile file", action="store_true")
parser.add_argument("--r", help="render option", action="store_true")
parser.add_argument("--cpu", help="run the program sequentially", action="store_true")
args = parser.parse_args()

def compile_prog(file_name):
    compiler: PyCompiler = sanajeh.PyCompiler(file_name, file_name.split("/")[-1].split(".")[0])
    compiler.compile()
    # compiler.build()

def run_prog(file_name):
    split_file = file_name.split("/")
    module = split_file[-1].split(".")[0]
    directory = "/".join(split_file[:-1])
    sys.path.append(directory)
    main_func = getattr(__import__(module), "main")
    if args.cpu:
        main_func(sanajeh.SeqAllocator(), args.r)
    else:
        main_func(sanajeh.PyAllocator(file_name, file_name.split("/")[-1].split(".")[0]), args.r)

if args.c:
    compile_prog(args.prog)
else:
    run_prog(args.prog)