import sanajeh
import sys

"""
Options parser
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("prog", help="program to compile/run", type=str)
parser.add_argument("--c", help="compile file", action="store_true")
parser.add_argument("--b", help="only build file", action="store_true")
parser.add_argument("--r", help="render option", action="store_true")
parser.add_argument("--cpu", help="run the program sequentially", action="store_true")
args = parser.parse_args()

def compile_prog(file_name,bonly):
    compiler: PyCompiler = sanajeh.PyCompiler(file_name, file_name.split("/")[-1].split(".")[0])
    if not bonly:
        print("compile file: <{}>, arg <{}>".format(file_name,file_name.split("/")[-1].split(".")[0]))
        compiler.compile()
    print("build")

    compiler.build()
    print("finished")

def run_prog(file_name):
    split_file = file_name.split("/")
    module = split_file[-1].split(".")[0]
    directory = "/".join(split_file[:-1])
    sys.path.append(directory)
    main_func = getattr(__import__(module), "main")
    if args.cpu:
        main_func(sanajeh.SeqAllocator(), args.r)
    else:
        print("PyAllocator <{}> <{}>".format(file_name,file_name.split("/")[-1].split(".")[0]))
        main_func(sanajeh.PyAllocator(file_name, file_name.split("/")[-1].split(".")[0]), args.r)

if args.c or args.b:
    bonly = False
    if args.b:
        bonly = True
    compile_prog(args.prog,bonly)
else:
    run_prog(args.prog)