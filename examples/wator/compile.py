import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyCompiler

compiler: PyCompiler = PyCompiler("examples/wa-tor/wator.py", "wa-tor")
compiler.compile()