import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyCompiler
from nbody import Body

compiler: PyCompiler = PyCompiler("examples/nbody/nbody.py", "nbody")
compiler.compile()