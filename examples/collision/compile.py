import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyCompiler
from nested import Body

compiler: PyCompiler = PyCompiler("examples/collision/nested.py", "collision-nested")
compiler.compile()