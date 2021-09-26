import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parentdir + "/src")

from sanajeh import PyAllocator
from nbody import Body
import time

"""
Compile only
"""
from sanajeh import PyCompiler

compiler: PyCompiler = PyCompiler("examples/nbody/nbody.py", "nbody")
compiler.compile()