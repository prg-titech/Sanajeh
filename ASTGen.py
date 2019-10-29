import ast

a = 1
print(a)
source = open('python_examples/nboby.py').read()
tree = ast.parse(source)
print(tree)
