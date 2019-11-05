import ast

source = open('./python_examples/nbody.py', encoding="utf-8").read()
tree = ast.parse(source)

tree
# for i in ast.walk(tree):
#     for j in ast.iter_fields(i):
#         print(j)



