import sys
import enum
import astunparse

def cpp_error(ast_node):
    print("UNSUPPORTED AST NODE: {}".format(astunparse.unparse(ast_node)),
        file=sys.stderr)
    sys.exit(1)

class Type(enum.Enum):
    Builder = 0
    Block = 1
    Expr = 2
    Stmt = 4
    arguments = 2048
    Comment = 9999

class Base():
    fields = []

    def __init__(self, type):
        self.type: Type = type
    
    def buildCpp(self, ctx):
        """
        :type ctx: BuildContext
        """
        assert False

    def buildHpp(self, ctx):
        return ""

class IgnoredNode(Base):
    def __init__(self, node):
        super(IgnoredNode, self).__init__(Type.Comment)
        self.node = node

class Module(Base):
    fields = ["body", "classes"]

    def __init__(self, body, classes):
        super(Module, self).__init__(Type.Block)
        self.body = body
        self.classes = classes

class Statement(Base):
    def __init__(self):
        super(Statement, self).__init__(Type.Stmt)
    
class ClassDef(Statement):
    fields = ["name", "bases", "body", "fields"]

    def __init__(self, name, bases, body, fields, **kwargs):
        super().__init__()
        self.name = name
        self.bases = bases
        self.keywords = kwargs.get("keywords", [])
        self.body = body
        self.fields = fields

class FunctionDef(Statement):
    fields = ["name", "args", "body", "returns"]

    def __init__(self, name, args, body, returns=None):
        super().__init__()
        self.name = name
        self.args = args
        self.returns = returns
        self.body = body

class Return(Statement):
    fields = ["value"]

    def __init__(self, value):
        super().__init__()
        self.value = value

class Assign(Statement):
    fields = ["targets", "value"]

    def __init__(self, targets, value):
        super().__init__()
        self.targets = targets
        self.value = value

class AnnAssign(Statement):
    fields = ["target", "value", "annotation", "is_global"]

    def __init__(self, target, value, annotation, is_global):
        self.target = target
        self.value = value
        self.annotation = annotation
        self.is_global = is_global

class Expression(Base):
    def __init__(self):
        super(Expression, self).__init__(Type.Expr)

class Name(Expression):
    fields = ["id"]

    def __init__(self, id):
        super().__init__()
        self.id = id
