import ast
from call_graph import CallGraph


class FieldTransformer:

    class Normalizer(ast.NodeTransformer):
        """Use variables to rewrite nested expressions"""
        pass

    class Expander(ast.NodeTransformer):
        """Replace function callings with specific implementation"""
        pass

    class Eliminator(ast.NodeTransformer):
        """Remove useless object constructors"""
        pass

    def transform(self, node):
        """Replace all self-defined type fields with specific implementation"""
        node = ast.fix_missing_locations(self.Normalizer().visit(node))
        node = ast.fix_missing_locations(self.Expander().visit(node))
        node = ast.fix_missing_locations(self.Eliminator().visit(node))
        return node


