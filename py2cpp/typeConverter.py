#
# type registry
#
type_map = {}


def register(pytype, cpptype):
    type_map[pytype] = cpptype


def convert(type_str, rettype=False):
    if type_str is None:
        # todo not sure if auto is fine
        return "auto"
    elif type_str not in type_map:
        return type_str + "*"
    return type_map[type_str]


register("bool", "bool")
register("int", "int")
register("float", "float")
