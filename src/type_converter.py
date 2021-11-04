import ast

type_map = {}

def register(pytype, cpptype):
    type_map[pytype] = cpptype

register("bool", "bool")
register("int", "int")
register("float", "float") 
register("uint32_t", "uint32_t")
register("uint8_t", "uint8_t") 
register("curandState", "curandState&")

def convert(type_str):
    if type_str is None:
        # todo not sure if auto is fine
        return "auto"
    elif type_str not in type_map:
        return type_str + "*"
    return type_map[type_str]

def convert_ann(ann):
    if type(ann) == ast.Name:
        if ann.id not in type_map:
            return ann.id + "*"
        else:
            return type_map[ann.id]
    elif type(ann) == ast.Attribute:
        if hasattr(ann.value, "id") and ann.value.id == "DeviceAllocator" \
        and ann.attr == "RandomState":
            return "curandState"
    elif type(ann) == ast.Subscript:
        if hasattr(ann.value, "id") and ann.value.id == "list":
            return "list"
    return None

def do_all_convert(type_str):
    return type_str if type_str in ["int", "bool", "float"] else "int"