type_map = {}

def register(pytype, cpptype):
    type_map[pytype] = cpptype

register("bool", "bool")
register("int", "int")
register("float", "float") 
register("uint32_t", "uint32_t")
register("uint8_t", "uint8_t")   
register("RandomState", "curandState")

def convert(type_str):
    if type_str is None:
        # todo not sure if auto is fine
        return "auto"
    elif type_str not in type_map:
        return type_str + "*"
    return type_map[type_str]

# This also needs to know about class declared in kernel_initialize_bodies.
def do_all_convert(type_str):
    return type_map[type_str] if type_str in type_map else "int"
    """
    if type_str not in type_map:
        return ["int", "class"]
    return [type_map[type_str], "primitive"]
    """
