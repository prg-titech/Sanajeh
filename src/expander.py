class RuntimeExpander:
  built = {}
  flattened = {}
  
  def __init__(self):
    pass

  def build_function(self, cls):
    module = cls.__dict__["__module__"]
    name = cls.__name__
    args = {}
    func = "\t" + "new_object: {} = {}.__new__({})\n".format(name, name, name)
    for field, ftype in cls.__dict__["__annotations__"].items():
      if field.split("_")[-1] != "ref" and ftype not in ["int", "float", "bool"]:
        if ftype not in self.built.keys():
          self.build_function(getattr(__import__(module), ftype))
        nested_args = []
        for nested_field, nested_ftype in self.flattened[ftype].items():
          nested_args.append(field + "_" + nested_field)
          args[field + "_" + nested_field] = nested_ftype
        func += "\t" + "new_object: {} = {}({})\n".format(field, ftype, ", ".join(nested_args))
      else:
        args[field] = ftype
        func += "\t" + "new_object: {} = {}\n".format(field, field)
    func += "\t" + "return new_object"
    func = "def __rebuild_{}({}):\n".format(name, ", ".join(args)) + func
    self.built[name] = func
    self.flattened[name] = args
    exec(func, globals())
    setattr(cls, "__rebuild_{}".format(name), eval("__rebuild_{}".format(name)))

"""
import cffi
ffi = cffi.FFI()

class Handler:

  built = {}

  def __init__(self):
    pass

  def rebuild_function(self, cls):
    module = cls.__dict__["__module__"]    
    flattened = self.flatten(cls)
    for field, ftype in cls.__dict__["__annotations__"].items():
      if field.split("_")[-1] != "ref" and ftype not in ["int", "float", "bool"] and ftype not in self.built.keys():
        self.rebuild_function(getattr(__import__(module), ftype))
    new_function =  "def __rebuild_{}({}):\n".format(cls.__name__, ", ".join(flattened[0])) +\
                    "\t" + "new_object: {} = {}.__new__({})\n".format(cls.__name__, cls.__name__, cls.__name__)
    for field, ftype in cls.__dict__["__annotations__"].items():
      if field in flattened[1].keys():
        nested_build = ", ".join([nested_field for nested_field in flattened[1][field]])
        new_function += "\t" + "new_object.{} = {}({})\n".format(field, ftype, nested_build)
      else:
        new_function += "\t" + "new_object.{} = {}\n".format(field, field)
    new_function += "\t" + "return new_object"
    self.built[cls.__name__] = new_function
    exec(new_function, globals())
    setattr(cls, "__rebuild_{}".format(cls.__name__), eval("__rebuild_{}".format(cls.__name__)))

  def flatten(self, cls):
    field_map = {}
    nested_map = {}
    module = cls.__dict__["__module__"]
    if "__annotations__" in cls.__dict__.keys():
      for field, ftype in cls.__dict__["__annotations__"].items():
        if field.split("_")[-1] == "ref":
          field_map[field] = "int"
        elif ftype in ["int", "float", "bool"]:
          field_map[field] = ftype
        else:
          nested_map[field] = {}
          nested_result = self.flatten(getattr(__import__(module), ftype))
          for nested_field, nested_ftype in nested_result[0].items():
              field_map[field + "_" + nested_field] = nested_ftype
              nested_map[field][field + "_" + nested_field] = nested_ftype
    return [field_map, nested_map]    

class Allocator:

  handler: Handler = Handler()

  def __init__(self):
    pass
  
  def do_all(self, cls, func):
    class_name = cls.__name__
    callback_types = "void({})".format(", ".join(self.handler.flatten(cls)[0].values()))
    if class_name not in self.handler.built.keys():
      self.handler.rebuild_function(cls)
    fields = ", ".join(self.handler.flatten(cls)[0])
    # lambda_for_create_host_objects = eval("lambda {}: func(cls({}))".format(fields, fields), locals())
    lambda_for_create_host_objects = eval("lambda {}: func(__rebuild_{}({}))".format(fields, class_name, fields), locals())
    lambda_for_callback = ffi.callback(callback_types, lambda_for_create_host_objects)
"""