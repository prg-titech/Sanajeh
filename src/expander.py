import inspect
import ast

class RuntimeExpander:
  built = {}
  flattened = {}
  
  def __init__(self):
    pass

  def get_init(self, cls):
    init_lines = inspect.getsourcelines(cls.__init__)[0]
    result = ""
    for i in range(len(init_lines)):
      if i != 0:
        result += "\t"
      result += init_lines[i].lstrip()
    return ast.parse(result).body[0]

  def get_dict(self, init):
    result = {}
    for body in init.body:
      if type(body) == ast.AnnAssign \
      and hasattr(body.target.value, "id") and body.target.value.id == "self":
        result[body.target.attr] = body.annotation.id    
    return result

  def build_function(self, cls):
    module = cls.__dict__["__module__"]
    name = cls.__name__
    args = {}
    func = "\t" + "new_object = cls.__new__(cls)\n"
    field_dict = self.get_dict(self.get_init(cls))
    for field, ftype in field_dict.items():
      if field.split("_")[-1] != "ref" and ftype not in ["int", "float", "bool"]:
        if ftype not in self.built.keys():
          self.build_function(getattr(__import__(module), ftype))
        nested_args = []
        for nested_field, nested_ftype in self.flattened[ftype].items():
          nested_args.append(field + "_" + nested_field)
          args[field + "_" + nested_field] = nested_ftype
        func += "\t" + "new_object.{} = getattr(__import__(cls.__dict__[\"__module__\"]), \"{}\")({})\n".format(field, ftype, ", ".join(nested_args))
      else:
        if field.split("_")[-1] == "ref":
          args[field] = "int"
        else:
          args[field] = ftype
        func += "\t" + "new_object.{} = {}\n".format(field, field)
    func += "\t" + "return new_object"
    func = "@classmethod\n" \
           + "def __rebuild_{}(cls, {}):\n".format(name, ", ".join(args)) + func
    self.built[name] = func
    self.flattened[name] = args
    exec(func, globals())
    setattr(cls, "__rebuild_{}".format(name), eval("__rebuild_{}".format(name)))
    