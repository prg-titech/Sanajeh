class RuntimeExpander:
  built = {}
  flattened = {}
  
  def __init__(self):
    pass

  def build_function(self, cls):
    module = cls.__dict__["__module__"]
    name = cls.__name__
    args = {}
    func = "\t" + "new_object = cls.__new__(cls)\n"
    for field, ftype in cls.__dict__["__annotations__"].items():
      if field.split("_")[-1] != "REF" and ftype not in ["int", "float", "bool"]:
        if ftype not in self.built.keys():
          self.build_function(getattr(__import__(module), ftype))
        nested_args = []
        for nested_field, nested_ftype in self.flattened[ftype].items():
          nested_args.append(field + "_" + nested_field)
          args[field + "_" + nested_field] = nested_ftype
        func += "\t" + "new_object.{} = getattr(__import__(cls.__dict__[\"__module__\"]), \"{}\")({})\n".format(field, ftype, ", ".join(nested_args))
      else:
        args[field] = ftype
        func += "\t" + "new_object.{} = {}\n".format(field, field)
    func += "\t" + "return new_object"
    func = "@classmethod\n" \
           + "def __rebuild_{}(cls, {}):\n".format(name, ", ".join(args)) + func
    self.built[name] = func
    self.flattened[name] = args
    getattr(__import__(module), "Vector")(1,2)
    exec(func, globals())
    setattr(cls, "__rebuild_{}".format(name), eval("__rebuild_{}".format(name)))