import ast, astunparse
import copy
import os, sys
from pathlib import Path
from pprint import pprint

"""
Additional info stored on class:
class.__sanajeh_enum__: int, class type as int
class.__sanajeh_trueparents__: int[], list of MI parents, list based on c3 linearization
class.__sanajeh_truechildren__: int[], list of MI childs

"""

def Append( lst, item ):
	if item not in lst:
		lst.append( item )
def Extend( lst1, lst2 ):
	lst = [x for x in lst2 if x not in lst1]
	lst1.extend(lst)

class FunctionData:
	def __init__(self):
		self.name = ""
		self.astNode = None
		self.ownerClass = None

	def printSelf(self,indent=0):
		print("\t"*indent + "Function {}: {}".format(self.name,""))
		print("\t"*indent + "\towner: {}, {}".format(self.ownerClass.name,""))

class VariableData:
	def __init__(self):
		self.name = ""
		self.astNode = None
		self.type = None
		self.ownerClass = None

	def printSelf(self,indent=0):
		print("\t"*indent + "Variable {}: {}".format(self.name,""))
		print("\t"*indent + "\towner: {}, {}".format(self.ownerClass.name,""))
		print("\t"*indent + "\ttype: {}".format(self.type))

class ClassData:
	def __init__(self):
		self.name = ""
		self.astNode = None
		self.enum = 0

		# original relationship, order matters.
		# left-most should be primary parent
		self.parents = []
		self.parents_name = []
		self.children = []

		# only in the class declaration
		self.declaredFuncs = []
		self.declaredVars = []

		# only in closest parents' declaration, including overridden ones
		self.inheritedFuncs = []
		self.inheritedVars = []

		# flags
		self.isCreated = False
		self.isTyped = False
		self.isParentCreated = False
		self.isParentTyped = False

		# single-inheritance relationship
		# semiparents/childs are parents/childs that will be severed
		# mergeable parents: if class or its ancestor is not instantiated nor typed.
		# applied mergeable: the mergeable class this one will be applied to, starting from the closest
		self.primeParent = None
		self.semiParents = []
		self.semiChildren = []
		self.mergeableParents = []
		self.appliedMergeableParents = []

		# methods that is directly inherited from semiparents
		# meaning: all semiparents (including semiancestor) methods that is not overridden
		# will be flattened as is to the class body
		self.semiInheritedFuncs = []
		self.semiInheritedVars = []

		# methods owned for each of the semiancestors, including overrides, may be a duplicate from above
		# will be flattened as classname_methodname
		self.semiInheritedAllFuncs = []

	def MarkAsCreated(self):
		if not self.isCreated:
			self.isCreated = True
			print("Mark as created {}".format(self.name))

	def MarkAsTyped(self):
		if not self.isTyped:
			self.isTyped = True
			print("Mark as typed {}".format(self.name))

	def IsMergeable(self):
		return not (self.isCreated or self.isTyped or self.isParentCreated or self.isParentTyped)

	def IsChildOf(self,classData):
		if classData in self.parents:
			return True
		return False

	def IsParentOf(self,classData):
		return classData.IsChildOf(self)

	def IsDescendantOf(self,classData):
		for cls in self.parents:
			if cls is classData:
				return True
			if cls.IsDescendantOf(classData):
				return True
		return False

	def GetAncestors(self,ancestors=[]):
		for parent in self.parents:
			if parent not in ancestors:
				ancestors.append(parent)
		for parent in self.parents:
			ancestors.extend(parent.GetAncestors(ancestors))
		return ancestors

	def printSelf(self):
		print("Class {}: {}".format(self.name,""))
		print("\tParents: {}".format([x.name for x in self.parents]))
		print("\tChildren: {}".format([x.name for x in self.children]))
		print("\tPrimary Parent: {}".format(getattr(self.primeParent, 'name', 'None')))
		print("\tSemi Parents: {}".format([x.name for x in self.semiParents]))
		print("\tMergeable Parents: {}".format([x.name for x in self.mergeableParents]))
		print("\tSemi Children: {}".format([x.name for x in self.semiChildren]))
		print("\tDeclared Funcs:")
		for func in self.declaredFuncs:
			func.printSelf(2)
		print("\tDeclared Vars:")
		for func in self.declaredVars:
			func.printSelf(2)
		print("\tInherited Funcs:")
		for func in self.inheritedFuncs:
			func.printSelf(2)
		print("\tInherited Vars:")
		for func in self.inheritedVars:
			func.printSelf(2)
		print("\tisCreated {}, isParentCreated {}, isTyped {}, isParentTyped {}".format(self.isCreated,self.isParentCreated,self.isTyped,self.isParentTyped))
		print("\tSemi-Inherited Vars:")
		for func in self.semiInheritedVars:
			func.printSelf(2)
		print("\tSemi-inherited Funcs:")
		for func in self.semiInheritedFuncs:
			func.printSelf(2)
		print("\tSemi-inherited All Funcs:")
		for func in self.semiInheritedAllFuncs:
			func.printSelf(2)

class ClassDataGraph:
	def __init__(self):
		self.classNodes = []
		self.moduleNode = None

	def GetClassData(self,name):
		return next( (x for x in self.classNodes if x.name==name) , None )

	def InspectReference(self):
		for cls in self.classNodes:
			# nodes with name in parents_name
			parentClass = [self.GetClassData(x) for x in cls.parents_name ]
			cls.parents = parentClass
			for parent in cls.parents:
				# add children ref
				if cls not in parent.children:
					parent.children.append(cls)

	def InspectOverrides(self):
		# 1. traverse for inherited func and vars
		grandparents = [x for x in self.classNodes if len(x.parents)==0]

		# dfs traversal, naive approach (with duplicate traversal)
		for cls in grandparents:
			for child in cls.children:
				self.InspectOverridesRecursive(child)

	def InspectOverridesRecursive(self,cls):
		for parent in cls.parents:
			# inherit declared definitions
			for func in parent.declaredFuncs:
				if func not in cls.inheritedFuncs:
					cls.inheritedFuncs.append(func)
			for var in parent.declaredVars:
				if var not in cls.inheritedVars:
					cls.inheritedVars.append(var)

			# inherit inherited definitions
			for func in parent.inheritedFuncs:
				if func not in cls.inheritedFuncs:
					cls.inheritedFuncs.append(func)
			for var in parent.inheritedVars:
				if var not in cls.inheritedVars:
					cls.inheritedVars.append(var)

		for child in cls.children:
			self.InspectOverridesRecursive(child)

	def InspectLabeling(self):
		# start from grandparents
		grandparents = [x for x in self.classNodes if len(x.parents)==0]

		# label down
		for cls in grandparents:
			self.InspectLabelingRecursive(cls)

	def InspectLabelingRecursive(self,cls):
		if cls.isCreated:
			cls.isParentCreated = True
		if cls.isTyped:
			cls.isParentTyped = True

		for child in cls.children:
			if cls.isCreated or cls.isParentCreated:
				child.isParentCreated = True

			if cls.isTyped or cls.isParentTyped:
				child.isParentTyped = True
			
			self.InspectLabelingRecursive(child)

	def InspectParents(self):
		for cls in self.classNodes:

			# not counting grandparents
			if len(cls.parents)==0:
				continue

			# set primary parents
			print("set primary parent {}: {}".format(cls.name, cls.parents[0]))
			cls.primeParent = cls.parents[0]

			for parent in cls.parents[1:]:
				# set mergeable parents
				if parent.IsMergeable():
					cls.mergeableParents.append(parent)

				# set secondary parents
				else:
					cls.semiParents.append(parent)
					if cls not in parent.semiChildren:
						parent.semiChildren.append(cls)

		# determine which mergeable class merged to where
		grandparents = [x for x in self.classNodes if len(x.parents)==0]
		for cls in grandparents:
			if cls.IsMergeable():
				continue
			merged = []
			self.InspectAppliedMergeable(cls,merged)

		for cls in grandparents:
			if cls.IsMergeable():
				continue
			self.LinearizeMergeable(cls)

	def InspectAppliedMergeable(self,cls,merged):
		print("inspect",cls.name)
		mergeables = cls.mergeableParents
		#TODO: consider parents of mergeable classes too
		notYetMerged = [x for x in cls.mergeableParents if x not in merged]
		Extend(cls.appliedMergeableParents, notYetMerged)

		for child in cls.children:
			Extend(merged,notYetMerged)
			self.InspectAppliedMergeable(child,merged)

	def LinearizeMergeable(self,cls):
		for child in cls.children:
			# Linearize mergeable parents
			reverse = child.appliedMergeableParents[::-1]
			currentParent = cls
			for mergeable in reverse:
				mergeable.primeParent = currentParent
				currentParent = mergeable
			child.primeParent = currentParent
			self.LinearizeMergeable(child)

	def ApplyPrimeParent(self):
		# TODO: Put this logic on better location
		# check class definition location, rearrange if necessary
		body = self.moduleNode.body

		# # Create base class
		# baseClassNode = ast.ClassDef(
		# 	name='BaseClass',
		# 	bases=[],
		# 	keywords=[],
		# 	body=[
		# 		ast.FunctionDef(
		# 			name='__init__',
		# 			args=[
		# 				ast.Name(id='self',ctx=ast.Load()),
		# 			],
		# 			body=[
		# 				ast.AnnAssign(
		# 					target=ast.Attribute(
		# 				        value=ast.Name(id='self', ctx=ast.Load()),
		# 				        attr='class_type',
		# 				        ctx=ast.Store()
		# 				    ),
		# 					annotation=ast.Name(id='int', ctx=ast.Load()),
		# 					value=ast.Constant(value=0,kind=''),
		# 					simple=0
		# 				)
		# 			],
		# 			decorator_list=[],
		# 			returns=[],
		# 		),
		# 		ast.FunctionDef(
		# 			name='BaseClass',
		# 			args=[
		# 				ast.Name(id='self',ctx=ast.Load()),
		# 			],
		# 			body=[
		# 				ast.Pass(),
		# 			],
		# 			decorator_list=[],
		# 			returns=[],
		# 		),
		# 	],
		# 	decorator_list=[],
		# 	type_ignores=[]
		# )

		# for node in body:
		# 	if type(node)==ast.ClassDef:
		# 		index = body.index(node)
		# 		body.insert(index,baseClassNode)
		# 		break

		# Transform AST so that it has prime parent as parent 
		for cls in self.classNodes:
			node = cls.astNode
			if cls.primeParent is None:
				pass
				# node.bases = [ast.Name(id='BaseClass',ctx=ast.Load())]
			else:
				node.bases = [ast.Name(id=cls.primeParent.name,ctx=ast.Load())]

				clsIdx = body.index(cls.astNode)
				prtIdx = body.index(cls.primeParent.astNode)
				if clsIdx<prtIdx:
					# move cls definition below parent definition 
					body.insert(clsIdx, body.pop(prtIdx))

		ast.fix_missing_locations(self.moduleNode)

	###############################
	def InspectFlatten(self):
		for cls in self.classNodes:
			# semi parent and its ancestor's list
			allSemiAncestors = []
			allSemiAncestors.extend(cls.semiParents)
			for semi in cls.semiParents:
				allSemiAncestors.extend(semi.GetAncestors())

			# get all funcs from all semiancestors
			cls.semiInheritedAllFuncs = [x for x in cls.inheritedFuncs if x.ownerClass in allSemiAncestors]

			# get all inherited non-overridden funcs
			uniqueInherit = []
			uniqueInheritNames = []
			declaredFuncNames = [x.name for x in cls.declaredFuncs]
			inheritedFuncNames = [x.name for x in cls.inheritedFuncs]
			for func in cls.inheritedFuncs:
				if func.ownerClass not in allSemiAncestors:
					pass
				elif func.name in declaredFuncNames:
					pass
				elif func.name in uniqueInheritNames:
					pass
				else:
					uniqueInherit.append(func)
					uniqueInheritNames.append(func.name)
			cls.semiInheritedFuncs = uniqueInherit

			# get all inherited non-overridden vars
			uniqueInherit = []
			uniqueInheritNames = []
			declaredVarNames = [x.name for x in cls.declaredVars]
			inheritedVarNames = [x.name for x in cls.inheritedVars]
			for var in cls.inheritedVars:
				if var.ownerClass not in allSemiAncestors:
					pass
				elif var.name in declaredVarNames:
					pass
				elif var.name in uniqueInheritNames:
					pass
				else:
					uniqueInherit.append(var)
					uniqueInheritNames.append(var.name)
			cls.semiInheritedVars = uniqueInherit

	def PrintResult(self):
		for cls in self.classNodes:
			cls.printSelf()

class FirstVisitInspector(ast.NodeVisitor):
	def __init__(self,classNodes):
		self.classNodes = classNodes
		self.visitorPath = []
		self.nodePath = []
		self.isInit = False
		self.moduleNode = None

	def visit_Module(self,node):
		# visit the ast
		# TODO: Put GetModule logic in proper location instead
		self.moduleNode = node
		self.visitorPath.append(node)
		self.generic_visit(node)
		self.visitorPath.pop()
		# visit finished

	def visit_ClassDef(self,node):
		# no recursive check
		prevNode = self.visitorPath[-1]
		if type(prevNode) is not ast.ClassDef and type(prevNode) is not ast.Module:
			msg = "Unsupported nested class definition: {}".format(type(prevNode))
			self.Exception(node,msg)

		# no duplicate check

		classData = ClassData()
		classData.name = node.name
		classData.parents_name = [x.id for x in node.bases]
		classData.astNode = node

		self.classNodes.append(classData)

		self.nodePath.append(classData)
		self.visitorPath.append(node)
		ast.NodeVisitor.generic_visit(self, node)
		self.visitorPath.pop()
		self.nodePath.pop()

	def visit_FunctionDef(self,node):
		# check if it is inside class def
		prevNode = self.visitorPath[-1]
		if type(prevNode) is not ast.ClassDef:
			return

		ownerClass = self.nodePath[-1]

		funcData = FunctionData()
		funcData.name = node.name
		funcData.astNode = node
		funcData.ownerClass = ownerClass

		ownerClass.declaredFuncs.append(funcData)

		# visit __init__ definition
		if node.name == '__init__':
			self.isInit = True
			self.visitorPath.append(node)
			ast.NodeVisitor.generic_visit(self, node)
			self.visitorPath.pop()
			self.isInit = False

	def visit_AnnAssign(self,node):
		if not self.isInit:
			return
		# check if it is only self
		if type(node.target) is not ast.Attribute:
			msg = "Only field declaration here: must be attribute: {}".format(type(node.target))
			self.Exception(node,msg)
		if node.target.value is not ast.Name and node.target.value.id != "self":
			msg = "Only field declaration here: must be self: {}".format(node.target.value.id)
			self.Exception(node,msg)

		ownerClass = self.nodePath[-1]

		varData = VariableData()
		varData.name = node.target.attr
		varData.astNode = node
		varData.ownerClass = ownerClass

		# check if type is subscript
		if type(node.annotation)==ast.Subscript:

			if type(node.annotation.slice) is not ast.Index:
				self.Exception(node,"not ast.Index {}".format(type(node.annotation.slice)))

			baseType = node.annotation.value.id
			sliceType = node.annotation.slice.value.id
			varData.type = baseType + "_" + sliceType

		elif type(node.annotation)==ast.Name:	
			varData.type = node.annotation.id
		else:
			self.Exception(node,"not subscript")

		ownerClass.declaredVars.append(varData)

	def Exception(self,node,msg="Unsupported"):
		print("ERROR: " + msg)
		line = -1
		col = -1
		if hasattr(node,"lineno"):
			line = node.lineno
		if hasattr(node,"col_offset"):
			col = node.col_offset
		print("In line {}:{}".format(line,col))
		sys.exit(1)

# find type references
class SecondVisitInspector(ast.NodeVisitor):
	def __init__(self,classNodes):
		self.classNodes = classNodes
		self.classNames = [x.name for x in classNodes]
		self.visitorPath = []
		self.nodePath = []
		self.isAnnAssign = False

	def GetClassData(self,name):
		return next( (x for x in self.classNodes if x.name==name) , None )

	def visit_Module(self,node):
		# visit the ast
		self.visitorPath.append(node)
		self.generic_visit(node)
		self.visitorPath.pop()
		# visit finished

	def visit_AnnAssign(self,node):
		baseType = None # int, classname, etc.
		subType = None # list, ref, etc.

		# check class is used as type
		if type(node.annotation)==ast.Subscript:

			if type(node.annotation.slice) is not ast.Index:
				self.Exception(node,"not ast.Index {}".format(type(node.annotation.slice)))

			subType = node.annotation.value.id
			baseType = node.annotation.slice.value.id

		elif type(node.annotation)==ast.Name:
			baseType = node.annotation.id

		if baseType in self.classNames:
			# mark as typed
			classData = self.GetClassData(baseType)
			classData.MarkAsTyped()

		ast.NodeVisitor.generic_visit(self, node)

	def visit_Call(self,node):
		# check if it is constructor: function with class name as its name
		if type(node.func) is ast.Name:
			if node.func.id in self.classNames:
				# mark as instantiated
				classData = self.GetClassData(node.func.id)
				classData.MarkAsCreated()

		funcName = None
		funcOwner = None
		if type(node.func) is ast.Attribute:
			funcName = node.func.attr

			if type(node.func.value) is ast.Name:
				funcOwner = node.func.value.id

		# check if it is parallel_new (for now, it is <somename>.parallel_new)
		if funcName=='parallel_new':
			# get 1st arg
			className = node.args[0].id

			# mark as instantiated
			classData = self.GetClassData(className)
			classData.MarkAsCreated()

		# check if it is DeviceAllocator.new
		if funcOwner=="DeviceAllocator" and funcName=="new":
			# get 1st arg
			className = node.args[0].id

			# mark as instantiated
			classData = self.GetClassData(className)
			classData.MarkAsCreated()

		ast.NodeVisitor.generic_visit(self, node)

	def visit_arg(self,node):
		baseType = None # int, classname, etc.
		subType = None # list, ref, etc.

		# check annotation
		if type(node.annotation)==ast.Subscript:

			if type(node.annotation.slice) is not ast.Index:
				self.Exception(node,"not ast.Index {}".format(type(node.annotation.slice)))

			subType = node.annotation.value.id
			baseType = node.annotation.slice.value.id

		elif type(node.annotation)==ast.Name:
			baseType = node.annotation.id

		if baseType in self.classNames:
			# mark as typed
			classData = self.GetClassData(baseType)
			classData.MarkAsTyped()

	def Exception(self,node,msg="Unsupported"):
		print("ERROR: " + msg)
		line = -1
		col = -1
		if hasattr(node,"lineno"):
			line = node.lineno
		if hasattr(node,"col_offset"):
			col = node.col_offset
		print("In line {}:{}".format(line,col))
		sys.exit(1)

class ClassInspectorMain:
	def __init__(self,code):
		self.graph = ClassDataGraph()
		self.code = code
		self.ast_code = None

	def Start(self):
		ast_code = ast.parse(self.code)
		graph = self.graph

		# 1. Find class definitions
		fv = FirstVisitInspector(graph.classNodes)
		fv.visit(ast_code)
		graph.moduleNode = fv.moduleNode

		# 2. Cross reference to get parent-children relationship
		graph.InspectReference()

		# 3. Check for functions and overridden functions
		graph.InspectOverrides()

		# 4. Find class instantiation and type reference
		SecondVisitInspector(graph.classNodes).visit(ast_code)

		# 5. Label classes for mergeable classes
		graph.InspectLabeling()

		# 6. Label parents for mergeable, primary, and semi
		graph.InspectParents()

		graph.ApplyPrimeParent()

		# graph.PrintResult()
		print(astunparse.unparse(ast_code))

		self.ast_code = ast_code
