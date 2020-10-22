
import re

PLACEHOLDER = re.compile('\\$([^\\$]+)\\$')

class Snippet:
	last = None
	def __init__(self, owner, text, postfix):
		self.owner = owner
		if self.owner.last is not None:
			with self.owner.last:
				pass
		self.owner.write("".join(text))
		self.owner.last = self
		self.postfix = postfix
		
	def __enter__(self):
		self.owner.write("{")
		self.owner.current_indent += 1
		self.owner.last = None
		
	def __exit__(self, a, b, c):
		if self.owner.last is not None:
			with self.owner.last:
				pass
		self.owner.current_indent -= 1
		self.owner.write("}" + self.postfix)
		
class Subs:
	def __init__(self, owner, subs):
		self.owner = owner
		self.subs = subs
		
	def __enter__(self):
		self.owner.substack = [self.subs] + self.owner.substack
		
	def __exit__(self, a, b, c):
		self.owner.substack = self.owner.substack[1:]
		

class CodeFile:
	def __init__(self, filename):
		self.current_indent = 0
		self.last = None
		self.out = open(filename,"w")
		self.indent = "    "
		self.substack = []
		
	def close(self):
		self.out.close()
		self.out = None
	
	def write(self, x, indent=0):
		self.out.write(self.indent * (self.current_indent+indent) + x + "\n")
		
	def format(self, text):
		while True:
			m = PLACEHOLDER.search(text)
			if m is None:
				return text
			s = None
			for sub in self.substack:
				if m.group(1) in sub:
					s = sub[m.group(1)]
					break
			if s is None:
				raise Exception("Substitution '%s' not set." % m.groups(1))
			text = text[:m.start()] + str(s) + text[m.end():]		
		
	def subs(self, **subs):
		return Subs(self, subs)
		
	def __call__(self, text):
		self.write(self.format(text))
		
	def block(self, text, postfix=""):
		return Snippet(self, self.format(text), postfix)

class CppFile(CodeFile):
	def __init__(self, filename):
		CodeFile.__init__(self, filename)
		
	def label(self, text):
		self.write(self.format(text) + ":", -1)
		

def isVarConst(name):
	return name.startswith("const") or name.startswith("typedef") or name.startswith("static")
	
def isVarStatic(name):
	return name.startswith("static")
	

def isVarSTD(name):
	return name.startswith("std::") or name.startswith("struct")


def genData(className, VARS, DEPS = []):
	print("[....] Generating", className)
	cpp = CppFile(className + ".h")
	cpp("// this file is generated DO NOT DIRECTLY MODIFY")
	cpp("#pragma once")
	cpp("#include \"tkCommon/data/SensorData.h\"")
	for d in DEPS:
		cpp(d)
	cpp("")
	cpp("namespace tk { namespace data {\n")

	with cpp.subs(ClassName=className):
		with cpp.block("class $ClassName$ : public SensorData", ";"):
			cpp.label("public")
			for var in VARS:
				with cpp.subs(type=var["type"], var=var["name"]):
					cpp("$type$ $var$;")
			cpp("")
			
			with cpp.block("void init() override"):
				cpp("SensorData::init();")
				for var in VARS:
					if("init" in var):
						with cpp.subs(init=var["init"]):
							cpp("$init$;")
					if(isVarStatic(var["type"])):
						continue
					if("default" in var):
						with cpp.subs(type=var["type"], var=var["name"], default=var["default"]):
							cpp("$var$ = $default$;")

			with cpp.block("$ClassName$& operator=(const $ClassName$& s)"):
				cpp("SensorData::operator=(s);")
				for var in VARS:
					if isVarConst(var["type"]):
						continue
					with cpp.subs(var=var["name"]):
						cpp("$var$ = s.$var$;")
				cpp("return *this;")

			with cpp.block("friend std::ostream& operator<<(std::ostream& os, $ClassName$& s)"):
				cpp("os<<\"$ClassName$\"<<std::endl;")
				cpp("os<<\"\theader.name:  \"<<s.header.name<<std::endl;")
				cpp("os<<\"\theader.stamp: \"<<s.header.stamp<<std::endl;")
				#cpp("SensorData::operator<<(s);")
				for var in VARS:
					if isVarConst(var["type"]) or isVarSTD(var["type"]):
						continue
					with cpp.subs(var=var["name"]):
						cpp("os<<\"\t$var$: \"<<s.$var$<<std::endl;")
				cpp("return os;")

			MATVARS = []
			for v in VARS:
				if isVarConst(v["type"]) or isVarSTD(v["type"]):
					continue
				MATVARS.append(v)

			with cpp.block("bool toVar(std::string name, tk::math::MatIO::var_t &var)"):
				with cpp.subs(nvars=len(MATVARS)+1):
					cpp("std::vector<tk::math::MatIO::var_t> structVars($nvars$);")
				cpp("structVars[0].set(\"header\", header);")
				for i in range(len(MATVARS)):
					with cpp.subs(i=i+1, var=MATVARS[i]["name"]):
						cpp("structVars[$i$].set(\"$var$\", $var$);")
				cpp("return var.setStruct(name, structVars);")

			with cpp.block("bool fromVar(tk::math::MatIO::var_t &var)"):
				cpp("if(var.empty()) return false;")
				cpp("var[\"header\"].get(header);")
				for var in MATVARS:
					with cpp.subs(var=var["name"]):
						cpp("var[\"$var$\"].get($var$);")
				cpp("return true;")


		cpp("")
		for var in VARS:
			if(isVarStatic(var["type"]) and "default" in var):
				tp = (var["type"][len("static"):]).split(" ")
				tp.insert(-1, " "+className +"::")
				tp = "".join(tp)
				with cpp.subs(type=tp, var=var["name"], default=var["default"]):
					cpp("$type$ $ClassName$::$var$ = $default$;")
		cpp("")
	cpp("\n}}")

	cpp.close()

