
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
		

def genData(className, VARS, DEPS = []):
	print("GENERATE: ", className)
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
				if("default" not in var):
					with cpp.subs(type=var["type"], var=var["name"]):
						cpp("$type$ $var$;")
				else:
					with cpp.subs(type=var["type"], var=var["name"], default=var["default"]):
						cpp("$type$ $var$ = $default$;")
			cpp("")

			with cpp.block("void init() override"):
				cpp("SensorData::init();")
				for var in VARS:
					if "init" not in var:
						continue
					with cpp.subs(init=var["init"]):
						cpp("$init$;")

			#with cpp.block("$ClassName$& operator=(const $ClassName$& s)"):
			#	cpp("SensorData::operator=(s);")
			#	for var in VARS:
			#		with cpp.subs(var=var["name"]):
			#			cpp("$var$ = s.$var$;")
			#	cpp("return *this;")

			#with cpp.block("friend std::ostream& operator<<(std::ostream& os, const $ClassName$& s)"):
			#	cpp("os<<\"$ClassName$:\"<<std::endl;")
			#	#cpp("SensorData::operator<<(s);")
			#	for var in VARS:
			#		with cpp.subs(var=var["name"]):
			#			cpp("os<<\"$var$: \"<<s.$var$<<std::endl;")
			#	cpp("return os;")

			#with cpp.block("bool toVar(std::string name, tk::math::MatIO::var_t &var)"):
			#	cpp("tk::math::MatIO::var_t hvar;")
			#	cpp("tk::data::SensorData::toVar(\"header\", hvar);")
			#	with cpp.subs(nvars=len(VARS)+1):
			#		cpp("std::vector<tk::math::MatIO::var_t> structVars($nvars$);")
			#	cpp("structVars[0] = hvar;")
			#	for i in range(len(VARS)):
			#		with cpp.subs(i=i+1, var=VARS[i]["name"]):
			#			cpp("structVars[$i$].set(\"$var$\", $var$);")
			#	cpp("return var.setStruct(name, structVars);")

			#with cpp.block("bool fromVar(tk::math::MatIO::var_t &var)"):
			#	cpp("if(var.empty()) return false;")
			#	cpp("tk::data::SensorData::fromVar(var[\"header\"]);")
			#	for var in VARS:
			#		with cpp.subs(var=var["name"]):
			#			cpp("var[\"$var$\"].get($var$);")
			#	cpp("return true;")
	cpp("\n}}")

	cpp.close()

