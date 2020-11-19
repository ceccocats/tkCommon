from os import listdir
from os.path import isfile, join
import sys

filename = "Drawables.h"
f = open(filename, "w")
f.write("// this file is generated DO NOT DIRECTLY MODIFY")
for file in [s for s in listdir(".") if isfile(join(".", s))]:
	if(file != sys.argv[0] and file != filename):
		f.write("\n#include <tkCommon/gui/Drawables/"+file+">")
f.close()