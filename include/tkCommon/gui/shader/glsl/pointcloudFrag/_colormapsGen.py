def createFragShaderuniformColor(filename):
	f = open(filename+"_uniformColor.frag", "w")
	f.write("// this file is generated DO NOT DIRECTLY MODIFY\n")
	f.write(
		"#version 330 core\n\n"+

		"out vec4 FragColor;\n\n"+

		"uniform vec4 color;\n\n"+

		"void main(){\n"+
		"\tFragColor\t= color;\n"+
		"}\n"
	)
	f.close()
	print("[....] Generating "+filename+"_uniformColor.frag")

def createFragShader(filename, header):
	f = open(filename, "w")
	f.write("// this file is generated DO NOT DIRECTLY MODIFY\n")
	f.write(
		"#version 330 core\n\n"+

		"#include \""+header+"\"\n\n"+

		"out vec4 FragColor;\n\n"+

		"in float feature;\n\n"+

		"uniform float alpha;\n"+
		"uniform float minFeature;\n"+
		"uniform float maxFeature;\n\n"+

		"void main(){\n"+
		"\tfloat value\t= (feature - minFeature) / (maxFeature - minFeature);\n"+
		"\tFragColor\t= colormap(value);\n"+
		"\tFragColor.a\t= alpha;\n"+
		"}\n"
	)
	f.close()
	print("[....] Generating "+filename)

path = "../colormaps/"
filename = "pointcloudFrag"

from os import listdir
from os.path import isfile, join
files = [f for f in listdir(path) if isfile(join(path, f))]

createFragShaderuniformColor(filename)
for file in files:
	createFragShader(filename+"_"+file,path+file)