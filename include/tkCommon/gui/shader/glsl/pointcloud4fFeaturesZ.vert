#version 330 core

layout (location = 0) in vec4 point;

uniform mat4 modelview;

uniform int axis;

out float feature;

void main(){

	// write point
	if(axis < 2)
    	feature    		= point.x;
	if(axis == 2)
    	feature    		= point.y;
	if(axis > 2)
    	feature    		= point.z;
	gl_Position 	= modelview * vec4(point.xyz, 1.0);
}
