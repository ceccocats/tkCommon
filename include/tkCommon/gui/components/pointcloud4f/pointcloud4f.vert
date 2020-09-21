#version 330 core

layout (location = 0) in vec4 point;

uniform mat4 modelview;

void main(){

	// write point
	gl_Position 	= modelview * vec4(point.xyz, 1.0);
}
