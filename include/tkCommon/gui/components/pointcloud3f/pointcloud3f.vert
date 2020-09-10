#version 330 core

layout (location = 0) in vec3 point;

uniform mat4 modelview;

void main(){

	// coordinates
	vec4 coordinate = vec4(point.xyz, 1.0);

	// write point
	gl_Position 	= modelview * coordinate;
}
