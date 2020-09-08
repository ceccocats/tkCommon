#version 330 core

layout (location = 0) in vec4 point;

uniform mat4 modelview;

out vec4 color;

void main(){

	// coordinates
	vec4 coordinate = vec4(point.x, point.y, point.z, 1.0);

	// write point
	gl_Position 	= modelview * coordinate;
}
