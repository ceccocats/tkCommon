#version 330 core
layout (location = 0) in vec3 point;
layout (location = 1) in vec4 pointColor;

uniform mat4 modelview;

out vec4 color;

void main(){

    color = pointColor;

	gl_Position 	= modelview * vec4(point.xyz, 1.0);
}
