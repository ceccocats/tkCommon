#version 330 core
layout (location = 0) in vec3 point;
layout (location = 1) in vec3 normal;

out vec3 	Normal;  
out vec3 	FragPos;

uniform mat4 	modelview;
uniform bool 	useLight;
uniform vec3	lightPos;

void main() {
	FragPos		= point;
	Normal 		= normal;
	gl_Position = modelview * vec4(point, 1.0);
}